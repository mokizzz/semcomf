import glob
import os
import sys

import cv2
import kagglehub
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from tqdm import tqdm

from semcomf import (
    AWGNChannel,
    BasePipeline,
    IdealChannel,
    SimpleVideoDecoder,
    SimpleVideoEncoder,
)
from semcomf.metrics.image_metrics import (
    calculate_lpips,
    calculate_ms_ssim,
    calculate_psnr,
    calculate_ssim,
)


class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, video_size=(256, 256)):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.video_size = video_size
        self.video_files = glob.glob(os.path.join(root_dir, "*.avi"))
        logger.info(f"Found {len(self.video_files)} video files in {root_dir}")

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []

        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = TF.resize(frame, self.video_size)
            frames.append(TF.pil_to_tensor(frame))

        cap.release()

        if not frames:
            logger.info(
                f"Warning: Video {video_path} is too short or empty. Returning zero tensor."
            )
            return torch.zeros(
                self.num_frames,
                3,
                self.video_size[0],
                self.video_size[1],
                dtype=torch.float32,
            )

        video_clip = torch.stack(frames).float() / 255.0

        if video_clip.shape[0] < self.num_frames:
            padding = torch.zeros(
                self.num_frames - video_clip.shape[0],
                *video_clip.shape[1:],
                dtype=video_clip.dtype,
            )
            video_clip = torch.cat((video_clip, padding), dim=0)

        return video_clip


if __name__ == "__main__":
    logger.add(
        "outputs/{time:YYYY-MM-DD}_{time:HH-mm-ss}.log",
        rotation="00:00",
        retention=30,
        level="INFO",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    path = kagglehub.dataset_download("aryashah2k/highway-traffic-videos-dataset")
    path = os.path.join(path, "video")
    logger.info("Path to dataset files:", path)

    video_size = (256, 256)
    num_frames_per_clip = 16
    batch_size = 4

    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")

    train_dataset = VideoDataset(
        root_dir=path, num_frames=num_frames_per_clip, video_size=video_size
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = VideoDataset(
        root_dir=path, num_frames=num_frames_per_clip, video_size=video_size
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    sys.stderr.close()
    sys.stderr = original_stderr

    transmitter = SimpleVideoEncoder(input_channels=3).to(device)
    channel = IdealChannel().to(device)
    receiver = SimpleVideoDecoder(output_channels=3).to(device)
    pipeline = BasePipeline(transmitter, channel, receiver).to(device)
    optimizer = torch.optim.Adam(pipeline.parameters(), lr=1e-4)

    num_epochs = 10

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        pipeline.train()
        pbar = tqdm(train_loader, desc="Processing video clips (Train)")

        for video_clips in pbar:
            video_clips = video_clips.to(device)
            B, T, C, H, W = video_clips.shape

            total_loss = 0
            enc_hidden_state = None
            dec_hidden_state = None
            prev_recon_frame = torch.zeros(B, C, H, W, device=device)

            optimizer.zero_grad()

            for t in range(T):
                current_frame = video_clips[:, t, :, :, :]

                if t == 0:
                    features_to_transmit, enc_hidden_state = pipeline.transmitter(
                        current_frame, None, enc_hidden_state
                    )
                else:
                    features_to_transmit, enc_hidden_state = pipeline.transmitter(
                        current_frame, prev_recon_frame.detach(), enc_hidden_state
                    )

                received_features, _ = pipeline.channel(features_to_transmit)

                recon_frame, dec_hidden_state = pipeline.receiver(
                    received_features, dec_hidden_state
                )

                loss = F.mse_loss(recon_frame, current_frame)
                total_loss += loss

                prev_recon_frame = recon_frame

            (total_loss / T).backward()
            optimizer.step()
            pbar.set_postfix({"loss": (total_loss / T).item()})

        pipeline.eval()

        val_pbar = tqdm(val_loader, desc="Validating video clips")

        with torch.no_grad():
            val_total_loss = 0.0
            val_psnr_total = 0.0
            val_ssim_total = 0.0
            val_ms_ssim_total = 0.0
            val_lpips_total = 0.0
            num_val_frames = 0

            for val_video_clips in val_pbar:
                val_video_clips = val_video_clips.to(device)
                B, T, C, H, W = val_video_clips.shape

                enc_hidden_state = None
                dec_hidden_state = None
                prev_recon_frame = torch.zeros(B, C, H, W, device=device)

                for t in range(T):
                    current_frame = val_video_clips[:, t, :, :, :]

                    if t == 0:
                        features_to_transmit, enc_hidden_state = pipeline.transmitter(
                            current_frame, None, enc_hidden_state
                        )
                    else:
                        features_to_transmit, enc_hidden_state = pipeline.transmitter(
                            current_frame, prev_recon_frame, enc_hidden_state
                        )

                    received_features, _ = pipeline.channel(features_to_transmit)
                    recon_frame, dec_hidden_state = pipeline.receiver(
                        received_features, dec_hidden_state
                    )

                    loss = F.mse_loss(recon_frame, current_frame)
                    val_total_loss += loss.item()

                    val_psnr_total += calculate_psnr(
                        current_frame, recon_frame, data_range=1.0
                    )
                    val_ssim_total += calculate_ssim(
                        current_frame, recon_frame, data_range=1.0
                    )
                    val_ms_ssim_total += calculate_ms_ssim(
                        current_frame, recon_frame, data_range=1.0
                    )
                    val_lpips_total += calculate_lpips(current_frame, recon_frame)
                    num_val_frames += B

                    prev_recon_frame = recon_frame

                val_pbar.set_postfix({"val_loss": (val_total_loss / T)})

            avg_val_loss = val_total_loss / (len(val_loader) * T)
            avg_val_psnr = val_psnr_total / num_val_frames
            avg_val_ssim = val_ssim_total / num_val_frames
            avg_val_ms_ssim = val_ms_ssim_total / num_val_frames
            avg_val_lpips = val_lpips_total / num_val_frames

            logger.info(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.6f}")
            logger.info(f"Epoch {epoch + 1} Average PSNR: {avg_val_psnr:.4f}")
            logger.info(f"Epoch {epoch + 1} Average SSIM: {avg_val_ssim:.4f}")
            logger.info(f"Epoch {epoch + 1} Average MS-SSIM: {avg_val_ms_ssim:.4f}")
            logger.info(f"Epoch {epoch + 1} Average LPIPS: {avg_val_lpips:.4f}")

            output_dir = "output_images"
            os.makedirs(output_dir, exist_ok=True)

            if epoch == 0:
                for i in range(min(1, val_video_clips.size(0))):
                    for j in range(min(4, val_video_clips.size(1))):
                        orig_frame_to_save = val_video_clips[i, j].cpu()
                        save_image(
                            orig_frame_to_save,
                            os.path.join(output_dir, f"orig_video_{i}_frame_{j}.png"),
                        )

            for i in range(min(1, recon_frame.size(0))):
                recon_frame_to_save = recon_frame[i].cpu()
                save_image(
                    recon_frame_to_save,
                    os.path.join(output_dir, f"recon_video_{i}_epoch_{epoch}.png"),
                )

        logger.info(f"Epoch {epoch + 1} completed.")
