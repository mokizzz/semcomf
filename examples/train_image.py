"""
python -m semcomf.examples.train
"""

import datasets
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from tqdm import tqdm

from semcomf import (
    AWGNChannel,
    BasePipeline,
    IdealChannel,
    SimpleImageDecoder,
    SimpleImageEncoder,
)


def get_datasets() -> tuple[Dataset, Dataset, Dataset]:
    ds = datasets.load_dataset("eugenesiow/Div2k", "bicubic_x4")
    return ds["train"], ds["validation"]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_set, val_set = get_datasets()
    print(f"Train dataset size: {len(train_set)}")
    print(f"Validation dataset size: {len(val_set)}")
    image_size = (256, 256)
    batch_size = 64

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    transmitter = SimpleImageEncoder(input_channels=3, image_size=image_size).to(device)
    # channel = AWGNChannel(snr_db=20.0, precision=torch.float16).to(device)
    channel = IdealChannel().to(device)
    receiver = SimpleImageDecoder(output_channels=3, output_size=image_size).to(device)
    pipeline = BasePipeline(transmitter, channel, receiver).to(device)
    optimizer = torch.optim.Adam(
        pipeline.parameters(), lr=1e-3
    )  # 1e-4 for slower better convergence

    for epoch in tqdm(range(10), desc="Epochs"):
        # Train
        pipeline.train()
        pbar = tqdm(train_loader, desc="Processing images")

        for data in pbar:
            image_paths = data["lr"]
            images = []
            for image_path in image_paths:
                image = Image.open(image_path).convert("RGB")
                image = image.resize(image_size, Image.BICUBIC)
                images.append(TF.pil_to_tensor(image))

            orig_tensor = torch.stack(images).to(device) / 255.0
            recon_tensor, diagnostics = pipeline(orig_tensor)

            loss = torch.nn.functional.mse_loss(recon_tensor, orig_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})

        # Validation
        pipeline.eval()
        pipeline.reset_statistics()
        val_pbar = tqdm(val_loader, desc="Validating images")

        with torch.no_grad():
            for data in val_pbar:
                image_paths = data["lr"]
                images = []
                for image_path in image_paths:
                    image = Image.open(image_path).convert("RGB")
                    image = image.resize(image_size, Image.BICUBIC)
                    images.append(TF.pil_to_tensor(image))

                orig_tensor = torch.stack(images).to(device) / 255.0
                recon_tensor, diagnostics = pipeline(orig_tensor)

                loss = torch.nn.functional.mse_loss(recon_tensor, orig_tensor)
                val_pbar.set_postfix({"val_loss": loss.item()})

            # Save images
            if epoch == 0:
                for i in range(min(2, orig_tensor.size(0))):
                    save_image(orig_tensor[i], f"output_images/orig_{i}.png")
                saved_original_images = True

            for i in range(min(2, recon_tensor.size(0))):
                save_image(
                    recon_tensor[i], f"output_images/recon_{i}_epoch_{epoch}.png"
                )

            print(f"Validation loss: {loss.item()}")
            pipeline.output_statistics()

        print(f"Epoch {epoch + 1} completed.")
