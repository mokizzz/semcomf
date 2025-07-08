import math

import numpy as np
import piqa
import torch
from PIL import Image

from semcomf.utils.image_utils import (
    convert_pil_to_tensor,
    convert_tensor_to_pil,
    resize_tensor_image,
)

_metric_models = {}


def _get_metric_model(metric_name: str, device: torch.device, **kwargs):
    """Initialize and cache metric models from piqa."""

    model_key = f"{metric_name}_{str(device)}"
    if model_key not in _metric_models:
        if metric_name == "psnr":
            _metric_models[model_key] = piqa.PSNR(**kwargs).to(device).eval()
        elif metric_name == "ssim":
            _metric_models[model_key] = piqa.SSIM(**kwargs).to(device).eval()
        elif metric_name == "ms_ssim":
            _metric_models[model_key] = piqa.MS_SSIM(**kwargs).to(device).eval()
        elif metric_name == "lpips":
            _metric_models[model_key] = (
                piqa.LPIPS(network=kwargs.get("network", "alex")).to(device).eval()
            )
        elif metric_name == "fid":
            _metric_models[model_key] = piqa.FID().to(device).eval()
        else:
            raise ValueError(f"Unknown metric model: {metric_name}")
    return _metric_models[model_key]


def _prepare_tensors_for_metric(
    img1: torch.Tensor | Image.Image,
    img2: torch.Tensor | Image.Image,
    device: torch.device | None = None,
    target_size: tuple[int, int] | None = None,
    data_range: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert inputs to tensors, move to device, resize, and ensure batch dim."""
    if isinstance(img1, Image.Image):
        img1 = convert_pil_to_tensor(img1)
    if isinstance(img2, Image.Image):
        img2 = convert_pil_to_tensor(img2)

    if not isinstance(img1, torch.Tensor) or not isinstance(img2, torch.Tensor):
        raise TypeError("Inputs must be PIL Images or PyTorch Tensors.")

    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
    if img2.ndim == 3:
        img2 = img2.unsqueeze(0)

    if device is None:
        device = img1.device

    img1 = img1.to(device)
    img2 = img2.to(device)

    if target_size:
        if img1.shape[-2:] != target_size:
            img1 = resize_tensor_image(img1, target_size)
        if img2.shape[-2:] != target_size:
            img2 = resize_tensor_image(img2, target_size)

    img1 = img1.clamp(0, data_range)
    img2 = img2.clamp(0, data_range)

    return img1, img2


def calculate_psnr(
    img1: torch.Tensor | Image.Image,
    img2: torch.Tensor | Image.Image,
    data_range: float = 1.0,
    device: torch.device | None = None,
) -> float:
    img1, img2 = _prepare_tensors_for_metric(img1, img2, device, data_range=data_range)
    model = _get_metric_model("psnr", img1.device)
    img1 = img1.to(torch.float32)
    img2 = img2.to(torch.float32)
    return model(img1, img2).item()


def calculate_ssim(
    img1: torch.Tensor | Image.Image,
    img2: torch.Tensor | Image.Image,
    data_range: float = 1.0,
    device: torch.device | None = None,
) -> float:
    img1, img2 = _prepare_tensors_for_metric(img1, img2, device, data_range=data_range)
    model = _get_metric_model("ssim", img1.device)
    img1 = img1.to(torch.float32)
    img2 = img2.to(torch.float32)
    return model(img1, img2).item()


def calculate_ms_ssim(
    img1: torch.Tensor | Image.Image,
    img2: torch.Tensor | Image.Image,
    data_range: float = 1.0,
    min_size_for_ms_ssim: int = 256,
    device: torch.device | None = None,
) -> float:
    if isinstance(img1, Image.Image) and isinstance(img2, Image.Image):
        temp_img1_pil = (
            img1 if isinstance(img1, Image.Image) else convert_tensor_to_pil(img1)
        )
        h, w = temp_img1_pil.height, temp_img1_pil.width
        target_h, target_w = max(h, min_size_for_ms_ssim), max(w, min_size_for_ms_ssim)

        img1, img2 = _prepare_tensors_for_metric(
            img1,
            img2,
            device,
            target_size=(target_h, target_w),
            data_range=data_range,
        )
    else:
        img1, img2 = _prepare_tensors_for_metric(
            img1, img2, device, data_range=data_range
        )

    model = _get_metric_model("ms_ssim", img1.device)
    img1 = img1.to(torch.float32)
    img2 = img2.to(torch.float32)
    try:
        return model(img1, img2).item()
    except RuntimeError as e:
        print(f"Warning: MS-SSIM calculation failed due to input size. Returning NaN. Error: {e}")
        return float('nan')


def calculate_lpips(
    img1: torch.Tensor | Image.Image,
    img2: torch.Tensor | Image.Image,
    device: torch.device | None = None,
    net: str = "alex",
) -> float:
    img1, img2 = _prepare_tensors_for_metric(img1, img2, device, data_range=1.0)
    model = _get_metric_model("lpips", img1.device, network=net)
    img1 = img1.to(torch.float32)
    img2 = img2.to(torch.float32)
    return model(img1, img2).item()


class FIDCalculator:
    """
    A stateful calculator for Frechet Inception Distance (FID).
    Feed features of real and generated images incrementally.
    """

    def __init__(self):
        self.device = torch.device("cpu")
        self.fid_model = _get_metric_model("fid", self.device)
        self.reset()

    def reset(self):
        """Resets the internal state of the FID model."""
        self.real_features_list = []
        self.fake_features_list = []
        if hasattr(self.fid_model, "reset"):
            self.fid_model.reset()

    def update_features(
        self,
        images: torch.Tensor | list[Image.Image] | list[torch.Tensor],
        real: bool,
    ):
        """Extracts and stores features for a batch of images."""
        if images is None or (
            isinstance(images, (list, torch.Tensor)) and len(images) == 0
        ):
            return

        processed_images = []
        if isinstance(images, torch.Tensor):
            if images.ndim == 3:
                images = images.unsqueeze(0)
            images, _ = _prepare_tensors_for_metric(
                images, images.clone(), self.device, data_range=1.0
            )
            processed_images = [images]
        elif isinstance(images, list):
            for img_item in images:
                img_t, _ = _prepare_tensors_for_metric(
                    img_item, img_item, self.device, data_range=1.0
                )
                processed_images.append(img_t)

        if not processed_images:
            return

        for img_batch_t in processed_images:
            features = self.fid_model.features(img_batch_t)
            if real:
                self.real_features_list.append(features)
            else:
                self.fake_features_list.append(features)

    def compute_fid(self) -> float:
        """Computes FID score based on all accumulated features."""
        if not self.real_features_list or not self.fake_features_list:
            return float("nan")

        all_real_features = torch.cat(self.real_features_list, dim=0)
        all_fake_features = torch.cat(self.fake_features_list, dim=0)

        if all_real_features.shape[0] == 0 or all_fake_features.shape[0] == 0:
            return float("nan")

        return self.fid_model(all_real_features, all_fake_features).item()


def calculate_all_metrics(
    img1_orig: torch.Tensor | Image.Image,
    img2_recon: torch.Tensor | Image.Image,
    device: torch.device | None = None,
    data_range: float = 1.0,
) -> dict[str, float]:
    """Calculates a standard set of image quality metrics."""
    metrics = {}
    try:
        metrics["psnr"] = calculate_psnr(
            img1_orig, img2_recon, data_range=data_range, device=device
        )
    except Exception as e:
        print(f"Could not calculate PSNR: {e}")
        metrics["psnr"] = float("nan")

    try:
        metrics["ssim"] = calculate_ssim(
            img1_orig, img2_recon, data_range=data_range, device=device
        )
    except Exception as e:
        print(f"Could not calculate SSIM: {e}")
        metrics["ssim"] = float("nan")
    try:
        metrics["ms_ssim"] = calculate_ms_ssim(
            img1_orig, img2_recon, data_range=data_range, device=device
        )
    except Exception as e:
        print(f"Could not calculate MS-SSIM: {e}")
        metrics["ms_ssim"] = float("nan")
    try:
        metrics["lpips"] = calculate_lpips(img1_orig, img2_recon, device=device)
    except Exception as e:
        print(f"Could not calculate LPIPS: {e}")
        metrics["lpips"] = float("nan")

    return metrics


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        pil_img1 = Image.new("RGB", (256, 256), "red")
        pil_img2 = Image.new("RGB", (256, 256), "pink")

        array2_perturbed = np.array(pil_img2).astype(np.float32)
        array2_perturbed[10:20, 10:20, 0] = 0
        pil_img2_perturbed = Image.fromarray(array2_perturbed.astype(np.uint8))

        print("--- Individual Metric Tests ---")
        psnr_val = calculate_psnr(pil_img1, pil_img2_perturbed, device=device)
        print(f"PSNR: {psnr_val:.2f} dB")

        ssim_val = calculate_ssim(pil_img1, pil_img2_perturbed, device=device)
        print(f"SSIM: {ssim_val:.4f}")
        ms_ssim_val = calculate_ms_ssim(pil_img1, pil_img2_perturbed, device=device)
        print(f"MS-SSIM: {ms_ssim_val:.4f}")
        lpips_val = calculate_lpips(pil_img1, pil_img2_perturbed, device=device)
        print(f"LPIPS: {lpips_val:.4f}")

        print("\n--- Calculate All Metrics ---")
        all_m = calculate_all_metrics(pil_img1, pil_img2_perturbed, device=device)
        for k, v in all_m.items():
            print(f"  {k}: {v:.4f}")

        print("\n--- FID Calculator Test ---")
        fid_calc = FIDCalculator()

        real_images_pil = [
            Image.new("RGB", (64, 64), color=(i * 10, i * 5, i * 2)) for i in range(5)
        ]
        fake_images_pil = [
            Image.new("RGB", (64, 64), color=(i * 10 + 5, i * 5 + 5, i * 2 + 5))
            for i in range(5)
        ]

        fid_calc.reset()
        fid_calc.update_features(real_images_pil, real=True)
        fid_calc.update_features(fake_images_pil, real=False)
        fid_score_pil = fid_calc.compute_fid()
        print(f"FID Score (from PIL): {fid_score_pil:.4f}")

        real_images_tensor = torch.stack(
            [convert_pil_to_tensor(img) for img in real_images_pil]
        ).squeeze(1)
        fake_images_tensor = torch.stack(
            [convert_pil_to_tensor(img) for img in fake_images_pil]
        ).squeeze(1)

        fid_calc.reset()
        fid_calc.update_features(real_images_tensor, real=True)
        fid_calc.update_features(fake_images_tensor, real=False)
        fid_score_tensor = fid_calc.compute_fid()
        print(f"FID Score (from Tensors): {fid_score_tensor:.4f}")

    except ImportError as e:
        print(f"Import error during test: {e}. Some metrics might not be available.")
    except Exception as e:
        print(f"An error occurred during testing: {e}")
        import traceback

        traceback.print_exc()
