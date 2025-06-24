import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


def convert_pil_to_tensor(
    pil_image: Image.Image, device: str | torch.device = None
) -> torch.Tensor:
    """
    Convert a PIL Image to a PyTorch tensor (C, H, W) in range [0, 1].
    Adds a batch dimension (1, C, H, W).
    """
    tensor = TF.to_tensor(pil_image)  # Converts to CxHxW, scales to [0,1]
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)  # Add batch dimension -> BxCxHxW
    if device:
        tensor = tensor.to(device)
    return tensor


def convert_tensor_to_pil(tensor: torch.Tensor, mode: str = None) -> Image.Image:
    """
    Convert a PyTorch tensor (typically BxCxHxW or CxHxW, range [0,1]) to a PIL Image.
    If batch dimension exists, it takes the first image from the batch.
    """
    if tensor.ndim == 4 and tensor.shape[0] == 1:  # BxCxHxW
        tensor = tensor.squeeze(0)  # CxHxW
    elif tensor.ndim != 3:
        raise ValueError(
            f"Input tensor must be 3D (CxHxW) or 4D (1xCxHxW), got {tensor.shape}"
        )

    tensor = tensor.clamp(0, 1).cpu()  # Ensure on CPU and values in [0,1]
    return TF.to_pil_image(tensor, mode=mode)


def resize_pil_image(
    image: Image.Image,
    size: int | tuple[int, int],
    interpolation=Image.Resampling.LANCZOS,
) -> Image.Image:
    """Resizes a PIL image."""
    if isinstance(size, int):
        size = (size, size)
    return image.resize(size, resample=interpolation)


def resize_tensor_image(
    tensor: torch.Tensor,
    size: int | tuple[int, int],
    interpolation: TF.InterpolationMode = TF.InterpolationMode.BILINEAR,
    antialias: bool = True,
) -> torch.Tensor:
    """
    Resizes a batch of tensor images (BxCxHxW).
    Size can be (h, w) or a single int for square.
    """
    if isinstance(size, int):
        new_h, new_w = size, size
    else:
        new_h, new_w = size

    # TF.resize expects size as [h, w]
    return TF.resize(
        tensor, [new_h, new_w], interpolation=interpolation, antialias=antialias
    )


def pil_to_numpy(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to NumPy array."""
    return np.array(pil_image)


def numpy_to_pil(np_array: np.ndarray, mode: str = None) -> Image.Image:
    """Convert NumPy array to PIL Image."""
    return Image.fromarray(np_array, mode=mode)
