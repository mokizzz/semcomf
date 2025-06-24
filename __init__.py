"""
Semantic Communication Framework (SemComF)
A flexible framework for building and evaluating semantic communication systems.
"""

__version__ = "0.1.1"  # Increment version

from .core.base_channel import BaseChannel
from .core.base_pipeline import BasePipeline
from .core.base_receiver import BaseReceiver
from .core.base_transmitter import BaseTransmitter
from .core.components import (
    AWGNChannel,
    IdealChannel,
    SimpleImageDecoder,
    SimpleImageEncoder,
    SimpleVideoDecoder,
    SimpleVideoEncoder,
)

# Export metrics
from .metrics import (
    FIDCalculator,
    calculate_all_metrics,
    calculate_lpips,
    calculate_ms_ssim,
    calculate_psnr,
    calculate_ssim,
)

# Export image utilities
from .utils.image_utils import convert_pil_to_tensor, convert_tensor_to_pil

# Export visualization utilities
from .utils.visualization import (
    arrange_images_grid,
    arrange_images_side_by_side,
    display_images,
)

print(f"SemComF version {__version__} loaded.")
