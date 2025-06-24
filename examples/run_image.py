import os
import sys

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from semcomf import (  # noqa: E402
    AWGNChannel,
    BasePipeline,
    IdealChannel,
    SimpleImageDecoder,
    SimpleImageEncoder,
)


def main():
    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_size = (128, 128)
    feature_dim = 32

    # --- Create a dummy image ---
    try:
        # Try to load a real image if you have one
        # img_path = "path/to/your/image.png"
        # original_pil_image = Image.open(img_path).convert("RGB").resize(image_size, Image.Resampling.LANCZOS)
        # For a quick test, create a synthetic image
        dummy_image_np = np.random.randint(0, 256, (*image_size, 3), dtype=np.uint8)
        original_pil_image = Image.fromarray(dummy_image_np, "RGB")
        print(f"Created a dummy PIL image of size: {original_pil_image.size}")
    except Exception as e:
        print(f"Error creating/loading image: {e}")
        print("Please ensure you have an image or Pillow is working correctly.")
        return

    # --- Instantiate Components ---
    # 1. Transmitter
    transmitter = SimpleImageEncoder(
        input_channels=3, feature_dim=feature_dim, image_size=image_size
    ).to(device)

    # 2. Channel (Choose one)
    # channel = IdealChannel().to(device)
    channel = AWGNChannel(snr_db=10.0).to(device)
    print(
        f"Using channel: {channel.__class__.__name__} with config: {channel.get_config()}"
    )

    # 3. Receiver
    receiver = SimpleImageDecoder(
        output_channels=3, feature_dim=feature_dim, output_size=image_size
    ).to(device)

    # 4. Pipeline
    pipeline = BasePipeline(transmitter, channel, receiver).to(device)
    pipeline.eval()

    # --- Process the image ---
    print("Starting semantic communication pipeline...")
    pipeline.reset_statistics()

    with torch.no_grad():
        reconstructed_tensor, diagnostics = pipeline(original_pil_image)

    # Save images
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)
    original_pil_image.save(os.path.join(output_dir, "original_simple.png"))
    reconstructed_pil_image = TF.to_pil_image(reconstructed_tensor.squeeze(0).cpu())
    reconstructed_pil_image.save(os.path.join(output_dir, "reconstructed_simple.png"))

    # Output collected statistics
    pipeline.output_statistics()


if __name__ == "__main__":
    main()
