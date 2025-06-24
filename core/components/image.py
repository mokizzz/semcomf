import time
from typing import Any

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

from ..base_receiver import BaseReceiver
from ..base_transmitter import BaseTransmitter


class SimpleImageEncoder(BaseTransmitter):
    """Simple U-Net image encoder."""

    def __init__(
        self, input_channels=3, base_channels=32, image_size=(128, 128), config=None
    ):
        super().__init__(config)
        self.image_size = image_size

        # Encoder
        self.enc1 = self._conv_block(input_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck_conv = self._conv_block(base_channels * 4, base_channels * 8)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(True),
        )

    def _prepare_input(self, input_data: Image.Image | torch.Tensor) -> torch.Tensor:
        if isinstance(input_data, Image.Image):
            img_tensor = TF.to_tensor(
                input_data.resize(self.image_size, Image.Resampling.BILINEAR)
            )
            if img_tensor.shape[0] == 1:
                img_tensor = img_tensor.repeat(3, 1, 1)
            elif img_tensor.shape[0] == 4:
                img_tensor = img_tensor[:3, :, :]
            img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.float() / 255.0
        elif isinstance(input_data, torch.Tensor):
            if input_data.ndim == 3:
                img_tensor = input_data.unsqueeze(0)
            else:
                img_tensor = input_data
            if img_tensor.dtype in [
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ]:
                img_tensor = img_tensor.float() / 255.0
        else:
            raise TypeError(
                f"SimpleImageEncoder expects PIL Image or torch.Tensor, got {type(input_data)}"
            )

        return img_tensor

    def forward(
        self, input_data: Image.Image | torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        start_time = time.time()
        img_tensor = self._prepare_input(input_data)

        device = next(self.parameters()).device
        img_tensor = img_tensor.to(device)

        # Encoder
        enc1 = self.enc1(img_tensor)
        pool1 = self.pool1(enc1)
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)

        # Bottleneck
        bottleneck = self.bottleneck_conv(pool3)

        timings = {"encoding_time": time.time() - start_time}
        return {
            "features": bottleneck,
            "enc1": enc1,
            "enc2": enc2,
            "enc3": enc3,
        }, timings


class SimpleImageDecoder(BaseReceiver):
    """Simple image decoder."""

    def __init__(
        self,
        output_channels: int = 3,
        base_channels: int = 32,
        output_size: tuple[int, int] = (128, 128),
        config: dict | None = None,
    ):
        super().__init__(config)
        self.output_size = output_size

        # Decoder
        self.up3 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4, kernel_size=2, stride=2
        )
        self.dec3 = self._conv_block(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.dec2 = self._conv_block(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1 = self._conv_block(base_channels * 2, base_channels)

        self.final_conv = nn.Conv2d(base_channels, output_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(True),
        )

    def forward(
        self, received_data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        start_time = time.time()
        if not isinstance(received_data, dict) or "features" not in received_data:
            raise ValueError("SimpleImageDecoder expects a dict with a 'features' key.")

        dtype = next(self.parameters()).dtype
        bottleneck = received_data["features"].to(dtype=dtype)
        enc1 = received_data["enc1"].to(dtype=dtype)
        enc2 = received_data["enc2"].to(dtype=dtype)
        enc3 = received_data["enc3"].to(dtype=dtype)

        # Decoder
        up3 = self.up3(bottleneck)
        if up3.shape[-2:] != enc3.shape[-2:]:
            up3 = TF.resize(up3, size=enc3.shape[-2:])
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))

        up2 = self.up2(dec3)
        if up2.shape[-2:] != enc2.shape[-2:]:
            up2 = TF.resize(up2, size=enc2.shape[-2:])
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))

        up1 = self.up1(dec2)
        if up1.shape[-2:] != enc1.shape[-2:]:
            up1 = TF.resize(up1, size=enc1.shape[-2:])
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        reconstructed_image = self.sigmoid(self.final_conv(dec1))

        timings = {"decoding_time": time.time() - start_time}
        return reconstructed_image, timings
