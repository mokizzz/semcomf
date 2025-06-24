# semcomf/semcomf/core/base_channel.py
import io
import sys
import zlib
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class BaseChannel(nn.Module, ABC):
    """Abstract base class for the communication channel."""

    def __init__(self, config: dict | None = None):
        super().__init__()
        self.config = config if config is not None else {}
        self.use_zlib_for_tensors = self.config.get("use_zlib_for_tensors", False)

    @abstractmethod
    def forward(self, transmitted_data: Any) -> tuple[Any, dict[str, float]]:
        """Processes data from the transmitter and returns received data and size info."""
        pass

    def get_config(self) -> dict:
        return self.config

    def _get_byte_size(self, obj: Any) -> int:
        """Calculate byte size of various objects."""
        if obj is None:
            return 0
        if isinstance(obj, (int, float)):
            return sys.getsizeof(obj)
        if isinstance(obj, str):
            return len(obj.encode("utf-8"))
        if isinstance(obj, bytes):
            return len(obj)
        if isinstance(obj, tuple):
            return sum(self._get_byte_size(v) for v in obj)
        if isinstance(obj, list):
            return sum(self._get_byte_size(v) for v in obj)
        if isinstance(obj, np.ndarray):
            if self.use_zlib_for_tensors:
                raw_bytes = obj.tobytes()
                return len(zlib.compress(raw_bytes))
            else:
                return obj.size * obj.itemsize
        if isinstance(obj, torch.Tensor):
            if self.use_zlib_for_tensors:
                try:
                    raw_bytes = obj.cpu().detach().contiguous().numpy().tobytes()
                    return len(zlib.compress(raw_bytes))
                except RuntimeError as e:
                    print(
                        f"Warning: Could not get raw bytes for tensor {type(obj)}, using element_size * nelement. Error: {e}"
                    )
                    return obj.element_size() * obj.nelement()
            else:
                return obj.element_size() * obj.nelement()
        if isinstance(obj, Image.Image):
            buffer = io.BytesIO()
            img_format = (
                obj.format
                if obj.format and obj.format in ["PNG", "JPEG", "GIF", "WEBP"]
                else "PNG"
            )
            try:
                if obj.mode == "1" and img_format != "PNG":
                    obj.save(buffer, format="PNG", optimize=True)
                else:
                    obj.save(buffer, format=img_format, optimize=True)
            except Exception:
                obj.save(buffer, format="PNG", optimize=True)
            return len(buffer.getvalue())
        if isinstance(obj, dict):
            return sum(self._get_byte_size(v) for _, v in obj.items())

        print(
            f"Warning: Unsupported type for size calculation: {type(obj)}. Returning sys.getsizeof or 0."
        )
        try:
            return sys.getsizeof(obj)
        except TypeError:
            return 0

    def calculate_data_size_kb(self, data: Any) -> dict[str, float]:
        """Calculates the size of the input data and its components in KB and total bits."""
        sizes_kb = {}
        if isinstance(data, dict):
            total_size_bytes = 0
            for key, value in data.items():
                component_size_bytes = self._get_byte_size(value)
                sizes_kb[key] = component_size_bytes / 1024.0
                total_size_bytes += component_size_bytes
            sizes_kb["total_kb"] = total_size_bytes / 1024.0
            sizes_kb["total_bits"] = total_size_bytes * 8
        else:
            total_size_bytes = self._get_byte_size(data)
            sizes_kb["total_kb"] = total_size_bytes / 1024.0
            sizes_kb["total_bits"] = total_size_bytes * 8
        return sizes_kb
