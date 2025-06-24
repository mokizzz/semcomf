from typing import Any

import torch
from PIL import Image

from .._qam_utils import (
    bit_array_to_pil_image,
    bit_array_to_tensor,
    bit_list_to_string,
    introduce_qam_noise_optimized,
    pil_image_to_bit_array,
    string_to_bit_list,
    tensor_to_bit_array,
)
from ..base_channel import BaseChannel


class IdealChannel(BaseChannel):
    """An ideal channel that passes data through without modification or noise."""

    def forward(self, transmitted_data: Any) -> tuple[Any, dict[str, float]]:
        size_info = self.calculate_data_size_kb(transmitted_data)
        return transmitted_data, size_info


class QAMNoiseChannel(BaseChannel):
    """Simulates a channel with QAM and AWGN or Rayleigh fading."""

    def __init__(
        self,
        snr_db: float = 20.0,
        qam_order: int = 16,
        bits_per_float_param: int = 32,
        bits_per_char_param: int = 8,
        bits_per_channel_value_param: int = 8,
        precision: torch.dtype = torch.float32,
        fading_model: str | None = None,
        rayleigh_avg_power_db: float = 0.0,
        noise_exempt_keys: list[str] | None = None,
        config: dict | None = None,
    ):
        super().__init__(config)
        self.snr_db = snr_db
        self.qam_order = qam_order
        self.bits_per_float = bits_per_float_param
        self.bits_per_char = bits_per_char_param
        self.bits_per_channel_value = bits_per_channel_value_param
        self.precision = precision
        self.fading_model = fading_model
        self.rayleigh_avg_power_db = rayleigh_avg_power_db

        self.noise_exempt_keys = (
            noise_exempt_keys
            if noise_exempt_keys is not None
            else ["trans_method", "original_size_info", "compression_ratio"]
        )
        if self.config:
            self.snr_db = self.config.get("snr_db", self.snr_db)
            self.qam_order = self.config.get("qam_order", self.qam_order)
            self.bits_per_float = self.config.get(
                "bits_per_float_param", self.bits_per_float
            )
            self.bits_per_char = self.config.get(
                "bits_per_char_param", self.bits_per_char
            )
            self.bits_per_channel_value = self.config.get(
                "bits_per_channel_value_param", self.bits_per_channel_value
            )
            self.precision = self.config.get("precision", self.precision)
            self.fading_model = self.config.get("fading_model", self.fading_model)
            self.rayleigh_avg_power_db = self.config.get(
                "rayleigh_avg_power_db", self.rayleigh_avg_power_db
            )
            self.noise_exempt_keys = self.config.get(
                "noise_exempt_keys", self.noise_exempt_keys
            )

    def _apply_qam_noise_recursive(self, data_item: Any) -> Any:
        if data_item is None:
            return None

        original_type = type(data_item)
        noisy_item = None

        try:
            if isinstance(data_item, torch.Tensor):
                original_shape = data_item.shape
                original_dtype = data_item.dtype
                device = data_item.device
                data_item = data_item.to(self.precision)
                bit_array = tensor_to_bit_array(
                    data_item, bits_per_float=self.bits_per_float
                )
                noisy_bit_array = introduce_qam_noise_optimized(
                    bit_array,
                    self.snr_db,
                    self.qam_order,
                    self.fading_model,
                    self.rayleigh_avg_power_db,
                )
                noisy_item = bit_array_to_tensor(
                    noisy_bit_array, original_shape, dtype=original_dtype
                ).to(device)

            elif isinstance(data_item, str):
                bit_array = string_to_bit_list(
                    data_item, bits_per_char=self.bits_per_char
                )
                noisy_bit_array = introduce_qam_noise_optimized(
                    bit_array,
                    self.snr_db,
                    self.qam_order,
                    self.fading_model,
                    self.rayleigh_avg_power_db,
                )
                noisy_item = bit_list_to_string(noisy_bit_array)

            elif isinstance(data_item, Image.Image):
                bit_array, shape, mode = pil_image_to_bit_array(
                    data_item, bits_per_channel_value=self.bits_per_channel_value
                )
                noisy_bit_array = introduce_qam_noise_optimized(
                    bit_array,
                    self.snr_db,
                    self.qam_order,
                    self.fading_model,
                    self.rayleigh_avg_power_db,
                )
                noisy_item = bit_array_to_pil_image(
                    noisy_bit_array,
                    shape,
                    mode,
                    bits_per_channel_value=self.bits_per_channel_value,
                )

            elif isinstance(data_item, dict):
                noisy_item = {
                    k: (
                        self._apply_qam_noise_recursive(v)
                        if k not in self.noise_exempt_keys
                        else v
                    )
                    for k, v in data_item.items()
                }
            elif isinstance(data_item, list):
                noisy_item = [
                    self._apply_qam_noise_recursive(item) for item in data_item
                ]
            elif isinstance(data_item, tuple):
                noisy_item = tuple(
                    self._apply_qam_noise_recursive(item) for item in data_item
                )
            else:
                noisy_item = data_item
        except Exception as e:
            print(
                f"Error applying QAM noise to {original_type}: {e}. Returning original item or None."
            )
            raise e
            return data_item

        return noisy_item

    def forward(self, transmitted_data: Any) -> tuple[Any, dict[str, float]]:
        size_info = self.calculate_data_size_kb(transmitted_data)

        if self.snr_db is None or self.snr_db >= 100 and self.fading_model is None:
            return transmitted_data, size_info

        data_after_channel = self._apply_qam_noise_recursive(transmitted_data)
        return data_after_channel, size_info


class AWGNChannel(QAMNoiseChannel):
    """Simulates an AWGN channel using QAM."""

    def __init__(
        self,
        snr_db: float = 20.0,
        qam_order: int = 16,
        bits_per_float_param: int = 32,
        bits_per_char_param: int = 8,
        bits_per_channel_value_param: int = 8,
        precision: torch.dtype = torch.float32,
        noise_exempt_keys: list[str] | None = None,
        config: dict | None = None,
    ):
        super().__init__(
            snr_db=snr_db,
            qam_order=qam_order,
            bits_per_float_param=bits_per_float_param,
            bits_per_char_param=bits_per_char_param,
            bits_per_channel_value_param=bits_per_channel_value_param,
            precision=precision,
            fading_model=None,
            rayleigh_avg_power_db=0.0,
            noise_exempt_keys=noise_exempt_keys,
            config=config,
        )
        if self.config:
            self.fading_model = self.config.get("fading_model", None)
            self.rayleigh_avg_power_db = self.config.get("rayleigh_avg_power_db", 0.0)


class RayleighChannel(QAMNoiseChannel):
    """Simulates a Rayleigh fading channel using QAM."""

    def __init__(
        self,
        snr_db: float = 20.0,
        qam_order: int = 16,
        bits_per_float_param: int = 32,
        bits_per_char_param: int = 8,
        bits_per_channel_value_param: int = 8,
        precision: torch.dtype = torch.float32,
        rayleigh_avg_power_db: float = 0.0,
        noise_exempt_keys: list[str] | None = None,
        config: dict | None = None,
    ):
        super().__init__(
            snr_db=snr_db,
            qam_order=qam_order,
            bits_per_float_param=bits_per_float_param,
            bits_per_char_param=bits_per_char_param,
            bits_per_channel_value_param=bits_per_channel_value_param,
            precision=precision,
            fading_model="rayleigh",
            rayleigh_avg_power_db=rayleigh_avg_power_db,
            noise_exempt_keys=noise_exempt_keys,
            config=config,
        )
        if self.config:
            self.fading_model = self.config.get("fading_model", "rayleigh")
            self.rayleigh_avg_power_db = self.config.get(
                "rayleigh_avg_power_db", self.rayleigh_avg_power_db
            )
