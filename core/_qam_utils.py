import math
import random
from typing import List, Tuple

import bitstring
import numpy as np
import torch
from PIL import Image
from scipy.special import erfc


def float_to_bin(f: float, length: int = 64) -> str:
    try:
        return bitstring.BitArray(float=float(f), length=length).bin
    except Exception:
        return "".join(random.choice("01") for _ in range(length))


def bin_to_float(b: str) -> float:
    try:
        num = bitstring.BitArray(bin=b).float
        if (
            math.isnan(num)
            or math.isinf(num)
            or abs(num) > 1e12
            or (abs(num) < 1e-12 and num != 0)
        ):
            return (random.random() - 0.5) * 2
        return num
    except Exception:
        return (random.random() - 0.5) * 2


def char_to_bin(char_val: str, length: int = 8) -> str:
    return format(ord(char_val), f"0{length}b")[-length:]


def bin_to_char(bin_val: str) -> str:
    try:
        return chr(int(bin_val, 2))
    except ValueError:
        return chr(random.randint(32, 126))


def tensor_to_bit_array(tensor: torch.Tensor, bits_per_float: int = 32) -> np.ndarray:
    if not tensor.is_floating_point():
        tensor = tensor.float()

    tensor_flat = tensor.cpu().detach().reshape(-1).numpy()

    if bits_per_float == 32:
        dtype = np.float32
    else:
        dtype = np.float64

    tensor_flat = tensor_flat.astype(dtype)
    bit_view = tensor_flat.view(np.uint32 if bits_per_float == 32 else np.uint64)

    bit_array = np.unpackbits(bit_view.view(np.uint8), bitorder="big").reshape(
        -1, bits_per_float
    )

    return bit_array


def bit_array_to_tensor(
    bit_array: np.ndarray,
    original_shape: Tuple,
    dtype: torch.dtype = torch.float32,
    bits_per_float: int = 32,
) -> torch.Tensor:
    packed_bytes = np.packbits(bit_array.flatten(), bitorder="big")

    if bits_per_float == 32:
        float_view = packed_bytes.view(np.float32)
    else:
        float_view = packed_bytes.view(np.float64)

    expected_len = int(np.prod(original_shape))
    if len(float_view) != expected_len:
        if len(float_view) < expected_len:
            padding = np.random.uniform(-1, 1, expected_len - len(float_view))
            float_view = np.concatenate([float_view, padding])
        else:
            float_view = float_view[:expected_len]

    return torch.tensor(float_view, dtype=dtype).reshape(original_shape)


def tensor_to_bit_list(tensor: torch.Tensor, bits_per_float: int = 32) -> List[str]:
    """Converts a flat tensor to a list of binary strings."""
    if not tensor.is_floating_point():
        tensor = tensor.float()

    tensor_flat = tensor.cpu().detach().reshape(-1).numpy()
    return [float_to_bin(f, length=bits_per_float) for f in tensor_flat]


def bit_list_to_tensor(
    bit_list: List[str], original_shape: Tuple, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Converts a list of binary strings back to a tensor of original_shape."""
    floats = [bin_to_float(b) for b in bit_list]
    if len(floats) != np.prod(original_shape):
        expected_len = int(np.prod(original_shape))
        if len(floats) < expected_len:
            floats.extend([(random.random() - 0.5) * 2] * (expected_len - len(floats)))
        else:
            floats = floats[:expected_len]

    return torch.tensor(floats, dtype=dtype).reshape(original_shape)


def string_to_bit_list(s: str, bits_per_char: int = 8) -> List[str]:
    return [char_to_bin(char, length=bits_per_char) for char in s]


def bit_list_to_string(bit_list: List[str]) -> str:
    return "".join([bin_to_char(b) for b in bit_list])


def pil_image_to_bit_array(
    image: Image.Image, bits_per_channel: int = 8
) -> Tuple[np.ndarray, Tuple, str]:
    """将PIL图像转换为二进制数组"""
    original_mode = image.mode
    img_array = np.array(image, dtype=np.uint8)
    original_shape = img_array.shape

    bit_array = np.unpackbits(img_array.flatten(), bitorder="big")
    bit_array = bit_array.reshape(-1, bits_per_channel)

    return bit_array, original_shape, original_mode


def bit_array_to_pil_image(
    bit_array: np.ndarray,
    original_shape: Tuple,
    original_mode: str,
    bits_per_channel: int = 8,
) -> Image.Image:
    if bit_array.ndim == 2 and bit_array.shape[1] != bits_per_channel:
        raise ValueError(
            f"bit_array second dimension ({bit_array.shape[1]}) must match bits_per_channel ({bits_per_channel})"
        )

    if bit_array.ndim == 1:
        if len(bit_array) % bits_per_channel != 0:
            pad_length = bits_per_channel - (len(bit_array) % bits_per_channel)
            bit_array = np.pad(
                bit_array, (0, pad_length), mode="constant", constant_values=0
            )
        bit_array = bit_array.reshape(-1, bits_per_channel)

    pixel_values = []
    max_val = (1 << bits_per_channel) - 1

    for bit_row in bit_array:
        try:
            bit_string = "".join(bit_row.astype(str))
            pixel_val = int(bit_string, 2)
            pixel_values.append(min(pixel_val, max_val))
        except (ValueError, OverflowError):
            pixel_values.append(np.random.randint(0, max_val + 1))

    expected_len = int(np.prod(original_shape))
    if len(pixel_values) != expected_len:
        if len(pixel_values) < expected_len:
            padding = np.random.randint(
                0, max_val + 1, expected_len - len(pixel_values)
            )
            pixel_values.extend(padding.tolist())
        else:
            pixel_values = pixel_values[:expected_len]

    reconstructed_array = np.array(pixel_values, dtype=np.uint8).reshape(original_shape)
    return Image.fromarray(reconstructed_array, mode=original_mode)


# def pil_image_to_bit_list(
#     image: Image.Image, bits_per_channel_value: int = 8
# ) -> Tuple[List[str], Tuple, str]:
#     """Converts PIL Image to list of binary strings for pixel values.
#     Returns bit_list, original_shape, original_mode."""
#     original_mode = image.mode
#     img_array = np.array(image)
#     original_shape = img_array.shape

#     bit_list = []
#     for val in img_array.flat:
#         bin_str = format(int(val), f"0{bits_per_channel_value}b")[
#             -bits_per_channel_value:
#         ]
#         bit_list.append(bin_str)
#     return bit_list, original_shape, original_mode


# def bit_list_to_pil_image(
#     bit_list: List[str],
#     original_shape: Tuple,
#     original_mode: str,
#     bits_per_channel_value: int = 8,
# ) -> Image.Image:
#     """Converts list of binary strings back to PIL Image."""
#     pixel_values = []
#     max_val = (1 << bits_per_channel_value) - 1
#     for bin_str in bit_list:
#         try:
#             pixel_values.append(int(bin_str, 2))
#         except ValueError:
#             pixel_values.append(random.randint(0, max_val))

#     if len(pixel_values) != np.prod(original_shape):
#         expected_len = int(np.prod(original_shape))
#         if len(pixel_values) < expected_len:
#             pixel_values.extend(
#                 [random.randint(0, max_val)] * (expected_len - len(pixel_values))
#             )
#         else:
#             pixel_values = pixel_values[:expected_len]

#     reconstructed_array = np.array(pixel_values, dtype=np.uint8).reshape(original_shape)
#     return Image.fromarray(reconstructed_array, mode=original_mode)


def introduce_qam_noise(
    bit_strings: List[str],
    snr_db: float,
    qam_order: int = 16,
    fading_model: str = None,
    rayleigh_avg_power_db: float = 0,
) -> List[str]:
    """
    Introduces noise to a list of bit strings based on QAM modulation and SNR.
    Each element in bit_strings is a string of '0's and '1's representing one symbol or part of it.
    This version applies BER directly to each bit using vectorized operations.
    """
    if not np.sqrt(qam_order).is_integer():
        raise ValueError("QAM order must be a perfect square (e.g., 4, 16, 64).")

    bits_per_symbol = int(np.log2(qam_order))

    snr_linear_per_symbol = 10 ** (snr_db / 10)

    all_bits = "".join(bit_strings)
    total_bits = len(all_bits)

    if total_bits == 0:
        return []

    current_snr_linear = snr_linear_per_symbol
    if fading_model == "rayleigh":
        avg_fade_power_linear = 10 ** (rayleigh_avg_power_db / 10)
        sigma_fade = np.sqrt(avg_fade_power_linear / 2.0)
        num_blocks = len(bit_strings)
        h_real = np.random.normal(0, sigma_fade, num_blocks)
        h_imag = np.random.normal(0, sigma_fade, num_blocks)
        channel_gain_power = h_real**2 + h_imag**2
        channel_gain_power_expanded = np.repeat(
            channel_gain_power, [len(b) for b in bit_strings]
        )
        current_snr_linear = channel_gain_power_expanded * snr_linear_per_symbol

    if qam_order == 4:
        snr_per_bit_eff = current_snr_linear / 2
        ber = 0.5 * erfc(np.sqrt(snr_per_bit_eff))
    elif qam_order >= 16:
        snr_per_bit_eff = current_snr_linear / bits_per_symbol
        ser_approx = (
            2
            * (1 - 1 / np.sqrt(qam_order))
            * erfc(np.sqrt(1.5 * current_snr_linear / (qam_order - 1)))
        )
        ber = ser_approx / bits_per_symbol

    ber = np.nan_to_num(ber, nan=0.5, posinf=0.5, neginf=0.5)
    ber = np.clip(ber, 0, 1)

    random_numbers = np.random.random(total_bits)
    flip_mask = random_numbers < ber

    noisy_bits_list = list(all_bits)
    for i in range(total_bits):
        if flip_mask[i]:
            noisy_bits_list[i] = "1" if noisy_bits_list[i] == "0" else "0"

    noisy_all_bits = "".join(noisy_bits_list)

    noisy_bit_strings = []
    current_index = 0
    for bit_block_str in bit_strings:
        block_len = len(bit_block_str)
        noisy_bit_strings.append(
            noisy_all_bits[current_index : current_index + block_len]
        )
        current_index += block_len

    return noisy_bit_strings


def introduce_qam_noise_optimized(
    bit_strings: np.ndarray,
    snr_db: float,
    qam_order: int = 16,
    fading_model: str = None,
    rayleigh_avg_power_db: float = 0,
) -> np.ndarray:
    """
    优化版本的QAM噪声引入函数。
    此版本针对 np.ndarray 类型的输入进行了优化，并返回一个 np.ndarray。

    主要优化：
    1. 减少不必要的数组转换。
    2. 使用更高效的向量化操作。
    3. 避免逐个处理比特。
    """
    if not np.sqrt(qam_order).is_integer():
        raise ValueError("QAM order must be a perfect square (e.g., 4, 16, 64).")

    if bit_strings is None or len(bit_strings) == 0:
        return np.array([], dtype=np.uint8).reshape(
            0, bit_strings.shape[1] if bit_strings.ndim > 1 else 0
        )

    # 从输入的 NumPy 数组获取长度信息
    num_blocks, bits_per_block = bit_strings.shape
    total_bits = num_blocks * bits_per_block
    if total_bits == 0:
        return bit_strings.copy()

    # 预计算参数
    bits_per_symbol = int(np.log2(qam_order))
    snr_linear_per_symbol = 10 ** (snr_db / 10)

    # 处理衰落信道
    if fading_model == "rayleigh":
        avg_fade_power_linear = 10 ** (rayleigh_avg_power_db / 10)
        sigma_fade = np.sqrt(avg_fade_power_linear / 2.0)

        h_real = np.random.normal(0, sigma_fade, num_blocks)
        h_imag = np.random.normal(0, sigma_fade, num_blocks)
        channel_gain_power = h_real**2 + h_imag**2
        effective_snr_per_block = channel_gain_power * snr_linear_per_symbol
    else:
        effective_snr_per_block = np.full(num_blocks, snr_linear_per_symbol)

    # 向量化BER计算 - 为每个块预计算BER
    if qam_order == 4:
        snr_per_bit_eff = effective_snr_per_block / 2
        ber_per_block = 0.5 * erfc(np.sqrt(snr_per_bit_eff))
    elif qam_order >= 16:
        ser_denominator = qam_order - 1
        ser_denominator = np.where(ser_denominator == 0, 1, ser_denominator)
        ser_approx = (
            2
            * (1 - 1 / np.sqrt(qam_order))
            * erfc(np.sqrt(1.5 * effective_snr_per_block / ser_denominator))
        )
        ber_per_block = ser_approx / bits_per_symbol

    ber_per_block = np.clip(np.nan_to_num(ber_per_block, nan=0.5), 0, 1)

    noisy_bit_array_full = bit_strings.copy()
    random_vals = np.random.random(bit_strings.shape)
    ber_expanded = ber_per_block.reshape(-1, 1)
    flip_mask = random_vals < ber_expanded
    noisy_bit_array_full[flip_mask] = 1 - noisy_bit_array_full[flip_mask]

    return noisy_bit_array_full
