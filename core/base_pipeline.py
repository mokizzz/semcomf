import time
from typing import Any, Dict, List

import torch
import torch.nn as nn
from loguru import logger
from PIL import Image
from torchvision import transforms

from ..metrics.image_metrics import (
    FIDCalculator,
    calculate_lpips,
    calculate_ms_ssim,
    calculate_psnr,
    calculate_ssim,
)
from .base_channel import BaseChannel


class BasePipeline(nn.Module):
    """
    A base class to orchestrate the semantic communication pipeline:
    Transmitter -> Channel -> Receiver.
    """

    def __init__(
        self,
        transmitter: nn.Module,
        channel: BaseChannel,
        receiver: nn.Module,
        enable_fid: bool = False,
    ):
        super().__init__()
        self.transmitter = transmitter
        self.channel = channel
        self.receiver = receiver
        self.enable_fid = enable_fid
        self._statistics: List[Dict[str, Any]] = []
        self._fid_calculator = FIDCalculator() if enable_fid else None
        self._fid_calculated = False
        self._min_fid_images = 2

    def reset_statistics(self):
        """Clears all collected statistics and resets the FID calculator."""
        self._statistics = []
        if self.enable_fid and self._fid_calculator is not None:
            self._fid_calculator.reset()
        self._fid_calculated = False

    def output_statistics(self) -> Dict[str, float]:
        """Outputs the collected statistics."""
        if not self._statistics:
            logger.warning("No statistics collected yet.")
            return

        logger.info("\n--- Semantic Communication Pipeline Statistics ---")

        total_images = len(self._statistics)
        avg_timings: Dict[str, float] = {
            k: 0.0 for k in self._statistics[0]["timings"].keys()
        }
        avg_size_info: Dict[str, float] = {
            k: 0.0 for k in self._statistics[0]["size_info"].keys()
        }
        avg_metrics: Dict[str, float] = {
            "psnr": 0.0,
            "ssim": 0.0,
            "ms_ssim": 0.0,
            "lpips": 0.0,
            "compression_ratio": 0.0,
            "bpp": 0.0,
        }

        for stats in self._statistics:
            for k, v in stats["timings"].items():
                avg_timings[k] += v
            for k, v in stats["size_info"].items():
                avg_size_info[k] += v
            avg_metrics["psnr"] += stats.get("psnr", 0.0)
            avg_metrics["ssim"] += stats.get("ssim", 0.0)
            avg_metrics["ms_ssim"] += stats.get("ms_ssim", 0.0)
            avg_metrics["lpips"] += stats.get("lpips", 0.0)
            avg_metrics["compression_ratio"] += stats.get("compression_ratio", 0.0)
            avg_metrics["bpp"] += stats.get("bpp", 0.0)

        for k in avg_timings:
            avg_timings[k] /= total_images
        for k in avg_size_info:
            avg_size_info[k] /= total_images
        for k in avg_metrics:
            avg_metrics[k] /= total_images

        logger.info("\nAverage Timings (seconds):")
        for k, v in avg_timings.items():
            logger.info(f"  {k}: {v:.6f}")

        logger.info("\nAverage Size Info:")
        for k, v in avg_size_info.items():
            logger.info(f"  {k}: {v:.4f}")

        logger.info("\nAverage Metrics:")
        logger.info(f"  PSNR: {avg_metrics['psnr']:.4f}")
        logger.info(f"  SSIM: {avg_metrics['ssim']:.4f}")
        logger.info(f"  MS-SSIM: {avg_metrics['ms_ssim']:.4f}")
        logger.info(f"  LPIPS: {avg_metrics['lpips']:.4f}")
        logger.info(f"  Compression Ratio: {avg_metrics['compression_ratio']:.4f}")
        logger.info(f"  BPP: {avg_metrics['bpp']:.4f}")

        if self.enable_fid and self._fid_calculator is not None:
            logger.info("\nCalculating FID...")
            if (
                len(self._fid_calculator.real_features_list) >= self._min_fid_images
                and len(self._fid_calculator.fake_features_list) >= self._min_fid_images
                and len(self._fid_calculator.real_features_list)
                == len(self._fid_calculator.fake_features_list)
            ):
                if not self._fid_calculated:
                    try:
                        fid_score = self._fid_calculator.compute_fid()
                        avg_metrics["fid"] = fid_score
                        logger.info(f"  FID: {fid_score:.4f}")
                        self._fid_calculated = True
                    except Exception as e:
                        logger.error(f"  FID calculation failed: {e}")
                else:
                    logger.info("  FID already calculated.")
            else:
                logger.info(
                    f"  FID requires at least {self._min_fid_images} images to calculate. Only {len(self._fid_calculator.real_features_list)} processed."
                )
        else:
            logger.info("\nFID calculation disabled (enable_fid=False)")

        logger.info("\n--- End of Statistics ---")
        return avg_metrics

    def _calculate_total_elements_numel(self, data: Any) -> int:
        """Recursively calculates the total number of elements in all tensors within a nested structure."""
        total_elements = 0
        if isinstance(data, torch.Tensor):
            total_elements += data.numel()
        elif isinstance(data, (list, tuple)):
            for item in data:
                total_elements += self._calculate_total_elements_numel(item)
        elif isinstance(data, dict):
            for value in data.values():
                total_elements += self._calculate_total_elements_numel(value)
        return total_elements

    def _calculate_compression_metrics(
        self,
        input_data: Any,
        data_after_channel: Any,
        diagnostics: Dict[str, Any],
        original_tensor_for_metrics_batched: torch.Tensor = None,
    ) -> None:
        """Calculate compression metrics: CBR, compression ratio, and BPP."""
        size_info = diagnostics.get("size_info", {})
        tx_size_bits = size_info.get("total_bits", 0)

        # Prepare tensors for calculation
        orig_tensor_for_calculation = None
        if isinstance(input_data, torch.Tensor):
            orig_tensor_for_calculation = input_data.detach().cpu()
        elif isinstance(input_data, Image.Image):
            orig_tensor_for_calculation = transforms.ToTensor()(input_data).cpu()
        elif original_tensor_for_metrics_batched is not None:
            orig_tensor_for_calculation = (
                original_tensor_for_metrics_batched.detach().cpu()
            )

        if orig_tensor_for_calculation is None:
            return

        # Calculate original size
        original_size_bytes = (
            orig_tensor_for_calculation.element_size()
            * orig_tensor_for_calculation.numel()
        )

        # CBR calculation (only in eval mode)
        if not self.training:
            n = orig_tensor_for_calculation.numel()
            k = self._calculate_total_elements_numel(data_after_channel) / 2
            if n > 0:
                diagnostics["size_info"]["cbr"] = k / n
            else:
                diagnostics["size_info"]["cbr"] = float("inf")

        # Compression ratio calculation
        if tx_size_bits > 0:
            diagnostics["compression_ratio"] = original_size_bytes * 8 / tx_size_bits
        else:
            diagnostics["compression_ratio"] = float("inf")

        # BPP (Bits Per Pixel) calculation
        if tx_size_bits > 0 and len(orig_tensor_for_calculation.shape) >= 2:
            # For images: N = W * H (total pixels)
            # Assume format is [C, H, W] or [B, C, H, W]
            if orig_tensor_for_calculation.ndim == 3:  # [C, H, W]
                height, width = (
                    orig_tensor_for_calculation.shape[1],
                    orig_tensor_for_calculation.shape[2],
                )
            elif orig_tensor_for_calculation.ndim == 4:  # [B, C, H, W]
                height, width = (
                    orig_tensor_for_calculation.shape[2],
                    orig_tensor_for_calculation.shape[3],
                )
            elif orig_tensor_for_calculation.ndim == 2:  # [H, W] grayscale
                height, width = (
                    orig_tensor_for_calculation.shape[0],
                    orig_tensor_for_calculation.shape[1],
                )
            else:
                height = width = 0

            total_pixels = height * width
            if total_pixels > 0:
                diagnostics["bpp"] = tx_size_bits / total_pixels
            else:
                diagnostics["bpp"] = float("inf")
        else:
            diagnostics["bpp"] = 0.0

    def forward(self, input_data: Any) -> tuple[Any, dict]:
        """
        Passes input_data through the full semantic communication pipeline.

        Args:
            input_data: The initial data to be processed.

        Returns:
            A tuple containing:
            - rx_data: The final output from the receiver.
            - diagnostics: A dictionary containing timings, size_info, and intermediate data.
        """
        diagnostics = {"timings": {}, "size_info": {}}

        # Transmitter
        tx_start_time = time.time()
        tx_output, tx_timings = self.transmitter(input_data)
        diagnostics["timings"]["transmitter_total"] = time.time() - tx_start_time
        diagnostics["timings"].update({f"tx_{k}": v for k, v in tx_timings.items()})
        diagnostics["tx_representation"] = (
            tx_output  # Store the full output for diagnostics
        )

        # Extract the payload for channel and receiver
        # If 'payload' key exists, use it; otherwise, assume the whole output is the payload.
        tx_payload = tx_output.get("payload", tx_output)

        # Channel
        ch_start_time = time.time()
        data_after_channel, size_info = self.channel(tx_payload)
        diagnostics["timings"]["channel_total"] = time.time() - ch_start_time
        diagnostics["size_info"] = size_info

        # Receiver
        rx_start_time = time.time()
        rx_data, rx_timings = self.receiver(data_after_channel)
        diagnostics["timings"]["receiver_total"] = time.time() - rx_start_time
        diagnostics["timings"].update({f"rx_{k}": v for k, v in rx_timings.items()})

        diagnostics["timings"]["total_pipeline_time"] = (
            diagnostics["timings"]["transmitter_total"]
            + diagnostics["timings"]["channel_total"]
            + diagnostics["timings"]["receiver_total"]
        )

        # Performance optimization: compute metrics only in eval mode
        if not self.training:
            original_tensor_for_metrics = None
            rx_tensor_for_metrics = None
            current_device = None

            if isinstance(input_data, torch.Tensor):
                original_tensor_for_metrics = input_data.detach()
                current_device = input_data.device
            elif isinstance(input_data, Image.Image):
                original_tensor_for_metrics = transforms.ToTensor()(input_data)
                if isinstance(rx_data, torch.Tensor):
                    current_device = rx_data.device
                    original_tensor_for_metrics = original_tensor_for_metrics.to(
                        current_device
                    )

            if isinstance(rx_data, torch.Tensor):
                rx_tensor_for_metrics = rx_data.detach()
                current_device = rx_data.device

            if (
                original_tensor_for_metrics is not None
                and rx_tensor_for_metrics is not None
            ):
                if original_tensor_for_metrics.ndim == 3:
                    original_tensor_for_metrics_batched = (
                        original_tensor_for_metrics.unsqueeze(0)
                    )
                else:
                    original_tensor_for_metrics_batched = original_tensor_for_metrics

                if rx_tensor_for_metrics.ndim == 3:
                    rx_tensor_for_metrics_batched = rx_tensor_for_metrics.unsqueeze(0)
                elif (
                    rx_tensor_for_metrics.ndim == 4
                    and rx_tensor_for_metrics.shape[0] == 1
                ):
                    rx_tensor_for_metrics_batched = rx_tensor_for_metrics.squeeze(0)
                else:
                    rx_tensor_for_metrics_batched = rx_tensor_for_metrics

                original_tensor_for_metrics_batched = torch.clamp(
                    original_tensor_for_metrics_batched, 0, 1
                )
                rx_tensor_for_metrics_batched = torch.clamp(
                    rx_tensor_for_metrics_batched, 0, 1
                )

                per_image_metrics = {
                    "psnr": calculate_psnr(
                        original_tensor_for_metrics_batched,
                        rx_tensor_for_metrics_batched,
                        device=current_device,
                    ),
                    "ssim": calculate_ssim(
                        original_tensor_for_metrics_batched,
                        rx_tensor_for_metrics_batched,
                        device=current_device,
                    ),
                    "ms_ssim": calculate_ms_ssim(
                        original_tensor_for_metrics_batched,
                        rx_tensor_for_metrics_batched,
                        device=current_device,
                    ),
                    "lpips": calculate_lpips(
                        original_tensor_for_metrics_batched,
                        rx_tensor_for_metrics_batched,
                        device=current_device,
                    ),
                }
                diagnostics.update(per_image_metrics)

                # Calculate compression metrics (CBR, compression ratio, BPP)
                self._calculate_compression_metrics(
                    input_data,
                    data_after_channel,
                    diagnostics,
                    original_tensor_for_metrics_batched,
                )

                # FID calculation - only if enabled to avoid expensive CPU transfers
                if self.enable_fid and self._fid_calculator is not None:
                    original_cpu = original_tensor_for_metrics_batched.cpu()
                    rx_cpu = rx_tensor_for_metrics_batched.cpu()
                    self._fid_calculator.update_features(original_cpu, real=True)
                    self._fid_calculator.update_features(rx_cpu, real=False)

            self._statistics.append(diagnostics)
        else:
            # Training mode: compute basic compression metrics
            self._calculate_compression_metrics(
                input_data, data_after_channel, diagnostics
            )

        return rx_data, diagnostics

    def to(self, *args, **kwargs):
        """Overrides nn.Module.to() to move all sub-modules to the specified device/dtype."""
        super().to(*args, **kwargs)
        self.transmitter.to(*args, **kwargs)
        self.channel.to(*args, **kwargs)
        self.receiver.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        """Overrides nn.Module.train() to set all sub-modules to train/eval mode."""
        super().train(mode)
        self.transmitter.train(mode)
        self.channel.train(mode)
        self.receiver.train(mode)
        return self

    def eval(self):
        """Sets all sub-modules to evaluation mode."""
        return self.train(False)
