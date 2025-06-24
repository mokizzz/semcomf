import time
from typing import Any, Dict, List

import torch
import torch.nn as nn
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
        self, transmitter: nn.Module, channel: BaseChannel, receiver: nn.Module
    ):
        super().__init__()
        self.transmitter = transmitter
        self.channel = channel
        self.receiver = receiver
        self._statistics: List[Dict[str, Any]] = []
        self._fid_calculator = FIDCalculator()
        self._fid_calculated = False
        self._min_fid_images = 2

    def reset_statistics(self):
        """Clears all collected statistics and resets the FID calculator."""
        self._statistics = []
        self._fid_calculator.reset()
        self._fid_calculated = False

    def output_statistics(self):
        """Outputs the collected statistics."""
        if not self._statistics:
            print("No statistics collected yet.")
            return

        print("\n--- Semantic Communication Pipeline Statistics ---")

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

        for k in avg_timings:
            avg_timings[k] /= total_images
        for k in avg_size_info:
            avg_size_info[k] /= total_images
        for k in avg_metrics:
            avg_metrics[k] /= total_images

        print("\nAverage Timings (seconds):")
        for k, v in avg_timings.items():
            print(f"  {k}: {v:.6f}")

        print("\nAverage Size Info:")
        for k, v in avg_size_info.items():
            print(f"  {k}: {v:.2f}")

        print("\nAverage Metrics:")
        print(f"  PSNR: {avg_metrics['psnr']:.4f}")
        print(f"  SSIM: {avg_metrics['ssim']:.4f}")
        print(f"  MS-SSIM: {avg_metrics['ms_ssim']:.4f}")
        print(f"  LPIPS: {avg_metrics['lpips']:.4f}")
        print(f"  Compression Ratio: {avg_metrics['compression_ratio']:.2f}")

        print("\nCalculating FID...")
        if (
            len(self._fid_calculator.real_features_list) >= self._min_fid_images
            and len(self._fid_calculator.fake_features_list) >= self._min_fid_images
            and len(self._fid_calculator.real_features_list)
            == len(self._fid_calculator.fake_features_list)
        ):
            if not self._fid_calculated:
                try:
                    fid_score = self._fid_calculator.compute_fid()
                    print(f"  FID: {fid_score:.4f}")
                    self._fid_calculated = True
                except Exception as e:
                    print(f"  FID calculation failed: {e}")
            else:
                print("  FID already calculated.")
        else:
            print(
                f"  FID requires at least {self._min_fid_images} images to calculate. Only {len(self._fid_calculator.real_features_list)} processed."
            )

        print("\n--- End of Statistics ---")

    def forward(self, input_data: Any) -> tuple[Any, dict]:
        """
        Passes input_data through the full semantic communication pipeline.

        Args:
            input_data: The initial data to be processed.

        Returns:
            A tuple containing:
            - reconstructed_data: The final output from the receiver.
            - diagnostics: A dictionary containing timings, size_info, and intermediate data.
        """
        diagnostics = {"timings": {}, "size_info": {}}

        # CBR (Channel Bandwidth Ratio) preparation
        orig_tensor_for_cbr = None
        if isinstance(input_data, torch.Tensor):
            orig_tensor_for_cbr = input_data.detach().cpu()
        elif isinstance(input_data, Image.Image):
            orig_tensor_for_cbr = transforms.ToTensor()(input_data).cpu()

        # Transmitter
        tx_start_time = time.time()
        transmitted_representation, tx_timings = self.transmitter(input_data)
        diagnostics["timings"]["transmitter_total"] = time.time() - tx_start_time
        diagnostics["timings"].update({f"tx_{k}": v for k, v in tx_timings.items()})

        # Channel
        ch_start_time = time.time()
        data_after_channel, size_info = self.channel(transmitted_representation)
        diagnostics["timings"]["channel_total"] = time.time() - ch_start_time
        diagnostics["size_info"] = size_info

        # CBR
        if orig_tensor_for_cbr is not None and isinstance(
            data_after_channel, torch.Tensor
        ):
            n = orig_tensor_for_cbr.numel()
            k = data_after_channel.numel() / 2
            diagnostics["size_info"]["cbr"] = k / n

        # Receiver
        rx_start_time = time.time()
        reconstructed_data, rx_timings = self.receiver(data_after_channel)
        diagnostics["timings"]["receiver_total"] = time.time() - rx_start_time
        diagnostics["timings"].update({f"rx_{k}": v for k, v in rx_timings.items()})

        diagnostics["timings"]["total_pipeline_time"] = (
            diagnostics["timings"]["transmitter_total"]
            + diagnostics["timings"]["channel_total"]
            + diagnostics["timings"]["receiver_total"]
        )

        original_tensor_for_metrics = None
        reconstructed_tensor_for_metrics = None

        if isinstance(input_data, torch.Tensor):
            original_tensor_for_metrics = input_data.detach().cpu()
        elif isinstance(input_data, Image.Image):
            original_tensor_for_metrics = transforms.ToTensor()(input_data).cpu()

        if isinstance(reconstructed_data, torch.Tensor):
            reconstructed_tensor_for_metrics = reconstructed_data.detach().cpu()

        if (
            original_tensor_for_metrics is not None
            and reconstructed_tensor_for_metrics is not None
        ):
            if original_tensor_for_metrics.ndim == 3:
                original_tensor_for_metrics_batched = (
                    original_tensor_for_metrics.unsqueeze(0)
                )
            else:
                original_tensor_for_metrics_batched = original_tensor_for_metrics

            if reconstructed_tensor_for_metrics.ndim == 3:
                reconstructed_tensor_for_metrics_batched = (
                    reconstructed_tensor_for_metrics.unsqueeze(0)
                )
            elif (
                reconstructed_tensor_for_metrics.ndim == 4
                and reconstructed_tensor_for_metrics.shape[0] == 1
            ):
                reconstructed_tensor_for_metrics_batched = (
                    reconstructed_tensor_for_metrics.squeeze(0)
                )
            else:
                reconstructed_tensor_for_metrics_batched = (
                    reconstructed_tensor_for_metrics
                )

            per_image_metrics = {
                "psnr": calculate_psnr(
                    original_tensor_for_metrics_batched,
                    reconstructed_tensor_for_metrics_batched,
                    data_range=255.0,
                ),
                "ssim": calculate_ssim(
                    original_tensor_for_metrics_batched,
                    reconstructed_tensor_for_metrics_batched,
                    data_range=255.0,
                ),
                "ms_ssim": calculate_ms_ssim(
                    original_tensor_for_metrics_batched,
                    reconstructed_tensor_for_metrics_batched,
                    data_range=255.0,
                ),
                "lpips": calculate_lpips(
                    original_tensor_for_metrics_batched,
                    reconstructed_tensor_for_metrics_batched,
                ),
            }
            diagnostics.update(per_image_metrics)

            original_size_bytes = (
                original_tensor_for_metrics_batched.element_size()
                * original_tensor_for_metrics_batched.numel()
            )
            transmitted_size_bits = diagnostics["size_info"].get("total_bits", 0)
            if transmitted_size_bits > 0:
                diagnostics["compression_ratio"] = (
                    original_size_bytes * 8
                ) / transmitted_size_bits
            else:
                diagnostics["compression_ratio"] = float("inf")

            self._fid_calculator.update_features(
                original_tensor_for_metrics_batched, real=True
            )
            self._fid_calculator.update_features(
                reconstructed_tensor_for_metrics_batched, real=False
            )

        self._statistics.append(diagnostics)

        return reconstructed_data, diagnostics

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
