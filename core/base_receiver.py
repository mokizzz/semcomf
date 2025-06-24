from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn


class BaseReceiver(nn.Module, ABC):
    """
    Abstract base class for the receiver in a semantic communication system.
    The receiver reconstructs the original data from the received semantic representation.
    """

    def __init__(self, config: dict | None = None):
        super().__init__()
        self.config = config if config is not None else {}

    @abstractmethod
    def forward(self, received_data: Any) -> tuple[Any, dict[str, float]]:
        """
        Processes the received data and reconstructs the original data.

        Args:
            received_data: The data output by the channel.

        Returns:
            A tuple containing:
            - reconstructed_data: The reconstructed data (e.g., PIL Image, torch.Tensor).
            - timings: A dictionary logging the time taken for different operations.
                       Example: {"decoding_time": 0.2, "reconstruction_time": 0.15}
        """
        pass

    def get_config(self) -> dict:
        return self.config
