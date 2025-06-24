from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn


class BaseTransmitter(nn.Module, ABC):
    """
    Abstract base class for the transmitter in a semantic communication system.
    The transmitter encodes the input data into a semantic representation.
    """

    def __init__(self, config: dict | None = None):
        super().__init__()
        self.config = config if config is not None else {}

    @abstractmethod
    def forward(self, input_data: Any) -> tuple[Any, dict[str, float]]:
        """
        Processes the input data and returns the transmitted representation
        and a dictionary of timings for different stages.

        Args:
            input_data: The data to be transmitted (e.g., PIL Image, torch.Tensor, video frames).

        Returns:
            A tuple containing:
            - transmitted_representation: The semantic representation to be sent over the channel.
                                          This can be a dictionary of features, tensors, etc.
            - timings: A dictionary logging the time taken for different operations.
                       Example: {"encoding_time": 0.1, "feature_extraction_time": 0.05}
        """
        pass

    def get_config(self) -> dict:
        return self.config
