import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image


class ConvGRUCell(nn.Module):
    """A single cell of a Convolutional GRU."""

    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv_gates = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=2 * hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=True,
        )

        self.conv_can = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=True,
        )

    def forward(self, input_tensor, hidden_state):
        if hidden_state is None:
            b, _, h, w = input_tensor.size()
            hidden_state = torch.zeros(
                b, self.hidden_dim, h, w, device=input_tensor.device
            )

        combined = torch.cat([input_tensor, hidden_state], dim=1)

        combined_conv = self.conv_gates(combined)
        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined_reset = torch.cat([input_tensor, reset_gate * hidden_state], dim=1)
        candidate_hidden = torch.tanh(self.conv_can(combined_reset))

        new_hidden_state = (
            1 - update_gate
        ) * hidden_state + update_gate * candidate_hidden

        return new_hidden_state


class SimpleVideoEncoder(nn.Module):
    """
    Encodes a video frame based on the previous reconstructed frame.
    For I-frames, it uses an image encoder.
    For P-frames, it encodes the difference/update information.
    """

    def __init__(self, input_channels=3, base_channels=32, latent_dim=128):
        super().__init__()

        self.image_encoder = nn.Sequential(
            self._conv_block(input_channels, base_channels),
            nn.MaxPool2d(2),
            self._conv_block(base_channels, base_channels * 2),
            nn.MaxPool2d(2),
            self._conv_block(base_channels * 2, base_channels * 4),
            nn.MaxPool2d(2),
            self._conv_block(base_channels * 4, latent_dim),
        )

        self.update_feature_extractor = nn.Sequential(
            self._conv_block(input_channels * 2, base_channels),
            nn.MaxPool2d(2),
            self._conv_block(base_channels, base_channels * 2),
            nn.MaxPool2d(2),
            self._conv_block(base_channels * 2, base_channels * 4),
            nn.MaxPool2d(2),
            self._conv_block(base_channels * 4, latent_dim),
        )

        self.gru_cell = ConvGRUCell(
            input_dim=latent_dim, hidden_dim=latent_dim, kernel_size=3
        )

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(True),
        )

    def forward(self, current_frame, prev_recon_frame=None, prev_hidden_state=None):
        if prev_recon_frame is None:
            features = self.image_encoder(current_frame)
            current_hidden_state = self.gru_cell(features, prev_hidden_state)
            features_to_transmit = current_hidden_state
        else:
            combined_frames = torch.cat([current_frame, prev_recon_frame], dim=1)
            update_features = self.update_feature_extractor(combined_frames)
            current_hidden_state = self.gru_cell(update_features, prev_hidden_state)
            features_to_transmit = current_hidden_state

        return features_to_transmit, current_hidden_state


class SimpleVideoDecoder(nn.Module):
    """
    Decodes a video frame based on the received features and its previous state.
    """

    def __init__(self, output_channels=3, base_channels=32, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        self.gru_cell = ConvGRUCell(
            input_dim=latent_dim, hidden_dim=latent_dim, kernel_size=3
        )

        self.feature_decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_channels * 4, kernel_size=2, stride=2),
            self._conv_block(base_channels * 4, base_channels * 4),
            nn.ConvTranspose2d(
                base_channels * 4, base_channels * 2, kernel_size=2, stride=2
            ),
            self._conv_block(base_channels * 2, base_channels * 2),
            nn.ConvTranspose2d(
                base_channels * 2, base_channels, kernel_size=2, stride=2
            ),
            self._conv_block(base_channels, base_channels),
            nn.Conv2d(base_channels, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(True),
        )

    def forward(self, received_features, prev_hidden_state=None):
        current_hidden_state = self.gru_cell(received_features, prev_hidden_state)
        reconstructed_frame = self.feature_decoder(current_hidden_state)

        return reconstructed_frame, current_hidden_state
