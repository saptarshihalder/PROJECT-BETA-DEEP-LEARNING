"""
BASIL Shared CNN Encoder-Decoder

Paper spec (Section 3):
  - Maps variable-sized boards to a fixed 64-dim latent vector
  - Input: s in R^(H x W x C), C=3 (player1, player2, empty)
  - All boards padded to 7x7x3 (from generate_data.py PAD_H=7, PAD_W=7)
  - Encoder: 3 Conv2D layers (3 -> 32 -> 64 -> 64) with ReLU, flatten, linear -> z in R^64
  - Decoder: mirrors encoder for autoencoder reconstruction
  - Loss: L_enc = ||s - Dec(Enc(s))||^2   (MSE)
  - Trained on mixed data from all games for game-agnostic features
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    CNN encoder: padded board (7x7x3) -> 64-dim latent vector.

    Architecture from paper Section 3:
        Conv2D(3  -> 32, 3x3, pad=1) + ReLU   -> (B, 32, 7, 7)
        Conv2D(32 -> 64, 3x3, pad=1) + ReLU   -> (B, 64, 7, 7)
        Conv2D(64 -> 64, 3x3, pad=1) + ReLU   -> (B, 64, 7, 7)
        Flatten                                 -> (B, 3136)
        Linear(3136, 64)                        -> (B, 64)

    padding=1 preserves spatial dims throughout so the flatten dim
    is deterministic: 64 channels * 7 * 7 = 3136.
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 64,
                 pad_h: int = 7, pad_w: int = 7):
        super().__init__()
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.latent_dim = latent_dim

        # Three Conv2D layers: C -> 32 -> 64 -> 64 (paper Section 3)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Flatten and project to latent_dim
        self.flat_dim = 64 * pad_h * pad_w  # 64 * 7 * 7 = 3136
        self.fc = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, H=7, W=7) padded board state  [CHW format]
        Returns:
            z: (batch, 64) latent vector
        """
        h = self.conv_layers(x)           # (B, 64, 7, 7)
        h = h.reshape(h.size(0), -1)      # (B, 3136)
        z = self.fc(h)                     # (B, 64)
        return z


class Decoder(nn.Module):
    """
    CNN decoder: 64-dim latent -> reconstructed board (7x7x3).

    Mirrors the encoder symmetrically:
        Linear(64, 3136)  + Reshape(64, 7, 7)
        ConvTranspose2d(64 -> 64, 3x3, pad=1) + ReLU
        ConvTranspose2d(64 -> 32, 3x3, pad=1) + ReLU
        ConvTranspose2d(32 ->  3, 3x3, pad=1) + Sigmoid

    Sigmoid on the final layer because board channels are binary
    indicators (0.0 or 1.0).
    """

    def __init__(self, out_channels: int = 3, latent_dim: int = 64,
                 pad_h: int = 7, pad_w: int = 7):
        super().__init__()
        self.pad_h = pad_h
        self.pad_w = pad_w

        self.flat_dim = 64 * pad_h * pad_w  # 3136
        self.fc = nn.Linear(latent_dim, self.flat_dim)

        # Mirror of encoder conv layers (channels reversed)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, 64) latent vector
        Returns:
            x_hat: (batch, 3, 7, 7) reconstructed board
        """
        h = self.fc(z)                                        # (B, 3136)
        h = h.reshape(h.size(0), 64, self.pad_h, self.pad_w)  # (B, 64, 7, 7)
        x_hat = self.deconv_layers(h)                          # (B, 3, 7, 7)
        return x_hat


class BoardAutoEncoder(nn.Module):
    """
    Full autoencoder combining Encoder + Decoder.

    Training objective (paper Section 3):
        L_enc = ||s - Dec(Enc(s))||^2     (MSE reconstruction loss)

    Trained on mixed padded states from TicTacToe + Connect4 so the
    encoder learns game-agnostic features such as threatening lines,
    blocking moves, and board occupancy patterns.

    After training, the encoder alone is used as the "shared CNN encoder"
    component of the full BASIL pipeline — its 64-dim output feeds into
    the policy/value networks and the Mamba world model.
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 64,
                 pad_h: int = 7, pad_w: int = 7):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim, pad_h, pad_w)
        self.decoder = Decoder(in_channels, latent_dim, pad_h, pad_w)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, 3, 7, 7) padded board
        Returns:
            x_hat: (batch, 3, 7, 7) reconstruction
            z:     (batch, 64) latent representation
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Board -> latent. Used at inference by the rest of BASIL."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Latent -> board reconstruction."""
        return self.decoder(z)
