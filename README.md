# BASIL Encoder-Decoder

Shared CNN autoencoder for the BASIL (Budget Aware State Imagination Learning) framework.

## Structure

```
basil/
├── models/
│   ├── __init__.py
│   └── encoder_decoder.py   # Encoder, Decoder, BoardAutoEncoder
├── envs/
│   ├── __init__.py
│   ├── tictactoe.py          # TicTacToe environment
│   ├── connect4.py           # Connect4 environment
│   └── generate_data.py      # Data collection + padding
├── train_autoencoder.py      # Training + 7 verification checks
└── README.md
```

## Quick Start

```bash
cd basil
pip install torch numpy
python train_autoencoder.py
```

This will:
1. Generate 5000 random game states each from TicTacToe and Connect4
2. Pad all states to 7×7×3 and shuffle into a mixed dataset
3. Train the autoencoder for 30 epochs with MSE loss
4. Save a checkpoint to `checkpoints/autoencoder.pt`
5. Run 7 verification checks (shapes, output range, reconstruction, etc.)

## Architecture (Paper Section 3)

**Encoder**: 7×7×3 padded board → 64-dim latent vector
- Conv2d(3→32, 3×3, pad=1) + ReLU
- Conv2d(32→64, 3×3, pad=1) + ReLU
- Conv2d(64→64, 3×3, pad=1) + ReLU
- Flatten(3136) → Linear(3136, 64)

**Decoder**: mirrors encoder with ConvTranspose2d, Sigmoid output

**Loss**: `L_enc = ‖s − Dec(Enc(s))‖²` (MSE)

## Using the Encoder in BASIL Pipeline

```python
from models.encoder_decoder import Encoder
import torch

encoder = Encoder(in_channels=3, latent_dim=64)
ckpt = torch.load("checkpoints/autoencoder.pt")
encoder.load_state_dict(ckpt["encoder_state_dict"])
encoder.eval()

# board_tensor: (batch, 3, 7, 7) padded CHW board
z = encoder(board_tensor)  # (batch, 64) → feed to Mamba / PPO
```

## Hyperparameters

Edit constants at the top of `train_autoencoder.py`:

| Parameter | Default | Paper |
|-----------|---------|-------|
| LATENT_DIM | 64 | 64 |
| BATCH_SIZE | 256 | — |
| EPOCHS | 30 | — |
| LR | 1e-3 | — |
| NUM_STATES_PER_GAME | 5000 | 50000 |
| TARGET_MSE | 0.1 | < 0.1 |

For full training, set `NUM_STATES_PER_GAME=50000` and `EPOCHS=100`.
