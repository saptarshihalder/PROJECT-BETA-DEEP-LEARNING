import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.encoder_decoder import BoardAutoEncoder
from envs.tictactoe import TicTacToe
from envs.connect4 import ConnectFour
from envs.generate_data import pad_state, PAD_H, PAD_W
LATENT_DIM           = 64       # paper: 64-dim latent
BATCH_SIZE           = 256
EPOCHS               = 30
LR                   = 1e-3     # Adam learning rate
NUM_STATES_PER_GAME  = 5000     # states per game for dataset
TARGET_MSE           = 0.1      # paper Section 5: target MSE < 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collect_states(env_class, n):
    """Play random games and collect n board states."""
    env = env_class()
    states = []
    while len(states) < n:
        state = env.reset()
        states.append(state.copy())
        done = False
        while not done and len(states) < n:
            action = random.choice(env.get_valid_actions())
            state, _, done, _ = env.step(action)
            states.append(state.copy())
    return states


def make_dataloaders():
    print("Generating game states...")
    ttt = collect_states(TicTacToe, NUM_STATES_PER_GAME)
    c4  = collect_states(ConnectFour, NUM_STATES_PER_GAME)
    print(f"  TTT: {len(ttt)}, C4: {len(c4)}")

    # Pad to 7x7x3 and shuffle
    all_padded = [pad_state(s) for s in ttt] + [pad_state(s) for s in c4]
    random.shuffle(all_padded)

    # Convert HWC -> CHW for PyTorch
    arr = np.array(all_padded).transpose(0, 3, 1, 2)   # (N, 3, 7, 7)
    tensor = torch.tensor(arr, dtype=torch.float32)

    # 90/10 train/val split
    n_train = int(0.9 * len(tensor))
    train_ds = TensorDataset(tensor[:n_train])
    val_ds   = TensorDataset(tensor[n_train:])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    print(f"  Train: {n_train}, Val: {len(tensor) - n_train}\n")
    return train_loader, val_loader
def train(model, train_loader, val_loader):
    """
    Train autoencoder with MSE loss.
    Paper loss: L_enc = ||s - Dec(Enc(s))||^2
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print("Training autoencoder...")
    print(f"{'Epoch':>7} | {'Train MSE':>12} | {'Val MSE':>12}")
    print("-" * 38)

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(DEVICE)
            x_hat, _ = model(batch)
            loss = criterion(x_hat, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch.size(0)
        train_loss /= len(train_loader.dataset)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(DEVICE)
                x_hat, _ = model(batch)
                val_loss += criterion(x_hat, batch).item() * batch.size(0)
        val_loss /= len(val_loader.dataset)

        # Print every 5 epochs + first and last
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == EPOCHS - 1:
            print(f"  {epoch+1:3d}/{EPOCHS}  |  {train_loss:.6f}    |  {val_loss:.6f}")

    return val_loss
def verify(model):
    model.eval()
    results = []

    def check(name, passed):
        tag = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
        results.append(passed)
        print(f"  [{tag}] {name}")

    print("\n" + "=" * 64)
    print("  VERIFICATION SUITE")
    print("=" * 64)

    # ── 1. Encoder/Decoder shape ─────────────────────────────────
    print("\n1. Shape checks")
    dummy = torch.randn(4, 3, PAD_H, PAD_W).to(DEVICE)
    x_hat, z = model(dummy)
    check(
        f"Encoder: {tuple(z.shape)} == (4, {LATENT_DIM})",
        z.shape == (4, LATENT_DIM),
    )
    check(
        f"Decoder: {tuple(x_hat.shape)} == (4, 3, {PAD_H}, {PAD_W})",
        x_hat.shape == (4, 3, PAD_H, PAD_W),
    )

    # ── 2. Decoder output in [0, 1] (sigmoid) ───────────────────
    print("\n2. Output range (sigmoid -> [0, 1])")
    lo, hi = x_hat.min().item(), x_hat.max().item()
    check(f"min={lo:.4f} >= 0, max={hi:.4f} <= 1", lo >= 0.0 and hi <= 1.0)

    # ── 3. Reconstruction quality per game ───────────────────────
    print(f"\n3. Per-game reconstruction (target MSE < {TARGET_MSE})")
    for name, EnvClass in [("TicTacToe", TicTacToe), ("Connect4", ConnectFour)]:
        env = EnvClass()
        state = env.reset()
        for _ in range(4):
            va = env.get_valid_actions()
            if not va:
                break
            state, _, done, _ = env.step(random.choice(va))
            if done:
                break

        padded = pad_state(state)
        x = torch.tensor(padded.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            x_hat_s, _ = model(x)
        mse = nn.functional.mse_loss(x_hat_s, x).item()
        check(f"{name:12s}  MSE = {mse:.6f}", mse < TARGET_MSE)

    # ── 4. Different states -> different latents ─────────────────
    print("\n4. Latent differentiation (different boards != same z)")
    env = ConnectFour()
    board_states = []
    state = env.reset()
    board_states.append(pad_state(state))
    for _ in range(5):
        va = env.get_valid_actions()
        if not va:
            break
        state, _, done, _ = env.step(random.choice(va))
        board_states.append(pad_state(state))
        if done:
            break

    batch = torch.tensor(
        np.array([s.transpose(2, 0, 1) for s in board_states]), dtype=torch.float32
    ).to(DEVICE)
    with torch.no_grad():
        z_batch = model.encode(batch)

    dists = torch.cdist(z_batch, z_batch)
    mask = ~torch.eye(len(board_states), dtype=bool, device=DEVICE)
    min_dist = dists[mask].min().item()
    check(f"Min pairwise L2 dist = {min_dist:.4f} > 0.01", min_dist > 0.01)

    # ── 5. Determinism (same input -> same output) ───────────────
    print("\n5. Determinism")
    x_test = batch[:1]
    with torch.no_grad():
        z1 = model.encode(x_test)
        z2 = model.encode(x_test)
    check("Same input -> identical latent", torch.allclose(z1, z2, atol=1e-6))

    # ── 6. Batch-size invariance ─────────────────────────────────
    print("\n6. Batch-size invariance")
    with torch.no_grad():
        z_single = model.encode(batch[:1])
        z_multi  = model.encode(batch)
    check(
        "z[0] matches regardless of batch size",
        torch.allclose(z_single[0], z_multi[0], atol=1e-5),
    )

    # ── 7. Parameter counts ──────────────────────────────────────
    print("\n7. Parameter counts")
    enc_p = sum(p.numel() for p in model.encoder.parameters())
    dec_p = sum(p.numel() for p in model.decoder.parameters())
    print(f"     Encoder:  {enc_p:>8,} params")
    print(f"     Decoder:  {dec_p:>8,} params")
    print(f"     Total:    {enc_p + dec_p:>8,} params")

    # ── Summary ──────────────────────────────────────────────────
    n_pass = sum(results)
    n_total = len(results)
    print("\n" + "=" * 64)
    if all(results):
        print(f"  ALL {n_total} CHECKS PASSED")
    else:
        print(f"  {n_pass}/{n_total} CHECKS PASSED — review failures above")
    print("=" * 64)
    return all(results)
def main():
    print("=" * 64)
    print("  BASIL Autoencoder Training")
    print("=" * 64)
    print(f"  Device:      {DEVICE}")
    print(f"  Latent dim:  {LATENT_DIM}")
    print(f"  Board size:  {PAD_H}x{PAD_W}x3 (padded)")
    print(f"  Epochs:      {EPOCHS}")
    print(f"  Batch size:  {BATCH_SIZE}")
    print(f"  LR:          {LR}")
    print()

    # ── Data ──
    train_loader, val_loader = make_dataloaders()

    # ── Model ──
    model = BoardAutoEncoder(
        in_channels=3, latent_dim=LATENT_DIM, pad_h=PAD_H, pad_w=PAD_W
    ).to(DEVICE)
    print(f"Model:\n{model}\n")

    # ── Train ──
    final_mse = train(model, train_loader, val_loader)
    mse_ok = final_mse < TARGET_MSE
    status = "PASS" if mse_ok else "NEEDS MORE EPOCHS"
    print(f"\nFinal val MSE: {final_mse:.6f}  (target < {TARGET_MSE}) [{status}]")

    # ── Save checkpoint ──
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/autoencoder.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "latent_dim": LATENT_DIM,
            "pad_h": PAD_H,
            "pad_w": PAD_W,
            "final_val_mse": final_mse,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint -> {ckpt_path}")

    # ── Verify ──
    verify(model)

    # ── Show how to load encoder for downstream use ──
    print("\n--- Usage in BASIL pipeline ---")
    print("  from models.encoder_decoder import Encoder")
    print("  encoder = Encoder(in_channels=3, latent_dim=64)")
    print("  ckpt = torch.load('checkpoints/autoencoder.pt')")
    print("  encoder.load_state_dict(ckpt['encoder_state_dict'])")
    print("  encoder.eval()")
    print("  z = encoder(board_tensor)  # (B, 64) latent for Mamba/PPO")


if __name__ == "__main__":
    main()
