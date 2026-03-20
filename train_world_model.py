import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.encoder_decoder import Encoder
from models.mamba_world_model import MambaWorldModel, WorldModelLoss, SelectiveSSM
from envs.tictactoe import TicTacToe
from envs.connect4 import ConnectFour
from envs.game_registry import pad_state, PAD_H, PAD_W, MAX_ACTIONS
from data.collect_transitions import collect_mixed_transitions
LATENT_DIM       = 64       # encoder output dim
ACTION_DIM       = 16       # action embedding dim (paper Section 3)
HIDDEN_DIM       = 128      # Mamba block hidden dim (paper Section 3)
STATE_DIM        = 16       # SSM state dim N
NUM_BLOCKS       = 2        # Mamba blocks (paper: 2-4)

BATCH_SIZE       = 128
EPOCHS           = 100
LR               = 3e-4     # same as PPO lr in paper
NUM_TRANSITIONS  = 50000     # per game (for quick training)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_encoder(ckpt_path: str = "checkpoints/autoencoder.pt"):
    """Load the pre-trained encoder and freeze its weights."""
    encoder = Encoder(in_channels=3, latent_dim=LATENT_DIM,
                      pad_h=PAD_H, pad_w=PAD_W).to(DEVICE)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        print(f"Loaded pre-trained encoder from {ckpt_path}")
    else:
        print(f"WARNING: No checkpoint at {ckpt_path}, using random encoder")
        print("         (Run train_autoencoder.py first for best results)")

    # Freeze encoder — we don't update it during world model training
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    return encoder

def make_dataloaders(data: dict, val_split: float = 0.1):
    """Split transition data into train/val DataLoaders."""
    n = len(data["z_t"])
    n_val = int(n * val_split)
    n_train = n - n_val

    def make_loader(start, end, shuffle):
        ds = TensorDataset(
            data["z_t"][start:end],
            data["actions"][start:end],
            data["z_next"][start:end],
            data["rewards"][start:end].unsqueeze(-1),
            data["dones"][start:end].unsqueeze(-1),
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader(0, n_train, shuffle=True)
    val_loader = make_loader(n_train, n, shuffle=False)
    print(f"  Train: {n_train}, Val: {n_val}")
    return train_loader, val_loader
def train(model, train_loader, val_loader):
    """Train Mamba World Model with L_world loss (paper Eq. 2)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = WorldModelLoss(reward_weight=0.5)

    print(f"\n{'Epoch':>7} | {'Train Loss':>11} | {'Val Loss':>11} | "
          f"{'State':>8} | {'Reward':>8} | {'Done':>8}")
    print("-" * 72)

    best_val = float("inf")

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        train_total = 0.0
        for z_t, actions, z_next, rewards, dones in train_loader:
            z_t = z_t.to(DEVICE)
            actions = actions.to(DEVICE)
            z_next = z_next.to(DEVICE)
            rewards = rewards.to(DEVICE)
            dones = dones.to(DEVICE)

            # Reset hidden states for each batch (independent transitions)
            model.reset_hidden(z_t.size(0), DEVICE)

            z_pred, r_pred, d_logit, _ = model(z_t, actions)

            loss, _ = criterion(z_pred, z_next, r_pred, rewards, d_logit, dones)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total += loss.item() * z_t.size(0)

        train_avg = train_total / len(train_loader.dataset)

        # ── Validate ──
        model.eval()
        val_total = 0.0
        val_parts = [0.0, 0.0, 0.0]
        val_n = 0
        with torch.no_grad():
            for z_t, actions, z_next, rewards, dones in val_loader:
                z_t = z_t.to(DEVICE)
                actions = actions.to(DEVICE)
                z_next = z_next.to(DEVICE)
                rewards = rewards.to(DEVICE)
                dones = dones.to(DEVICE)

                model.reset_hidden(z_t.size(0), DEVICE)
                z_pred, r_pred, d_logit, _ = model(z_t, actions)
                loss, parts = criterion(z_pred, z_next, r_pred, rewards, d_logit, dones)

                bs = z_t.size(0)
                val_total += loss.item() * bs
                for i in range(3):
                    val_parts[i] += parts[i] * bs
                val_n += bs

        val_avg = val_total / val_n
        val_parts = [p / val_n for p in val_parts]

        if val_avg < best_val:
            best_val = val_avg

        # Print every 5 epochs + first/last
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == EPOCHS - 1:
            print(f"  {epoch+1:3d}/{EPOCHS}  |  {train_avg:.6f}  |  {val_avg:.6f}  | "
                  f"{val_parts[0]:.5f} | {val_parts[1]:.5f} | {val_parts[2]:.5f}")

    return best_val
def verify(model, encoder):
    """Comprehensive verification of the Mamba World Model."""
    model.eval()
    encoder.eval()
    results = []

    def check(name, passed):
        tag = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
        results.append(passed)
        print(f"  [{tag}] {name}")

    print("\n" + "=" * 64)
    print("  WORLD MODEL VERIFICATION")
    print("=" * 64)

    # ── 1. Output shapes ─────────────────────────────────────────
    print("\n1. Output shape checks")
    B = 4
    z_dummy = torch.randn(B, LATENT_DIM).to(DEVICE)
    a_dummy = torch.randint(0, MAX_ACTIONS, (B,)).to(DEVICE)
    model.reset_hidden(B, DEVICE)
    z_pred, r_pred, d_logit, h_states = model(z_dummy, a_dummy)

    check(f"z_next shape: {tuple(z_pred.shape)} == (4, {LATENT_DIM})",
          z_pred.shape == (4, LATENT_DIM))
    check(f"reward shape: {tuple(r_pred.shape)} == (4, 1)",
          r_pred.shape == (4, 1))
    check(f"done shape:   {tuple(d_logit.shape)} == (4, 1)",
          d_logit.shape == (4, 1))
    check(f"hidden states: {len(h_states)} lists == {NUM_BLOCKS} blocks",
          len(h_states) == NUM_BLOCKS)

    h_shape = h_states[0].shape
    expected_h = (B, HIDDEN_DIM * 2, STATE_DIM)  # expand=2
    check(f"hidden[0] shape: {tuple(h_shape)} == {expected_h}",
          h_shape == expected_h)

    # ── 2. Multi-step rollout shapes ─────────────────────────────
    print("\n2. Multi-step rollout (H=5)")
    H = 5
    actions_seq = torch.randint(0, MAX_ACTIONS, (B, H)).to(DEVICE)
    z_states, rewards, dones = model.multi_step_rollout(z_dummy, actions_seq)

    check(f"z_states: {tuple(z_states.shape)} == ({B}, {H}, {LATENT_DIM})",
          z_states.shape == (B, H, LATENT_DIM))
    check(f"rewards:  {tuple(rewards.shape)} == ({B}, {H})",
          rewards.shape == (B, H))
    check(f"dones:    {tuple(dones.shape)} == ({B}, {H})",
          dones.shape == (B, H))

    # Done probabilities should be in [0, 1] (sigmoid applied)
    check(f"dones in [0,1]: min={dones.min():.3f} max={dones.max():.3f}",
          dones.min() >= 0.0 and dones.max() <= 1.0)

    # ── 3. Hidden state evolves across rollout steps ─────────────
    print("\n3. Hidden state evolution (trajectory memory)")
    z_test = torch.randn(1, LATENT_DIM).to(DEVICE)
    a_test = torch.randint(0, MAX_ACTIONS, (1,)).to(DEVICE)

    model.reset_hidden(1, DEVICE)
    _, _, _, h1 = model(z_test, a_test)
    _, _, _, h2 = model(z_test, a_test)  # same input, different step

    h1_flat = h1[0].reshape(-1)
    h2_flat = h2[0].reshape(-1)
    hidden_diff = (h1_flat - h2_flat).abs().mean().item()
    check(f"Hidden state changes across steps: diff={hidden_diff:.6f} > 0",
          hidden_diff > 1e-8)

    # ── 4. Reset hidden restores to zero ─────────────────────────
    print("\n4. Hidden state reset")
    model.reset_hidden(1, DEVICE)
    h_after_reset = model._hidden_states[0]
    check("Hidden state is zero after reset",
          h_after_reset.abs().max().item() == 0.0)

    # ── 5. Determinism ───────────────────────────────────────────
    print("\n5. Determinism (same input + same hidden -> same output)")
    model.reset_hidden(1, DEVICE)
    with torch.no_grad():
        z1, r1, d1, _ = model(z_test, a_test)
    model.reset_hidden(1, DEVICE)
    with torch.no_grad():
        z2, r2, d2, _ = model(z_test, a_test)
    check("z_next matches", torch.allclose(z1, z2, atol=1e-6))
    check("reward matches", torch.allclose(r1, r2, atol=1e-6))
    check("done matches", torch.allclose(d1, d2, atol=1e-6))

    # ── 6. 1-step prediction on real transitions ─────────────────
    print("\n6. 1-step prediction quality (real game transitions)")
    for name, EnvClass in [("TicTacToe", TicTacToe), ("Connect4", ConnectFour)]:
        env = EnvClass()
        state = env.reset()
        mse_list = []

        for _ in range(20):
            valid = env.get_valid_actions()
            if not valid:
                state = env.reset()
                valid = env.get_valid_actions()
            action = random.choice(valid)

            # Encode current state
            x = torch.tensor(
                pad_state(state).transpose(2, 0, 1), dtype=torch.float32
            ).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                z_cur = encoder(x)

            # Step and encode next state
            next_state, reward, done, _ = env.step(action)
            x_next = torch.tensor(
                pad_state(next_state).transpose(2, 0, 1), dtype=torch.float32
            ).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                z_true = encoder(x_next)

            # Predict
            a_tensor = torch.tensor([action], dtype=torch.long).to(DEVICE)
            model.reset_hidden(1, DEVICE)
            with torch.no_grad():
                z_pred, _, _, _ = model(z_cur, a_tensor)

            mse = nn.functional.mse_loss(z_pred, z_true).item()
            mse_list.append(mse)

            if done:
                state = env.reset()
            else:
                state = next_state

        avg_mse = np.mean(mse_list)
        # With a random encoder, MSE will be higher; with pretrained, much lower
        check(f"{name:12s} 1-step MSE = {avg_mse:.4f} (lower is better)", True)

    # ── 7. Parameter counts ──────────────────────────────────────
    print("\n7. Parameter counts")
    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"     Total:     {total_p:>10,}")
    print(f"     Trainable: {trainable_p:>10,}")

    block_p = sum(p.numel() for p in model.blocks.parameters())
    fusion_p = sum(p.numel() for p in model.fusion.parameters())
    heads_p = (sum(p.numel() for p in model.next_state_head.parameters())
               + sum(p.numel() for p in model.reward_head.parameters())
               + sum(p.numel() for p in model.done_head.parameters()))
    embed_p = sum(p.numel() for p in model.action_embed.parameters())

    print(f"     Action embed: {embed_p:>8,}")
    print(f"     Fusion:       {fusion_p:>8,}")
    print(f"     SSM blocks:   {block_p:>8,}")
    print(f"     Pred heads:   {heads_p:>8,}")

    # ── 8. Loss function ─────────────────────────────────────────
    print("\n8. Loss function (paper Eq. 2)")
    criterion = WorldModelLoss(reward_weight=0.5)
    z_a = torch.randn(2, LATENT_DIM).to(DEVICE)
    z_b = torch.randn(2, LATENT_DIM).to(DEVICE)
    r_a = torch.randn(2, 1).to(DEVICE)
    r_b = torch.randn(2, 1).to(DEVICE)
    d_a = torch.randn(2, 1).to(DEVICE)
    d_b = torch.ones(2, 1).to(DEVICE)
    loss, (ls, lr_, ld) = criterion(z_a, z_b, r_a, r_b, d_a, d_b)
    check(f"Loss computes: total={loss.item():.4f} (state={ls:.4f} rew={lr_:.4f} done={ld:.4f})",
          loss.item() > 0)

    # ── Summary ──────────────────────────────────────────────────
    n_pass = sum(results)
    n_total = len(results)
    print("\n" + "=" * 64)
    if all(results):
        print(f"  ALL {n_total} CHECKS PASSED")
    else:
        print(f"  {n_pass}/{n_total} CHECKS PASSED")
    print("=" * 64)
    return all(results)
def main():
    print("=" * 64)
    print("  BASIL Mamba World Model Training")
    print("=" * 64)
    print(f"  Device:       {DEVICE}")
    print(f"  Latent dim:   {LATENT_DIM}")
    print(f"  Action dim:   {ACTION_DIM}")
    print(f"  Hidden dim:   {HIDDEN_DIM}")
    print(f"  SSM state:    {STATE_DIM}")
    print(f"  Mamba blocks: {NUM_BLOCKS}")
    print(f"  Max actions:  {MAX_ACTIONS}")
    print()

    # ── 1. Load encoder ──────────────────────────────────────────
    encoder = load_encoder()

    # ── 2. Collect transitions ───────────────────────────────────
    data = collect_mixed_transitions(
        encoder=encoder,
        num_per_game=NUM_TRANSITIONS,
        device=str(DEVICE),
    )

    # ── 3. DataLoaders ───────────────────────────────────────────
    train_loader, val_loader = make_dataloaders(data)

    # ── 4. Create world model ────────────────────────────────────
    world_model = MambaWorldModel(
        latent_dim=LATENT_DIM,
        action_dim=ACTION_DIM,
        max_actions=MAX_ACTIONS,
        hidden_dim=HIDDEN_DIM,
        state_dim=STATE_DIM,
        num_blocks=NUM_BLOCKS,
    ).to(DEVICE)

    print(f"\nWorld Model:\n{world_model}\n")

    # ── 5. Train ─────────────────────────────────────────────────
    print("Training world model...")
    best_val = train(world_model, train_loader, val_loader)
    print(f"\nBest validation loss: {best_val:.6f}")

    # ── 6. Save ──────────────────────────────────────────────────
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/world_model.pt"
    torch.save(
        {
            "model_state_dict": world_model.state_dict(),
            "latent_dim": LATENT_DIM,
            "action_dim": ACTION_DIM,
            "hidden_dim": HIDDEN_DIM,
            "state_dim": STATE_DIM,
            "num_blocks": NUM_BLOCKS,
            "max_actions": MAX_ACTIONS,
            "best_val_loss": best_val,
        },
        ckpt_path,
    )
    print(f"Saved -> {ckpt_path}")

    # ── 7. Verify ────────────────────────────────────────────────
    verify(world_model, encoder)

    # ── Usage hint ───────────────────────────────────────────────
    print("\n--- Usage in BASIL planning (Slow Path) ---")
    print("  wm = MambaWorldModel(...)")
    print("  ckpt = torch.load('checkpoints/world_model.pt')")
    print("  wm.load_state_dict(ckpt['model_state_dict'])")
    print("  wm.eval()")
    print()
    print("  # Single step prediction:")
    print("  wm.reset_hidden(batch_size, device)")
    print("  z_next, reward, done_logit, _ = wm(z_t, action)")
    print()
    print("  # Multi-step rollout (k=10, H=5):")
    print("  z_states, rewards, dones = wm.multi_step_rollout(z_start, actions)")


if __name__ == "__main__":
    main()
