import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.encoder_decoder import Encoder
from models.mamba_world_model import MambaWorldModel, WorldModelLoss
from envs.tictactoe import TicTacToe
from envs.connect4 import ConnectFour
from envs.game_registry import pad_state, PAD_H, PAD_W, MAX_ACTIONS

LATENT_DIM = 64
ACTION_DIM = 16
HIDDEN_DIM = 128
STATE_DIM = 16
NUM_BLOCKS = 2
BATCH_SIZE = 64
EPOCHS = 100
LR = 3e-4
NUM_GAMES = 5000
ROLLOUT_H = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_encoder():
    encoder = Encoder(in_channels=3, latent_dim=LATENT_DIM, pad_h=PAD_H, pad_w=PAD_W).to(DEVICE)
    ckpt_path = "checkpoints/autoencoder.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        print(f"Loaded encoder from {ckpt_path}")
    else:
        print(f"WARNING: No encoder at {ckpt_path}")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def encode_state(encoder, state):
    padded = pad_state(state)
    x = torch.tensor(padded.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return encoder(x).squeeze(0).cpu()


def collect_trajectories(encoder, env_class, num_games, min_len=6):
    trajectories = []
    env = env_class()
    collected = 0
    while collected < num_games:
        state = env.reset()
        z_list = [encode_state(encoder, state)]
        a_list = []
        r_list = []
        d_list = []
        done = False
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            action = random.choice(valid)
            a_list.append(action)
            state, reward, done, _ = env.step(action)
            z_list.append(encode_state(encoder, state))
            r_list.append(reward)
            d_list.append(1.0 if done else 0.0)
        if len(a_list) >= min_len:
            trajectories.append({
                "z": torch.stack(z_list),
                "a": torch.tensor(a_list, dtype=torch.long),
                "r": torch.tensor(r_list, dtype=torch.float32),
                "d": torch.tensor(d_list, dtype=torch.float32),
            })
            collected += 1
    return trajectories


def extract_windows(trajectories, window_size):
    z_windows = []
    a_windows = []
    r_windows = []
    d_windows = []
    for traj in trajectories:
        T = len(traj["a"])
        for t in range(T - window_size + 1):
            z_windows.append(traj["z"][t:t+window_size+1])
            a_windows.append(traj["a"][t:t+window_size])
            r_windows.append(traj["r"][t:t+window_size])
            d_windows.append(traj["d"][t:t+window_size])
    return (
        torch.stack(z_windows),
        torch.stack(a_windows),
        torch.stack(r_windows),
        torch.stack(d_windows),
    )


def train(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    state_loss_fn = nn.MSELoss()
    reward_loss_fn = nn.MSELoss()
    done_loss_fn = nn.BCEWithLogitsLoss()
    best_val = float("inf")

    print(f"\n{'Epoch':>7} | {'Train':>10} | {'Val':>10} | {'Val 1-step':>10} | {'Val 5-step':>10}")
    print("-" * 58)

    for epoch in range(EPOCHS):
        model.train()
        train_total = 0.0
        train_n = 0
        for z_win, a_win, r_win, d_win in train_loader:
            z_win = z_win.to(DEVICE)
            a_win = a_win.to(DEVICE)
            r_win = r_win.to(DEVICE)
            d_win = d_win.to(DEVICE)
            B = z_win.size(0)

            model.reset_hidden(B, DEVICE)
            total_loss = torch.tensor(0.0, device=DEVICE)

            z_curr = z_win[:, 0]
            for t in range(ROLLOUT_H):
                z_next_pred, r_pred, d_logit, _ = model(z_curr, a_win[:, t])
                z_next_true = z_win[:, t+1]
                r_true = r_win[:, t].unsqueeze(-1)
                d_true = d_win[:, t].unsqueeze(-1)

                step_loss = (
                    state_loss_fn(z_next_pred, z_next_true)
                    + 0.5 * reward_loss_fn(r_pred, r_true)
                    + done_loss_fn(d_logit, d_true)
                )
                total_loss = total_loss + step_loss
                z_curr = z_next_pred

            avg_loss = total_loss / ROLLOUT_H
            optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += avg_loss.item() * B
            train_n += B

        train_avg = train_total / train_n

        model.eval()
        val_total = 0.0
        val_1step = 0.0
        val_5step = 0.0
        val_n = 0
        with torch.no_grad():
            for z_win, a_win, r_win, d_win in val_loader:
                z_win = z_win.to(DEVICE)
                a_win = a_win.to(DEVICE)
                r_win = r_win.to(DEVICE)
                d_win = d_win.to(DEVICE)
                B = z_win.size(0)

                model.reset_hidden(B, DEVICE)
                total_loss = 0.0
                z_curr = z_win[:, 0]
                for t in range(ROLLOUT_H):
                    z_next_pred, r_pred, d_logit, _ = model(z_curr, a_win[:, t])
                    z_next_true = z_win[:, t+1]
                    r_true = r_win[:, t].unsqueeze(-1)
                    d_true = d_win[:, t].unsqueeze(-1)
                    step_loss = (
                        state_loss_fn(z_next_pred, z_next_true).item()
                        + 0.5 * reward_loss_fn(r_pred, r_true).item()
                        + done_loss_fn(d_logit, d_true).item()
                    )
                    total_loss += step_loss
                    if t == 0:
                        val_1step += nn.functional.mse_loss(z_next_pred, z_next_true).item() * B
                    z_curr = z_next_pred

                val_5step += nn.functional.mse_loss(z_curr, z_win[:, ROLLOUT_H]).item() * B
                val_total += (total_loss / ROLLOUT_H) * B
                val_n += B

        val_avg = val_total / val_n
        v1 = val_1step / val_n
        v5 = val_5step / val_n

        if val_avg < best_val:
            best_val = val_avg
            torch.save({
                "model_state_dict": model.state_dict(),
                "latent_dim": LATENT_DIM, "action_dim": ACTION_DIM,
                "hidden_dim": HIDDEN_DIM, "state_dim": STATE_DIM,
                "num_blocks": NUM_BLOCKS, "max_actions": MAX_ACTIONS,
                "best_val_loss": best_val,
            }, "checkpoints/world_model.pt")

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == EPOCHS - 1:
            print(f"  {epoch+1:3d}/{EPOCHS}  | {train_avg:.6f} | {val_avg:.6f} | {v1:.6f} | {v5:.6f}")

    print(f"\nBest val loss: {best_val:.6f}")
    print("Saved -> checkpoints/world_model.pt")
    return best_val


def main():
    print("=" * 58)
    print("  BASIL — Multi-Step World Model Training")
    print("=" * 58)
    print(f"  Device: {DEVICE}")
    print(f"  Rollout horizon: {ROLLOUT_H}")
    print(f"  Games per env: {NUM_GAMES}")
    print()

    encoder = load_encoder()

    print("\nCollecting trajectories...")
    ttt_traj = collect_trajectories(encoder, TicTacToe, NUM_GAMES, min_len=ROLLOUT_H+1)
    c4_traj = collect_trajectories(encoder, ConnectFour, NUM_GAMES, min_len=ROLLOUT_H+1)
    all_traj = ttt_traj + c4_traj
    random.shuffle(all_traj)
    print(f"  TTT: {len(ttt_traj)}, C4: {len(c4_traj)}, Total: {len(all_traj)} trajectories")

    z_win, a_win, r_win, d_win = extract_windows(all_traj, ROLLOUT_H)
    print(f"  Windows: {z_win.shape[0]} (each {ROLLOUT_H} steps)")

    n_train = int(0.9 * len(z_win))
    perm = torch.randperm(len(z_win))
    z_win, a_win, r_win, d_win = z_win[perm], a_win[perm], r_win[perm], d_win[perm]

    train_ds = TensorDataset(z_win[:n_train], a_win[:n_train], r_win[:n_train], d_win[:n_train])
    val_ds = TensorDataset(z_win[n_train:], a_win[n_train:], r_win[n_train:], d_win[n_train:])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    print(f"  Train: {n_train}, Val: {len(z_win) - n_train}")

    model = MambaWorldModel(
        latent_dim=LATENT_DIM, action_dim=ACTION_DIM, max_actions=MAX_ACTIONS,
        hidden_dim=HIDDEN_DIM, state_dim=STATE_DIM, num_blocks=NUM_BLOCKS,
    ).to(DEVICE)

    train(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
