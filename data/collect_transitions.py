import os
import random
import numpy as np
import torch

from envs.tictactoe import TicTacToe
from envs.connect4 import ConnectFour
from envs.game_registry import pad_state, PAD_H, PAD_W


def collect_transitions(
    env_class,
    encoder,
    num_transitions: int = 10000,
    device: str = "cpu",
):
    encoder.eval()
    env = env_class()

    z_t_list = []
    action_list = []
    z_next_list = []
    reward_list = []
    done_list = []

    collected = 0
    while collected < num_transitions:
        state = env.reset()
        done = False

        while not done and collected < num_transitions:
            # Encode current state
            padded = pad_state(state)
            x = torch.tensor(
                padded.transpose(2, 0, 1), dtype=torch.float32
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                z = encoder(x).squeeze(0)  # (64,)

            # Random action
            valid = env.get_valid_actions()
            action = random.choice(valid)

            # Step environment
            next_state, reward, done, info = env.step(action)

            # Encode next state
            padded_next = pad_state(next_state)
            x_next = torch.tensor(
                padded_next.transpose(2, 0, 1), dtype=torch.float32
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                z_next = encoder(x_next).squeeze(0)  # (64,)

            # Store transition
            z_t_list.append(z.cpu())
            action_list.append(action)
            z_next_list.append(z_next.cpu())
            reward_list.append(reward)
            done_list.append(1.0 if done else 0.0)

            state = next_state
            collected += 1

    return {
        "z_t": torch.stack(z_t_list),                          # (N, 64)
        "actions": torch.tensor(action_list, dtype=torch.long),  # (N,)
        "z_next": torch.stack(z_next_list),                      # (N, 64)
        "rewards": torch.tensor(reward_list, dtype=torch.float32),  # (N,)
        "dones": torch.tensor(done_list, dtype=torch.float32),     # (N,)
    }


def collect_mixed_transitions(
    encoder,
    num_per_game: int = 5000,
    device: str = "cpu",
):
    print("Collecting transitions...")

    print(f"  TicTacToe ({num_per_game} transitions)...")
    ttt = collect_transitions(TicTacToe, encoder, num_per_game, device)
    print(f"    collected {len(ttt['z_t'])} transitions")

    print(f"  Connect4 ({num_per_game} transitions)...")
    c4 = collect_transitions(ConnectFour, encoder, num_per_game, device)
    print(f"    collected {len(c4['z_t'])} transitions")

    # Merge
    merged = {
        "z_t": torch.cat([ttt["z_t"], c4["z_t"]], dim=0),
        "actions": torch.cat([ttt["actions"], c4["actions"]], dim=0),
        "z_next": torch.cat([ttt["z_next"], c4["z_next"]], dim=0),
        "rewards": torch.cat([ttt["rewards"], c4["rewards"]], dim=0),
        "dones": torch.cat([ttt["dones"], c4["dones"]], dim=0),
    }

    # Shuffle
    n = len(merged["z_t"])
    perm = torch.randperm(n)
    for key in merged:
        merged[key] = merged[key][perm]

    print(f"  Total: {n} transitions (shuffled)")

    # Print statistics
    n_done = merged["dones"].sum().item()
    n_reward = (merged["rewards"] > 0).sum().item()
    print(f"  Done episodes: {n_done:.0f} ({100*n_done/n:.1f}%)")
    print(f"  Reward > 0:    {n_reward:.0f} ({100*n_reward/n:.1f}%)")

    return merged


def save_transitions(data: dict, path: str):
    """Save transition data to disk."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(data, path)
    print(f"Saved transitions -> {path}")


def load_transitions(path: str) -> dict:
    """Load transition data from disk."""
    data = torch.load(path, weights_only=True)
    print(f"Loaded {len(data['z_t'])} transitions from {path}")
    return data
