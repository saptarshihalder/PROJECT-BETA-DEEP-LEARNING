"""
Data generation for autoencoder training.
Collects random game states from TicTacToe and Connect4,
pads them to 7x7x3, and saves a mixed shuffled dataset.
"""

import numpy as np
import random
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.tictactoe import TicTacToe
from envs.connect4 import ConnectFour

PAD_H = 7
PAD_W = 7


def pad_state(state, pad_h=PAD_H, pad_w=PAD_W):
    """Pad a board state (H, W, 3) to (pad_h, pad_w, 3) with zeros."""
    h, w, c = state.shape
    if h > pad_h or w > pad_w:
        raise ValueError(f"State size ({h}x{w}) exceeds pad size ({pad_h}x{pad_w})")
    padded = np.zeros((pad_h, pad_w, c), dtype=np.float32)
    padded[:h, :w, :] = state
    return padded


def collect_states(env_class, num_states=50000):
    """Play random games and collect board states."""
    env = env_class()
    states = []
    while len(states) < num_states:
        state = env.reset()
        states.append(state.copy())
        done = False
        while not done and len(states) < num_states:
            action = random.choice(env.get_valid_actions())
            state, _, done, _ = env.step(action)
            states.append(state.copy())
    print(f"  Collected {len(states)} states from {env_class.__name__}")
    return states


def generate_dataset(num_states_per_game=50000, save_path="data/"):
    """Generate and save mixed padded dataset from all games."""
    os.makedirs(save_path, exist_ok=True)

    print("Collecting TicTacToe states...")
    ttt_states = collect_states(TicTacToe, num_states_per_game)

    print("Collecting Connect4 states...")
    c4_states = collect_states(ConnectFour, num_states_per_game)

    # Save raw states
    ttt_array = np.array(ttt_states)
    c4_array = np.array(c4_states)
    np.save(os.path.join(save_path, "ttt_states.npy"), ttt_array)
    np.save(os.path.join(save_path, "c4_states.npy"), c4_array)
    print(f"Saved raw — TTT: {ttt_array.shape}, C4: {c4_array.shape}")

    # Pad and mix
    print(f"Padding all states to {PAD_H}x{PAD_W}...")
    all_padded = [pad_state(s) for s in ttt_states] + [pad_state(s) for s in c4_states]
    random.shuffle(all_padded)
    mixed_array = np.array(all_padded)

    np.save(os.path.join(save_path, "mixed_padded.npy"), mixed_array)
    print(f"Saved mixed padded: {mixed_array.shape}")
    print(f"Total: {len(all_padded)} states (padded to {PAD_H}x{PAD_W}x3)")
    return mixed_array


if __name__ == "__main__":
    generate_dataset(num_states_per_game=50000, save_path="data/")
