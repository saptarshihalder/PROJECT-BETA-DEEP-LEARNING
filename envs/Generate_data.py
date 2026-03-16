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
    h, w, c = state.shape
    if h > pad_h or w > pad_w:
        raise ValueError(f"State size ({h}x{w}) exceeds pad size ({pad_h}x{pad_w})")

    padded = np.zeros((pad_h, pad_w, c), dtype=np.float32)
    padded[:h, :w, :] = state
    return padded


def collect_states_from_game(env_class, num_states=50000):
    env = env_class()
    states = []

    while len(states) < num_states:
        state = env.reset()
        states.append(state.copy())

        done = False
        while not done and len(states) < num_states:
            valid_actions = env.get_valid_actions()
            action = random.choice(valid_actions)
            state, reward, done, info = env.step(action)
            states.append(state.copy())

    print(f"  Collected {len(states)} states from {env_class.__name__}")
    return states


def generate_dataset(num_states_per_game=50000, save_path="data/"):
    os.makedirs(save_path, exist_ok=True)

    print("Collecting TicTacToe states...")
    ttt_states = collect_states_from_game(TicTacToe, num_states_per_game)

    print("Collecting Connect4 states...")
    c4_states = collect_states_from_game(ConnectFour, num_states_per_game)

    ttt_array = np.array(ttt_states)
    c4_array = np.array(c4_states)

    np.save(os.path.join(save_path, "ttt_states.npy"), ttt_array)
    np.save(os.path.join(save_path, "c4_states.npy"), c4_array)
    print(f"Saved raw states — TTT: {ttt_array.shape}, C4: {c4_array.shape}")

    print(f"Padding all states to {PAD_H}x{PAD_W}...")
    all_padded = []

    for state in ttt_states:
        all_padded.append(pad_state(state))

    for state in c4_states:
        all_padded.append(pad_state(state))

    random.shuffle(all_padded)
    mixed_array = np.array(all_padded)

    np.save(os.path.join(save_path, "mixed_padded.npy"), mixed_array)
    print(f"Saved mixed padded dataset: {mixed_array.shape}")

    print(f"\nTicTacToe states:  {len(ttt_states)}")
    print(f"Connect4 states:   {len(c4_states)}")
    print(f"Total mixed:       {len(all_padded)} (padded to {PAD_H}x{PAD_W}x3)")
    print(f"Saved to:          {os.path.abspath(save_path)}")


if __name__ == "__main__":
    generate_dataset(num_states_per_game=50000, save_path="data/")
