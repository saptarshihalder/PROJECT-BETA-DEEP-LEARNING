import numpy as np


class ConnectFour:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.reset()

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        state = np.zeros((self.rows, self.cols, 3), dtype=np.float32)
        state[:, :, 0] = (self.board == 1).astype(np.float32)
        state[:, :, 1] = (self.board == 2).astype(np.float32)
        state[:, :, 2] = (self.board == 0).astype(np.float32)
        return state

    def get_valid_actions(self):
        actions = []
        for c in range(self.cols):
            if self.board[0][c] == 0:
                actions.append(c)
        return actions

    def _get_drop_row(self, col):
        for r in range(self.rows - 1, -1, -1):
            if self.board[r][col] == 0:
                return r
        return -1

    def step(self, action):
        if self.done:
            raise ValueError("Game is already over")
        col = action
        row = self._get_drop_row(col)
        if row == -1:
            raise ValueError(f"Column {col} is full")
        self.board[row][col] = self.current_player
        if self._check_win(row, col, self.current_player):
            self.done = True
            self.winner = self.current_player
            reward = 1.0
        elif len(self.get_valid_actions()) == 0:
            self.done = True
            self.winner = None
            reward = 0.0
        else:
            reward = 0.0
        info = {"winner": self.winner, "current_player": self.current_player}
        self.current_player = 3 - self.current_player
        return self.get_state(), reward, self.done, info

    def _check_win(self, row, col, player):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            r, c = row + dr, col + dc
            while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r][c] == player:
                count += 1
                r += dr
                c += dc
            r, c = row - dr, col - dc
            while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r][c] == player:
                count += 1
                r -= dr
                c -= dc
            if count >= 4:
                return True
        return False
