from typing import List, Tuple
import random
from copy import deepcopy

# rows x cols, top to bottom, left to right
Board = List[List[bool]]


class Tetromino:
    def __init__(self, shape: Board):
        self.shape = shape
        self.rows = len(shape)
        self.cols = len(shape[0])


class ITetromino:
    @staticmethod
    def get_tetrominos() -> List[Tetromino]:
        return [
            Tetromino([[True, True, True, True]]),
            Tetromino([[True],
                       [True],
                       [True],
                       [True]])
        ]


class OTetromino:
    @staticmethod
    def get_tetrominos() -> List[Tetromino]:
        return [
            Tetromino([[True, True],
                       [True, True]])
        ]


class LTetromino:
    @staticmethod
    def get_tetrominos() -> List[Tetromino]:
        return [
            Tetromino([[True, False],
                       [True, False],
                       [True, True]]),
            Tetromino([[True, True, True],
                       [True, False, False]]),
            Tetromino([[True, True],
                       [False, True],
                       [False, True]]),
            Tetromino([[False, False, True],
                       [True, True, True]])
        ]


class JTetromino:
    @staticmethod
    def get_tetrominos() -> List[Tetromino]:
        return [
            Tetromino([[False, True],
                       [False, True],
                       [True, True]]),
            Tetromino([[True, False, False],
                       [True, True, True]]),
            Tetromino([[True, True],
                       [True, False],
                       [True, False]]),
            Tetromino([[True, True, True],
                       [False, False, True]])
        ]


class STetromino:
    @staticmethod
    def get_tetrominos() -> List[Tetromino]:
        return [
            Tetromino([[False, True, True],
                       [True, True, False]]),
            Tetromino([[True, False],
                       [True, True],
                       [False, True]]),
        ]


class ZTetromino:
    @staticmethod
    def get_tetrominos() -> List[Tetromino]:
        return [
            Tetromino([[True, True, False],
                       [False, True, True]]),
            Tetromino([[False, True],
                       [True, True],
                       [True, False]])
        ]


class TTetromino:
    @staticmethod
    def get_tetrominos() -> List[Tetromino]:
        return [
            Tetromino([[True, True, True],
                       [False, True, False]]),
            Tetromino([[False, True],
                       [True, True],
                       [False, True]]),
            Tetromino([[False, True, False],
                       [True, True, True]]),
            Tetromino([[True, False],
                       [True, True],
                       [True, False]])
        ]


class SavedGame:
    def __init__(self, first_tetromino: int):
        self.tetrominos: List[int] = [first_tetromino]
        self.actions: List[int] = []

    def add(self, tetromino: int, action: int):
        self.tetrominos.append(tetromino)
        self.actions.append(action)


class Tetris:
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.board: Board = [[False for _ in range(cols)] for _ in range(rows)]
        self.tetrominos = [ITetromino, OTetromino, LTetromino, JTetromino, STetromino, ZTetromino, TTetromino]
        self.next: int = self._calculate_next()
        self.actions, self.rewards = self._calculate_actions()
        self.archive = SavedGame(self.next)

    def get_next(self):
        return self.next

    def get_state(self) -> Board:
        return self.board

    def get_actions(self) -> List[Board]:
        return self.actions

    def take_action(self, action: int) -> float:
        self.board = self.actions[action]
        reward = self.rewards[action]

        self.next = self._calculate_next()
        self.actions, self.rewards = self._calculate_actions()
        self.archive.add(self.next, action)

        return reward

    def is_done(self) -> bool:
        return len(self.actions) == 0

    def _calculate_next(self) -> int:
        return random.randrange(len(self.tetrominos))

    def _calculate_actions(self) -> Tuple[List[Board], List[float]]:
        actions = []
        rewards = []
        for t in self.tetrominos[self.next].get_tetrominos():
            for drop in range(self.cols - t.cols + 1):
                next_row = 0
                while self._can_drop(t, drop, next_row):
                    next_row += 1
                # make sure the tetromino can fit
                if next_row >= t.rows:
                    action = deepcopy(self.board)
                    for t_row in range(t.rows):
                        for t_col in range(t.cols):
                            if t.shape[t_row][t_col]:
                                action[next_row - t.rows + t_row][drop + t_col] = 1
                    rows_cleared = clear_filled_rows(action)
                    rewards.append(score(rows_cleared))
                    actions.append(action)
        return actions, rewards

    def _can_drop(self, t: Tetromino, col: int, next_row: int) -> bool:
        if next_row == self.rows:
            return False
        for t_col in range(t.cols):
            lowest_row = t.rows - 1
            while lowest_row > 0 and not t.shape[lowest_row][t_col]:
                lowest_row -= 1
            row_check = next_row - t.rows + lowest_row + 1
            col_check = col + t_col
            if row_check >= 0 and self.board[row_check][col_check]:
                return False
        return True


def clear_filled_rows(board: Board) -> int:
    rows_cleared = 0
    cols = len(board[0])
    row = len(board) - 1
    while row >= 0:
        if sum(board[row]) == cols:
            rows_cleared += 1
            for other_row in range(row - 1, -1, -1):
                board[other_row + 1] = board[other_row]
            board[0] = [0] * cols
        else:
            row -= 1
    return rows_cleared


def score(rows_cleared: int) -> float:
    if rows_cleared == 1:
        return 0.1
    if rows_cleared == 2:
        return 0.3
    if rows_cleared == 3:
        return 0.5
    if rows_cleared == 4:
        return 0.8
    return 0
