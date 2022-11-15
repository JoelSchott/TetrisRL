from tetris import Tetris, SavedGame, Board
from deep_rl import ROWS, COLS


class CustomNext:
    def __init__(self, nexts):
        self.nexts = nexts
        self.index = 0

    def __call__(self):
        self.index += 1
        return self.nexts[self.index - 1]


def print_board(board: Board):
    edge_string = '+'
    for _ in range(COLS):
        edge_string += '-+'
    print(edge_string)
    for row in board:
        row_string = '|'
        for e in row:
            row_string += 'X' if e else '.'
            row_string += '|'
        print(row_string)
    print(edge_string)


def show(saved_game: SavedGame):
    game = Tetris(ROWS, COLS)
    game._calculate_next = CustomNext(saved_game.tetrominos)
    game.next = game._calculate_next()
    game.actions, game.rewards = game._calculate_actions()

    print_board(game.board)
    for action in saved_game.actions:
        game.take_action(action)
        print_board(game.board)
