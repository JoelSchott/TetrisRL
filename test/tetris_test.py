from src import tetris


def test_empty_falling():
    game = tetris.Tetris(3, 5)
    # hack the game to make next a s tetromino and recalculate actions
    game.next = 4
    game.actions, game.rewards = game._calculate_actions()
    assert game.get_actions() == [
        [[0, 0, 0, 0, 0],
         [0, 1, 1, 0, 0],
         [1, 1, 0, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 1, 0],
         [0, 1, 1, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 1, 1],
         [0, 0, 1, 1, 0]],

        [[1, 0, 0, 0, 0],
         [1, 1, 0, 0, 0],
         [0, 1, 0, 0, 0]],

        [[0, 1, 0, 0, 0],
         [0, 1, 1, 0, 0],
         [0, 0, 1, 0, 0]],

        [[0, 0, 1, 0, 0],
         [0, 0, 1, 1, 0],
         [0, 0, 0, 1, 0]],

        [[0, 0, 0, 1, 0],
         [0, 0, 0, 1, 1],
         [0, 0, 0, 0, 1]]
    ]
    assert game.rewards == [0, 0, 0, 0, 0, 0, 0]


def test_blocked_falling():
    game = tetris.Tetris(5, 5)
    # hack the game to make a custom board
    game.board = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1]
    ]
    # hack the game to make next a j tetromino and recalculate actions
    game.next = 3
    game.actions, game.rewards = game._calculate_actions()
    assert game.get_actions() == [
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 1, 0, 1, 0],
         [0, 1, 1, 1, 0]],

        [[0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 1, 1, 1, 0],
         [0, 0, 1, 1, 0],
         [0, 0, 1, 1, 1]],

        [[0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [1, 1, 1, 1, 0],
         [0, 0, 1, 1, 0],
         [0, 0, 1, 1, 1]],

        [[0, 1, 0, 0, 0],
         [0, 1, 1, 1, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 1, 1, 0],
         [0, 0, 1, 1, 1]],

        [[0, 0, 1, 0, 0],
         [0, 0, 1, 1, 1],
         [0, 0, 0, 1, 0],
         [0, 0, 1, 1, 0],
         [0, 0, 1, 1, 1]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [1, 1, 0, 1, 0],
         [1, 0, 1, 1, 0],
         [1, 0, 1, 1, 1]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 1, 1, 1, 0],
         [0, 1, 1, 1, 0],
         [0, 1, 1, 1, 1]],

        [[0, 0, 1, 1, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 1, 0],
         [0, 0, 1, 1, 0],
         [0, 0, 1, 1, 1]],

        [[0, 0, 0, 0, 0],
         [1, 1, 1, 0, 0],
         [0, 0, 1, 1, 0],
         [0, 0, 1, 1, 0],
         [0, 0, 1, 1, 1]],

        [[0, 1, 1, 1, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 1, 1, 0],
         [0, 0, 1, 1, 1]],

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 1, 1],
         [0, 0, 0, 1, 1],
         [0, 0, 1, 1, 0],
         [0, 0, 1, 1, 1]]
    ]
    assert game.rewards == [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def test_clear_row():
    game = tetris.Tetris(5, 5)
    # hack the game to make a custom board
    game.board = [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 1, 1, 1, 1]
    ]
    # hack the game to make next a t tetromino and recalculate actions
    game.next = 6
    game.actions, game.rewards = game._calculate_actions()
    assert game.get_actions() == [
        [[1, 1, 1, 0, 0],
         [0, 1, 1, 0, 0],
         [0, 0, 1, 1, 1],
         [0, 0, 1, 1, 1],
         [0, 1, 1, 1, 1]],

        [[0, 0, 1, 1, 1],
         [0, 0, 1, 1, 0],
         [0, 0, 1, 1, 1],
         [0, 0, 1, 1, 1],
         [0, 1, 1, 1, 1]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 1, 1, 0, 0],
         [0, 1, 1, 1, 1],
         [0, 1, 1, 1, 1]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [1, 0, 1, 1, 1]]
    ]
    assert game.rewards == [0, 0, 0.1, 0.3]


def test_full_clear():
    game = tetris.Tetris(5, 5)
    # hack the game to make a custom board
    game.board = [
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0]
    ]
    # hack the game to make next a i tetromino and recalculate actions
    game.next = 0
    game.actions, game.rewards = game._calculate_actions()
    assert game.get_actions() == [
        [[1, 1, 1, 1, 0],
         [1, 1, 1, 1, 0],
         [1, 1, 1, 1, 0],
         [1, 1, 1, 1, 0],
         [1, 1, 1, 1, 0]],

        [[0, 1, 1, 1, 1],
         [1, 1, 1, 1, 0],
         [1, 1, 1, 1, 0],
         [1, 1, 1, 1, 0],
         [1, 1, 1, 1, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]
    ]
    assert game.rewards == [0, 0, 0.8]
