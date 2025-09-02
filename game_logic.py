import random
import copy

def new_game(rows=4, cols=4):
    """
    Starts a new game by creating a board of a given size.
    By default, it creates a 4x4 board.
    """
    board = [[None for _ in range(cols)] for _ in range(rows)]
    return board

def add_random_tile(board):
    """
    Adds a new tile to a random empty spot on the board.
    The new tile will be a 2 (with 90% probability) or a 4 (with 10% probability).
    """
    empty_cells = []
    for r in range(len(board)):
        for c in range(len(board[r])):
            if board[r][c] is None:
                empty_cells.append((r, c))

    if not empty_cells:
        return board

    r, c = random.choice(empty_cells)
    board[r][c] = 2 if random.random() < 0.9 else 4
    return board

def get_game_state(board):
    """
    Checks the current state of the game.
    Returns 'win' if the 2048 tile is on the board.
    Returns 'lose' if there are no more possible moves.
    Otherwise, returns 'play'.
    """
    has_empty_cell = False
    for r in range(len(board)):
        for c in range(len(board[r])):
            if board[r][c] == 2048:
                return 'win'
            if board[r][c] is None:
                has_empty_cell = True

    if not has_empty_cell:
        # Check for possible merges
        for r in range(len(board)):
            for c in range(len(board[r])):
                # Check merge right
                if c < len(board[r]) - 1 and board[r][c] == board[r][c + 1]:
                    return 'play'
                # Check merge down
                if r < len(board) - 1 and board[r][c] == board[r + 1][c]:
                    return 'play'
        return 'lose'

    return 'play'

def transpose(board):
    """
    Transposes the board (rows become columns and vice-versa).
    This is used to apply horizontal logic to vertical moves.
    """
    return [list(row) for row in zip(*board)]

def reverse(board):
    """
    Reverses each row of the board.
    Used for right and down moves.
    """
    return [row[::-1] for row in board]

def merge_left(board):
    """
    Merges the board to the left.
    Returns the new board and the score obtained from the merges.
    """
    new_board = []
    score = 0
    for row in board:
        # 1. Slide non-empty cells to the left
        slid_row = [cell for cell in row if cell is not None]
        # 2. Merge adjacent equal cells
        merged_row = []
        i = 0
        while i < len(slid_row):
            if i + 1 < len(slid_row) and slid_row[i] == slid_row[i + 1]:
                merged_value = slid_row[i] * 2
                merged_row.append(merged_value)
                score += merged_value
                i += 2
            else:
                merged_row.append(slid_row[i])
                i += 1
        # 3. Fill the rest of the row with None
        while len(merged_row) < len(row):
            merged_row.append(None)
        new_board.append(merged_row)
    
    # Check if the board has changed
    has_moved = board != new_board
    
    return new_board, score, has_moved

def move_left(board):
    """Performs a left move."""
    return merge_left(board)

def move_right(board):
    """Performs a right move."""
    reversed_board = reverse(board)
    merged_board, score, has_moved = merge_left(reversed_board)
    final_board = reverse(merged_board)
    return final_board, score, has_moved

def move_up(board):
    """Performs an up move."""
    transposed_board = transpose(board)
    merged_board, score, has_moved = merge_left(transposed_board)
    final_board = transpose(merged_board)
    return final_board, score, has_moved

def move_down(board):
    """Performs a down move."""
    transposed_board = transpose(board)
    reversed_board = reverse(transposed_board)
    merged_board, score, has_moved = merge_left(reversed_board)
    final_board = reverse(merged_board)
    final_board = transpose(final_board)
    return final_board, score, has_moved





# New AI Algorithm Integration (from new_ai.py)
INF = 2**64
PERFECT_SNAKE = [[2,   2**2, 2**3, 2**4],
                [2**8, 2**7, 2**6, 2**5],
                [2**9, 2**10,2**11,2**12],
                [2**16,2**15,2**14,2**13]]

def snake_heuristic(board):
    """Snake heuristic from new_ai.py adapted for our board structure."""
    h = 0
    for i in range(4):
        for j in range(4):
            if board[i][j] is not None:
                h += board[i][j] * PERFECT_SNAKE[i][j]
    return h

def get_open_tiles(board):
    """Get list of open tile positions."""
    open_tiles = []
    for r in range(4):
        for c in range(4):
            if board[r][c] is None:
                open_tiles.append((r, c))
    return open_tiles

def check_loss(board):
    """Check if the game is lost (no valid moves)."""
    # Check if there are empty cells
    if any(cell is None for row in board for cell in row):
        return False
    
    # Check for possible merges
    for r in range(4):
        for c in range(4):
            # Check right neighbor
            if c < 3 and board[r][c] == board[r][c + 1]:
                return False
            # Check bottom neighbor
            if r < 3 and board[r][c] == board[r + 1][c]:
                return False
    
    return True

def simulate_move_with_tile_placement(board, direction):
    """Simulate a move and return the new board state."""
    board_copy = [row[:] for row in board]
    
    if direction == 'up':
        new_board, _, has_moved = move_up(board_copy)
    elif direction == 'down':
        new_board, _, has_moved = move_down(board_copy)
    elif direction == 'left':
        new_board, _, has_moved = move_left(board_copy)
    elif direction == 'right':
        new_board, _, has_moved = move_right(board_copy)
    else:
        return board, False
    
    return new_board, has_moved

def expectiminimax_new(board, depth, direction=None):
    """Expectiminimax algorithm from new_ai.py adapted for our board structure."""
    if check_loss(board):
        return -INF, direction
    elif depth < 0:
        return snake_heuristic(board), direction
    
    a = 0
    if depth != int(depth):
        # Player's turn, pick max
        a = -INF
        directions = ['up', 'down', 'left', 'right']
        for dir in directions:
            new_board, has_moved = simulate_move_with_tile_placement(board, dir)
            if has_moved:
                res = expectiminimax_new(new_board, depth - 0.5, dir)[0]
                if res > a:
                    a = res
    elif depth == int(depth):
        # Nature's turn, calc average
        a = 0
        open_tiles = get_open_tiles(board)
        for tile_pos in open_tiles:
            r, c = tile_pos
            # Try placing 2 (90% probability)
            board_copy = [row[:] for row in board]
            board_copy[r][c] = 2
            a += 1.0 / len(open_tiles) * expectiminimax_new(board_copy, depth - 0.5, direction)[0]
    
    return (a, direction)

def get_next_best_move_expectiminimax(board, depth=2):
    """Get the best move using expectiminimax algorithm from new_ai.py."""
    best_score = -INF
    best_next_move = 'up'
    
    directions = ['up', 'down', 'left', 'right']
    
    for direction in directions:
        new_board, has_moved = simulate_move_with_tile_placement(board, direction)
        if not has_moved:
            continue
        
        score, _ = expectiminimax_new(new_board, depth, direction)
        if score >= best_score:
            best_score = score
            best_next_move = direction
    
    return best_next_move

def get_ai_suggestion_new(board):
    """
    New AI suggestion using expectiminimax algorithm from new_ai.py.
    This is the main AI function that will be used by default.
    """
    # Check if there are any valid moves
    valid_moves = []
    for direction in ['up', 'down', 'left', 'right']:
        new_board, has_moved = simulate_move_with_tile_placement(board, direction)
        if has_moved:
            valid_moves.append(direction)
    
    if not valid_moves:
        return None  # No moves available
    
    # Use expectiminimax with depth 2 for good performance
    return get_next_best_move_expectiminimax(board, depth=2)

def get_ai_suggestion(board):
    """
    Main AI suggestion function - now uses the new expectiminimax algorithm.
    This provides better strategic play with the snake heuristic.
    """
    return get_ai_suggestion_new(board)