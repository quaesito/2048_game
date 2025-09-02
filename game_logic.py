import random
import copy
import multiprocessing as mp

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

def evaluate_board_expert(board):
    """
    Expert evaluation function optimized for 95-99% win rate.
    Empties dominate, with monotonicity, smoothness, and corner strategy.
    """
    # Expert weight factors - empties dominate as recommended
    WEIGHT_EMPTIES = 10.0        # Dominates - most important for maneuverability
    WEIGHT_MONOTONICITY = 2.0    # Critical for maintaining order
    WEIGHT_SMOOTHNESS = 1.0      # Important for merge opportunities
    WEIGHT_MAX_TILE = 1.0        # Rewards high tiles
    WEIGHT_CORNER_STRATEGY = 3.0 # Keep max in corner
    WEIGHT_POTENTIAL_MERGES = 2.0 # Potential merge opportunities
    WEIGHT_CLUSTERING_PENALTY = 0.5 # Mild penalty for scattered tiles
    
    score = 0.0
    
    # 1. EMPTIES DOMINATE - most critical factor
    empty_cells = sum(1 for row in board for cell in row if cell is None)
    score += WEIGHT_EMPTIES * empty_cells * empty_cells  # Quadratic bonus for more empties
    
    # 2. Monotonicity - prefer ordered boards
    score += WEIGHT_MONOTONICITY * get_monotonicity_score(board)
    
    # 3. Smoothness - prefer similar adjacent values
    score += WEIGHT_SMOOTHNESS * get_smoothness_score(board)
    
    # 4. Max tile value
    score += WEIGHT_MAX_TILE * get_max_tile_score(board)
    
    # 5. Corner strategy - keep max tile in corner
    score += WEIGHT_CORNER_STRATEGY * get_corner_strategy_expert(board)
    
    # 6. Potential merges - reward merge opportunities
    score += WEIGHT_POTENTIAL_MERGES * get_potential_merges(board)
    
    # 7. Clustering penalty - mild penalty for scattered high tiles
    score -= WEIGHT_CLUSTERING_PENALTY * get_clustering_penalty(board)
    
    return score

def get_corner_strategy_expert(board):
    """
    Expert corner strategy with "keep max in corner" guardrails.
    """
    # Find the highest tile
    max_tile = 0
    max_positions = []
    for r in range(4):
        for c in range(4):
            if board[r][c] is not None and board[r][c] > max_tile:
                max_tile = board[r][c]
                max_positions = [(r, c)]
            elif board[r][c] is not None and board[r][c] == max_tile:
                max_positions.append((r, c))
    
    if max_tile == 0:
        return 0
    
    # Check if max tile is in a corner
    corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
    for pos in max_positions:
        if pos in corners:
            # Strong bonus for corner placement
            return max_tile * 5
    
    # Penalty for max tile not in corner
    return -max_tile * 2

def get_potential_merges(board):
    """
    Calculate potential merge opportunities.
    """
    merge_score = 0
    
    # Check for adjacent equal tiles
    for r in range(4):
        for c in range(4):
            if board[r][c] is not None:
                current_value = board[r][c]
                
                # Check all directions
                directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 4 and 0 <= nc < 4:
                        if board[nr][nc] == current_value:
                            # Found a potential merge
                            merge_score += current_value * 2
    
    return merge_score

def get_clustering_penalty(board):
    """
    Mild penalty for scattered high-value tiles.
    """
    # Calculate center of mass of high-value tiles
    total_value = 0
    weighted_row = 0
    weighted_col = 0
    
    for r in range(4):
        for c in range(4):
            if board[r][c] is not None:
                value = board[r][c]
                total_value += value
                weighted_row += r * value
                weighted_col += c * value
    
    if total_value == 0:
        return 0
    
    center_row = weighted_row / total_value
    center_col = weighted_col / total_value
    
    # Calculate variance (how spread out the tiles are)
    variance = 0
    for r in range(4):
        for c in range(4):
            if board[r][c] is not None:
                value = board[r][c]
                distance = abs(r - center_row) + abs(c - center_col)
                variance += value * distance * distance
    
    return variance / total_value

def get_monotonicity_score(board):
    """Calculates monotonicity score for rows and columns."""
    def get_monotonicity_direction(values):
        if not values:
            return 0
        
        increasing = 0
        decreasing = 0
        
        for i in range(len(values) - 1):
            if values[i] is not None and values[i + 1] is not None:
                if values[i] < values[i + 1]:
                    increasing += 1
                elif values[i] > values[i + 1]:
                    decreasing += 1
        
        return max(increasing, decreasing)
    
    # Check rows
    row_scores = []
    for row in board:
        row_values = [cell for cell in row if cell is not None]
        row_scores.append(get_monotonicity_direction(row_values))
    
    # Check columns
    col_scores = []
    for c in range(len(board[0])):
        col_values = [board[r][c] for r in range(len(board)) if board[r][c] is not None]
        col_scores.append(get_monotonicity_direction(col_values))
    
    return sum(row_scores) + sum(col_scores)

def get_smoothness_score(board):
    """Calculates smoothness based on adjacent tile differences."""
    smoothness = 0
    
    # Check horizontal adjacents
    for r in range(len(board)):
        for c in range(len(board[r]) - 1):
            if board[r][c] is not None and board[r][c + 1] is not None:
                smoothness -= abs(board[r][c] - board[r][c + 1])
    
    # Check vertical adjacents
    for r in range(len(board) - 1):
        for c in range(len(board[r])):
            if board[r][c] is not None and board[r + 1][c] is not None:
                smoothness -= abs(board[r][c] - board[r + 1][c])
    
    return smoothness

def get_max_tile_score(board):
    """Returns score based on the highest tile value."""
    max_tile = 0
    for row in board:
        for cell in row:
            if cell is not None and cell > max_tile:
                max_tile = cell
    return max_tile

def get_corner_strategy_score(board):
    """Rewards keeping the largest tile in a corner."""
    max_tile = 0
    max_positions = []
    
    # Find all positions with the maximum tile value
    for r in range(len(board)):
        for c in range(len(board[r])):
            if board[r][c] is not None and board[r][c] > max_tile:
                max_tile = board[r][c]
                max_positions = [(r, c)]
            elif board[r][c] is not None and board[r][c] == max_tile:
                max_positions.append((r, c))
    
    if not max_positions:
        return 0
    
    # Check if any max tile is in a corner
    corners = [(0, 0), (0, len(board[0]) - 1), (len(board) - 1, 0), (len(board) - 1, len(board[0]) - 1)]
    for pos in max_positions:
        if pos in corners:
            return max_tile * 2  # Bonus for corner placement
    
    return 0

def simulate_move_with_random_tile(board, direction):
    """
    Simulates a move and adds a random tile, returning the new board.
    """
    moves = {
        'up': move_up,
        'down': move_down,
        'left': move_left,
        'right': move_right
    }
    
    if direction not in moves:
        return board
    
    # Create a deep copy of the board
    board_copy = [row[:] for row in board]
    new_board, _, has_moved = moves[direction](board_copy)
    
    if has_moved:
        add_random_tile(new_board)
    
    return new_board

def expectimax_expert(board, depth, is_maximizing):
    """
    Expert expectimax with optimized evaluation and move ordering.
    """
    if depth == 0:
        return evaluate_board_expert(board)
    
    if is_maximizing:
        # Player's turn - try all possible moves with move ordering
        moves = ['up', 'down', 'left', 'right']
        move_scores = []
        
        # Quick evaluation for move ordering
        for move in moves:
            new_board = simulate_move_with_random_tile(board, move)
            if new_board != board:  # Valid move
                quick_score = evaluate_board_expert(new_board)
                move_scores.append((move, quick_score))
        
        # Sort by quick evaluation (best first)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Search top moves deeply
        max_eval = float('-inf')
        for move, _ in move_scores[:min(2, len(move_scores))]:  # Top 2 moves only
            new_board = simulate_move_with_random_tile(board, move)
            if new_board != board:  # Valid move
                eval_score = expectimax_expert(new_board, depth - 1, False)
                max_eval = max(max_eval, eval_score)
        
        return max_eval
    else:
        # Random tile placement - use expected value with proper probabilities
        empty_cells = []
        for r in range(len(board)):
            for c in range(len(board[r])):
                if board[r][c] is None:
                    empty_cells.append((r, c))
        
        if not empty_cells:
            return evaluate_board_expert(board)
        
        # Calculate expected value: 90% chance of 2, 10% chance of 4
        expected_score = 0
        for r, c in empty_cells:
            # Try placing 2 (90% probability)
            board_copy_2 = [row[:] for row in board]
            board_copy_2[r][c] = 2
            score_2 = expectimax_expert(board_copy_2, depth - 1, True)
            
            # Try placing 4 (10% probability)
            board_copy_4 = [row[:] for row in board]
            board_copy_4[r][c] = 4
            score_4 = expectimax_expert(board_copy_4, depth - 1, True)
            
            # Weighted average
            expected_score += 0.9 * score_2 + 0.1 * score_4
        
        return expected_score / len(empty_cells)

def evaluate_board_fast(board):
    """
    Fast board evaluation using only the most critical heuristics.
    Optimized for speed while maintaining good decision quality.
    """
    score = 0.0
    
    # 1. Empty tiles - most important factor (simplified)
    empty_tiles = sum(1 for row in board for cell in row if cell is None)
    score += empty_tiles * 15.0  # Higher weight for speed
    
    # 2. Max tile in corner - critical strategy
    max_tile = 0
    max_in_corner = False
    corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
    
    for r in range(4):
        for c in range(4):
            if board[r][c] is not None and board[r][c] > max_tile:
                max_tile = board[r][c]
                max_in_corner = (r, c) in corners
    
    if max_in_corner:
        score += max_tile * 3.0  # Strong bonus for corner placement
    else:
        score -= max_tile * 1.0  # Penalty for not in corner
    
    # 3. Potential merges - quick check
    merge_score = 0
    for r in range(4):
        for c in range(4):
            if board[r][c] is not None:
                # Check right and down only (faster)
                if c < 3 and board[r][c] == board[r][c + 1]:
                    merge_score += board[r][c] * 2
                if r < 3 and board[r][c] == board[r + 1][c]:
                    merge_score += board[r][c] * 2
    
    score += merge_score * 1.5
    
    return score

def get_ai_suggestion_fast(board):
    """
    Fast AI using simple heuristics and minimal lookahead for quick responses.
    Target: < 0.1 seconds response time while maintaining good play.
    """
    moves = ['up', 'down', 'left', 'right']
    possible_moves = []
    
    # Find all valid moves
    for move in moves:
        board_copy = [row[:] for row in board]
        if move == 'up':
            _, _, has_moved = move_up(board_copy)
        elif move == 'down':
            _, _, has_moved = move_down(board_copy)
        elif move == 'left':
            _, _, has_moved = move_left(board_copy)
        elif move == 'right':
            _, _, has_moved = move_right(board_copy)
        
        if has_moved:
            possible_moves.append(move)

    if not possible_moves:
        return None  # No moves available

    # Fast evaluation - no lookahead, just immediate board evaluation
    best_move = possible_moves[0]
    best_score = float('-inf')
    
    for move in possible_moves:
        board_copy = [row[:] for row in board]
        if move == 'up':
            new_board, _, _ = move_up(board_copy)
        elif move == 'down':
            new_board, _, _ = move_down(board_copy)
        elif move == 'left':
            new_board, _, _ = move_left(board_copy)
        elif move == 'right':
            new_board, _, _ = move_right(board_copy)
        
        # Quick evaluation of the resulting board
        score = evaluate_board_fast(new_board)
        
        if score > best_score:
            best_score = score
            best_move = move
    
    return best_move

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

def get_ai_suggestion_expert(board):
    """
    Expert AI with expectimax depth 5-6, move ordering, and optimized heuristics for 95-99% win rate.
    Use this for maximum performance but slower response times.
    """
    moves = ['up', 'down', 'left', 'right']
    possible_moves = []
    
    # First, find all valid moves
    for move in moves:
        board_copy = [row[:] for row in board]
        if move == 'up':
            _, _, has_moved = move_up(board_copy)
        elif move == 'down':
            _, _, has_moved = move_down(board_copy)
        elif move == 'left':
            _, _, has_moved = move_left(board_copy)
        elif move == 'right':
            _, _, has_moved = move_right(board_copy)
        
        if has_moved:
            possible_moves.append(move)

    if not possible_moves:
        return None  # No moves available

    # Expert depth: 5-6 for maximum win rate
    empty_cells = sum(1 for row in board for cell in row if cell is None)
    max_depth = 6 if empty_cells >= 8 else 5 if empty_cells >= 4 else 4
    
    # Move ordering: evaluate moves quickly first, then deep search on promising ones
    move_scores = []
    for move in possible_moves:
        new_board = simulate_move_with_random_tile(board, move)
        if new_board != board:  # Valid move
            # Quick evaluation for move ordering
            quick_score = evaluate_board_expert(new_board)
            move_scores.append((move, quick_score))
    
    # Sort moves by quick evaluation (best first)
    move_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Deep search on top moves only
    best_move = move_scores[0][0]
    best_score = float('-inf')
    
    # Search top 2-3 moves deeply
    top_moves = move_scores[:min(3, len(move_scores))]
    
    for move, _ in top_moves:
        new_board = simulate_move_with_random_tile(board, move)
        if new_board != board:  # Valid move
            # Deep expectimax search
            score = expectimax_expert(new_board, max_depth, False)
            
            if score > best_score:
                best_score = score
                best_move = move
    
    return best_move

