"""
AI Backend for 2048 Game

This module contains all AI-related functionality for the 2048 game,
including the AIEvaluator class and AI decision-making algorithms.
"""

from typing import List, Tuple, Optional
from game_logic import Merger


class AIEvaluator:
    """Handles AI evaluation and decision making."""
    
    # Large number representing infinity for loss states
    INF = 2**64
    # Snake pattern weights - higher values in corner and snake pattern for optimal tile placement
    PERFECT_SNAKE = [[2,   2**2, 2**3, 2**4],
                    [2**8, 2**7, 2**6, 2**5],
                    [2**9, 2**10,2**11,2**12],
                    [2**16,2**15,2**14,2**13]]
    
    @staticmethod
    def snake_heuristic(board: List[List[Optional[int]]]) -> int:
        """Calculate board score using snake pattern heuristic for AI evaluation."""
        score = 0
        # Multiply each tile value by its position weight in the snake pattern
        for i in range(4):
            for j in range(4):
                if board[i][j] is not None:
                    score += board[i][j] * AIEvaluator.PERFECT_SNAKE[i][j]
        return score
    
    @staticmethod
    def check_loss(board: List[List[Optional[int]]]) -> bool:
        """Check if the game is lost (no empty cells and no possible merges)."""
        # Game continues if there are empty cells
        if any(cell is None for row in board for cell in row):
            return False
        
        # Check for possible merges in all directions
        for r in range(4):
            for c in range(4):
                # Check if can merge with right neighbor
                if c < 3 and board[r][c] == board[r][c + 1]:
                    return False
                # Check if can merge with bottom neighbor
                if r < 3 and board[r][c] == board[r + 1][c]:
                    return False
        
        return True
    
    @staticmethod
    def simulate_move(board: List[List[Optional[int]]], direction: str) -> Tuple[List[List[Optional[int]]], bool]:
        """Simulate a move in given direction and return new board state and move validity."""
        # Create a copy to avoid modifying the original board
        board_copy = [row[:] for row in board]
        
        # Apply the appropriate merge operation based on direction
        if direction == 'up':
            new_board, _, has_moved = Merger.merge_up(board_copy)
        elif direction == 'down':
            new_board, _, has_moved = Merger.merge_down(board_copy)
        elif direction == 'left':
            new_board, _, has_moved = Merger.merge_left(board_copy)
        elif direction == 'right':
            new_board, _, has_moved = Merger.merge_right(board_copy)
        else:
            return board, False
        
        return new_board, has_moved
    
    @staticmethod
    def expectiminimax(board: List[List[Optional[int]]], depth: float, direction: Optional[str] = None) -> Tuple[float, Optional[str]]:
        """Expectiminimax algorithm for AI decision making with stochastic tile placement."""
        # Return negative infinity if game is lost
        if AIEvaluator.check_loss(board):
            return -AIEvaluator.INF, direction
        # Return heuristic score when depth limit reached
        elif depth < 0:
            return AIEvaluator.snake_heuristic(board), direction
        
        score = 0
        # Player's turn (maximizing) - choose best move
        if depth != int(depth):
            score = -AIEvaluator.INF
            directions = ['up', 'down', 'left', 'right']
            for dir in directions:
                new_board, has_moved = AIEvaluator.simulate_move(board, dir)
                if has_moved:
                    res = AIEvaluator.expectiminimax(new_board, depth - 0.5, dir)[0]
                    if res > score:
                        score = res
        # Nature's turn (chance) - calculate expected value
        elif depth == int(depth):
            score = 0
            empty_cells = []
            # Find all empty cells for random tile placement
            for r in range(4):
                for c in range(4):
                    if board[r][c] is None:
                        empty_cells.append((r, c))
            
            # Calculate expected score for each possible tile placement
            for tile_pos in empty_cells:
                r, c = tile_pos
                # Simulate placing a 2 tile (90% probability in actual game)
                board_copy = [row[:] for row in board]
                board_copy[r][c] = 2
                score += 1.0 / len(empty_cells) * AIEvaluator.expectiminimax(board_copy, depth - 0.5, direction)[0]
        
        return (score, direction)
    
    @staticmethod
    def get_best_move(board: List[List[Optional[int]]], depth: int = 2) -> str:
        """Find the best move using expectiminimax algorithm with given search depth."""
        best_score = -AIEvaluator.INF
        best_move = 'up'  # Default fallback move
        
        directions = ['up', 'down', 'left', 'right']
        
        # Evaluate each possible move
        for direction in directions:
            new_board, has_moved = AIEvaluator.simulate_move(board, direction)
            # Skip invalid moves that don't change the board
            if not has_moved:
                continue
            
            # Get score for this move using expectiminimax
            score, _ = AIEvaluator.expectiminimax(new_board, depth, direction)
            # Update best move if this one is better
            if score >= best_score:
                best_score = score
                best_move = direction
        
        return best_move
    
    @staticmethod
    def get_ai_suggestion(board: List[List[Optional[int]]]) -> Optional[str]:
        """Get AI move suggestion using expectiminimax algorithm with snake heuristic."""
        # First check if there are any valid moves available
        valid_moves = []
        for direction in ['up', 'down', 'left', 'right']:
            new_board, has_moved = AIEvaluator.simulate_move(board, direction)
            if has_moved:
                valid_moves.append(direction)
        
        # Return None if no valid moves (game over)
        if not valid_moves:
            return None
        
        # Use expectiminimax with depth 2 for good balance of performance and intelligence
        return AIEvaluator.get_best_move(board, depth=2)


# Public API function for backward compatibility
def get_ai_suggestion(board: List[List[Optional[int]]]) -> Optional[str]:
    """Main AI function that returns the best move suggestion for the current board."""
    # Delegate to the AIEvaluator class method
    return AIEvaluator.get_ai_suggestion(board)
