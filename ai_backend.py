"""
AI Backend for 2048 Game

This module contains all AI-related functionality for the 2048 game,
including the AIEvaluator class and AI decision-making algorithms.
"""

import random
from typing import List, Tuple, Optional
from game_logic import Merger


class AIEvaluator:
    """Handles AI evaluation and decision making."""
    
    # Constants for AI evaluation
    INF = 2**64
    PERFECT_SNAKE = [[2,   2**2, 2**3, 2**4],
                    [2**8, 2**7, 2**6, 2**5],
                    [2**9, 2**10,2**11,2**12],
                    [2**16,2**15,2**14,2**13]]
    
    @staticmethod
    def snake_heuristic(board: List[List[Optional[int]]]) -> int:
        """Calculate board score using snake pattern heuristic for AI evaluation."""
        score = 0
        for i in range(4):
            for j in range(4):
                if board[i][j] is not None:
                    score += board[i][j] * AIEvaluator.PERFECT_SNAKE[i][j]
        return score
    
    @staticmethod
    def check_loss(board: List[List[Optional[int]]]) -> bool:
        """Check if the game is lost (no empty cells and no possible merges)."""
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
    
    @staticmethod
    def simulate_move(board: List[List[Optional[int]]], direction: str) -> Tuple[List[List[Optional[int]]], bool]:
        """Simulate a move in given direction and return new board state and move validity."""
        board_copy = [row[:] for row in board]
        
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
        if AIEvaluator.check_loss(board):
            return -AIEvaluator.INF, direction
        elif depth < 0:
            return AIEvaluator.snake_heuristic(board), direction
        
        score = 0
        if depth != int(depth):
            # Player's turn, pick max
            score = -AIEvaluator.INF
            directions = ['up', 'down', 'left', 'right']
            for dir in directions:
                new_board, has_moved = AIEvaluator.simulate_move(board, dir)
                if has_moved:
                    res = AIEvaluator.expectiminimax(new_board, depth - 0.5, dir)[0]
                    if res > score:
                        score = res
        elif depth == int(depth):
            # Nature's turn, calculate average
            score = 0
            empty_cells = []
            for r in range(4):
                for c in range(4):
                    if board[r][c] is None:
                        empty_cells.append((r, c))
            
            for tile_pos in empty_cells:
                r, c = tile_pos
                # Try placing 2 (90% probability)
                board_copy = [row[:] for row in board]
                board_copy[r][c] = 2
                score += 1.0 / len(empty_cells) * AIEvaluator.expectiminimax(board_copy, depth - 0.5, direction)[0]
        
        return (score, direction)
    
    @staticmethod
    def get_best_move(board: List[List[Optional[int]]], depth: int = 2) -> str:
        """Find the best move using expectiminimax algorithm with given search depth."""
        best_score = -AIEvaluator.INF
        best_move = 'up'
        
        directions = ['up', 'down', 'left', 'right']
        
        for direction in directions:
            new_board, has_moved = AIEvaluator.simulate_move(board, direction)
            if not has_moved:
                continue
            
            score, _ = AIEvaluator.expectiminimax(new_board, depth, direction)
            if score >= best_score:
                best_score = score
                best_move = direction
        
        return best_move
    
    @staticmethod
    def get_ai_suggestion(board: List[List[Optional[int]]]) -> Optional[str]:
        """Get AI move suggestion using expectiminimax algorithm with snake heuristic."""
        # Check if there are any valid moves
        valid_moves = []
        for direction in ['up', 'down', 'left', 'right']:
            new_board, has_moved = AIEvaluator.simulate_move(board, direction)
            if has_moved:
                valid_moves.append(direction)
        
        if not valid_moves:
            return None  # No moves available
        
        # Use expectiminimax with depth 2 for good performance
        return AIEvaluator.get_best_move(board, depth=2)


# Public API function for backward compatibility
def get_ai_suggestion(board: List[List[Optional[int]]]) -> Optional[str]:
    """Main AI function that returns the best move suggestion for the current board."""
    return AIEvaluator.get_ai_suggestion(board)
