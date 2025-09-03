"""
2048 Game Logic - Modular Implementation

This module provides a clean, modular implementation of the 2048 game logic,
broken down into distinct stages with clear responsibilities:
- Board generation and management
- Move logic and validation
- Merging rules and scoring
- New tile placement
- End-game detection
- AI decision making
"""

import random
import copy
from typing import List, Tuple, Optional, Dict, Any


class GameBoard:
    """Handles board generation, state management, and basic operations."""
    
    def __init__(self, rows: int = 4, cols: int = 4):
        """Initialize a new game board with specified dimensions."""
        self.rows = rows
        self.cols = cols
        # Create initial empty board state
        self.board = self._create_empty_board()
    
    def _create_empty_board(self) -> List[List[Optional[int]]]:
        """Create an empty board filled with None values."""
        return [[None for _ in range(self.cols)] for _ in range(self.rows)]
    
    def new_game(self) -> List[List[Optional[int]]]:
        """Reset the board to a new empty state."""
        self.board = self._create_empty_board()
        return self.board
    
    def get_board(self) -> List[List[Optional[int]]]:
        """Get a copy of the current board state."""
        # Return a deep copy to prevent external modifications
        return [row[:] for row in self.board]
    
    def set_board(self, board: List[List[Optional[int]]]) -> None:
        """Set the board to a new state."""
        # Create a deep copy to prevent external modifications
        self.board = [row[:] for row in board]
    
    def get_empty_cells(self) -> List[Tuple[int, int]]:
        """Get list of all empty cell positions (row, col)."""
        empty_cells = []
        # Scan entire board for empty cells (None values)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] is None:
                    empty_cells.append((r, c))
        return empty_cells
    
    def has_empty_cells(self) -> bool:
        """Check if the board has any empty cells."""
        return len(self.get_empty_cells()) > 0


class TilePlacer:
    """Handles random tile placement on the board."""
    
    @staticmethod
    def add_random_tile(board: List[List[Optional[int]]]) -> List[List[Optional[int]]]:
        """Add a random tile (2 or 4) to an empty cell on the board."""
        empty_cells = []
        # Find all empty cells on the board
        for r in range(len(board)):
            for c in range(len(board[r])):
                if board[r][c] is None:
                    empty_cells.append((r, c))
        
        # Return unchanged board if no empty cells
        if not empty_cells:
            return board
        
        # Choose random empty cell and place tile
        r, c = random.choice(empty_cells)
        # 90% chance for 2, 10% chance for 4 (matching original 2048 game)
        board[r][c] = 2 if random.random() < 0.9 else 4
        return board


class MoveValidator:
    """Handles move validation and board transformations."""
    
    @staticmethod
    def transpose(board: List[List[Optional[int]]]) -> List[List[Optional[int]]]:
        """Transpose the board matrix (swap rows and columns)."""
        # Convert zip result to list of lists for proper matrix format
        return [list(row) for row in zip(*board)]
    
    @staticmethod
    def reverse(board: List[List[Optional[int]]]) -> List[List[Optional[int]]]:
        """Reverse each row of the board."""
        # Reverse each row using slice notation
        return [row[::-1] for row in board]
    
    @staticmethod
    def is_valid_move(board: List[List[Optional[int]]], direction: str) -> bool:
        """Check if a move in the given direction is valid (will change the board)."""
        # Test the move by applying the appropriate merge operation
        if direction == 'up':
            new_board, _, has_moved = Merger.merge_up(board)
        elif direction == 'down':
            new_board, _, has_moved = Merger.merge_down(board)
        elif direction == 'left':
            new_board, _, has_moved = Merger.merge_left(board)
        elif direction == 'right':
            new_board, _, has_moved = Merger.merge_right(board)
        else:
            return False
        
        # Return whether the board actually changed
        return has_moved


class Merger:
    """Handles tile merging logic and scoring."""
    
    @staticmethod
    def merge_left(board: List[List[Optional[int]]]) -> Tuple[List[List[Optional[int]]], int, bool]:
        """Merge tiles to the left and return new board, score, and move status."""
        new_board = []
        total_score = 0
        
        for row in board:
            # Step 1: Remove empty cells by sliding non-empty cells to the left
            slid_row = [cell for cell in row if cell is not None]
            
            # Step 2: Merge adjacent equal cells and calculate score
            merged_row = []
            i = 0
            while i < len(slid_row):
                # Check if current cell can merge with next cell
                if i + 1 < len(slid_row) and slid_row[i] == slid_row[i + 1]:
                    merged_value = slid_row[i] * 2
                    merged_row.append(merged_value)
                    total_score += merged_value
                    i += 2  # Skip the next cell as it's been merged
                else:
                    merged_row.append(slid_row[i])
                    i += 1
            
            # Step 3: Pad the row with None values to maintain original length
            while len(merged_row) < len(row):
                merged_row.append(None)
            
            new_board.append(merged_row)
        
        # Determine if the move actually changed the board state
        has_moved = board != new_board
        
        return new_board, total_score, has_moved
    
    @staticmethod
    def merge_right(board: List[List[Optional[int]]]) -> Tuple[List[List[Optional[int]]], int, bool]:
        """Merge tiles to the right and return new board, score, and move status."""
        # Use left merge by reversing, merging left, then reversing back
        reversed_board = MoveValidator.reverse(board)
        merged_board, score, has_moved = Merger.merge_left(reversed_board)
        final_board = MoveValidator.reverse(merged_board)
        return final_board, score, has_moved
    
    @staticmethod
    def merge_up(board: List[List[Optional[int]]]) -> Tuple[List[List[Optional[int]]], int, bool]:
        """Merge tiles upward and return new board, score, and move status."""
        # Use left merge by transposing, merging left, then transposing back
        transposed_board = MoveValidator.transpose(board)
        merged_board, score, has_moved = Merger.merge_left(transposed_board)
        final_board = MoveValidator.transpose(merged_board)
        return final_board, score, has_moved
    
    @staticmethod
    def merge_down(board: List[List[Optional[int]]]) -> Tuple[List[List[Optional[int]]], int, bool]:
        """Merge tiles downward and return new board, score, and move status."""
        # Use left merge by transposing, reversing, merging left, then reversing and transposing back
        transposed_board = MoveValidator.transpose(board)
        reversed_board = MoveValidator.reverse(transposed_board)
        merged_board, score, has_moved = Merger.merge_left(reversed_board)
        final_board = MoveValidator.reverse(merged_board)
        final_board = MoveValidator.transpose(final_board)
        return final_board, score, has_moved


class GameStateDetector:
    """Handles end-game detection and game state management."""
    
    @staticmethod
    def get_game_state(board: List[List[Optional[int]]]) -> str:
        """Check if game is won (2048 tile), lost (no moves), or still playing."""
        # First check for 2048 tile (win condition)
        for row in board:
            for cell in row:
                if cell == 2048:
                    return 'win'
        
        # Game continues if there are empty cells
        has_empty_cell = any(cell is None for row in board for cell in row)
        if has_empty_cell:
            return 'play'
        
        # Check for possible merges in all directions
        for r in range(len(board)):
            for c in range(len(board[r])):
                # Check if can merge with right neighbor
                if c < len(board[r]) - 1 and board[r][c] == board[r][c + 1]:
                    return 'play'
                # Check if can merge with bottom neighbor
                if r < len(board) - 1 and board[r][c] == board[r + 1][c]:
                    return 'play'
        
        # No moves possible - game lost
        return 'lose'
    
    @staticmethod
    def get_game_state_autopilot(board: List[List[Optional[int]]]) -> str:
        """Check game state for autopilot mode - allows continuing past 2048."""
        # Check for 2048 tile (special state for autopilot - allows continuing)
        for row in board:
            for cell in row:
                if cell == 2048:
                    return 'won_2048'
        
        # Game continues if there are empty cells
        has_empty_cell = any(cell is None for row in board for cell in row)
        if has_empty_cell:
            return 'play'
        
        # Check for possible merges in all directions
        for r in range(len(board)):
            for c in range(len(board[r])):
                # Check if can merge with right neighbor
                if c < len(board[r]) - 1 and board[r][c] == board[r][c + 1]:
                    return 'play'
                # Check if can merge with bottom neighbor
                if r < len(board) - 1 and board[r][c] == board[r + 1][c]:
                    return 'play'
        
        # No moves possible - game lost
        return 'lose'


class GameController:
    """Main game controller that orchestrates all game operations."""
    
    def __init__(self):
        # Initialize all game components
        self.board_manager = GameBoard()
        self.tile_placer = TilePlacer()
        self.move_validator = MoveValidator()
        self.merger = Merger()
        self.state_detector = GameStateDetector()
    
    def new_game(self) -> List[List[Optional[int]]]:
        """Start a new game and return the initial board state."""
        return self.board_manager.new_game()
    
    def add_random_tile(self, board: List[List[Optional[int]]]) -> List[List[Optional[int]]]:
        """Add a random tile to the board."""
        return self.tile_placer.add_random_tile(board)
    
    def move(self, board: List[List[Optional[int]]], direction: str) -> Tuple[List[List[Optional[int]]], int, bool]:
        """Execute a move in the given direction and return new board, score, and move status."""
        # Route to appropriate merge operation based on direction
        if direction == 'up':
            return self.merger.merge_up(board)
        elif direction == 'down':
            return self.merger.merge_down(board)
        elif direction == 'left':
            return self.merger.merge_left(board)
        elif direction == 'right':
            return self.merger.merge_right(board)
        else:
            # Invalid direction - return unchanged board
            return board, 0, False
    
    def get_game_state(self, board: List[List[Optional[int]]]) -> str:
        """Get the current game state."""
        return self.state_detector.get_game_state(board)
    
    def get_game_state_autopilot(self, board: List[List[Optional[int]]]) -> str:
        """Get the current game state for autopilot mode."""
        return self.state_detector.get_game_state_autopilot(board)

# Global game controller instance for singleton pattern
_game_controller = GameController()

# Public API functions for backward compatibility with existing code
def new_game(rows: int = 4, cols: int = 4) -> List[List[Optional[int]]]:
    """Create a new empty 4x4 game board."""
    return _game_controller.new_game()

def add_random_tile(board: List[List[Optional[int]]]) -> List[List[Optional[int]]]:
    """Add a random tile (2 or 4) to an empty cell on the board."""
    return _game_controller.add_random_tile(board)

def get_game_state(board: List[List[Optional[int]]]) -> str:
    """Check if game is won (2048 tile), lost (no moves), or still playing."""
    return _game_controller.get_game_state(board)

def get_game_state_autopilot(board: List[List[Optional[int]]]) -> str:
    """Check game state for autopilot mode - allows continuing past 2048."""
    return _game_controller.get_game_state_autopilot(board)

def move_left(board: List[List[Optional[int]]]) -> Tuple[List[List[Optional[int]]], int, bool]:
    """Move all tiles to the left and merge equal adjacent tiles."""
    return _game_controller.move(board, 'left')

def move_right(board: List[List[Optional[int]]]) -> Tuple[List[List[Optional[int]]], int, bool]:
    """Move all tiles to the right and merge equal adjacent tiles."""
    return _game_controller.move(board, 'right')

def move_up(board: List[List[Optional[int]]]) -> Tuple[List[List[Optional[int]]], int, bool]:
    """Move all tiles upward and merge equal adjacent tiles."""
    return _game_controller.move(board, 'up')

def move_down(board: List[List[Optional[int]]]) -> Tuple[List[List[Optional[int]]], int, bool]:
    """Move all tiles downward and merge equal adjacent tiles."""
    return _game_controller.move(board, 'down')
def get_ai_suggestion(board: List[List[Optional[int]]]) -> Optional[str]:
    """Get AI move suggestion using the AI backend."""
    # Import here to avoid circular imports
    from ai_backend import get_ai_suggestion as ai_suggestion
    return ai_suggestion(board)


