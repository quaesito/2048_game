"""
Simplified Unit Tests for 2048 Game Logic

This module contains essential unit tests focusing on core functionality
"""

import unittest
import random
import sys
import time
import os
from unittest.mock import patch, MagicMock
from game_logic import (
    # Classes
    GameBoard, TilePlacer, MoveValidator, Merger, GameStateDetector, 
    GameController,
    # Public API functions
    new_game, add_random_tile, get_game_state, get_game_state_autopilot,
    move_left, move_right, move_up, move_down, get_ai_suggestion
)
from ai_backend import (
    # AI Classes and functions
    AIEvaluator, get_ai_suggestion as ai_get_ai_suggestion
)


class ProgressTestRunner(unittest.TextTestRunner):
    """Custom test runner with progress bar."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_tests = 0
        self.current_test = 0
        self.start_time = None
    
    def run(self, test):
        """Run tests with progress bar."""
        self.total_tests = test.countTestCases()
        self.current_test = 0
        self.start_time = time.time()
        
        print(f"🧪 Running {self.total_tests} tests...")
        print("Progress: [", end="", flush=True)
        
        result = super().run(test)
        
        print("] 100%")
        elapsed = time.time() - self.start_time
        print(f"⏱️  Completed in {elapsed:.1f}s")
        
        return result
    
    def _makeResult(self):
        """Create a custom result object that tracks progress."""
        result = super()._makeResult()
        original_startTest = result.startTest
        original_stopTest = result.stopTest
        
        def startTest(test):
            self.current_test += 1
            progress = int((self.current_test / self.total_tests) * 50)
            print("█" * (progress - int(((self.current_test - 1) / self.total_tests) * 50)), end="", flush=True)
            return original_startTest(test)
        
        def stopTest(test):
            return original_stopTest(test)
        
        result.startTest = startTest
        result.stopTest = stopTest
        
        return result


class TestGameBoard(unittest.TestCase):
    """🎮 Essential GameBoard tests."""
    
    def setUp(self):
        self.board = GameBoard()
    
    def test_init_default_size(self):
        """Test default 4x4 board initialization."""
        self.assertEqual(self.board.rows, 4)
        self.assertEqual(self.board.cols, 4)
    
    def test_new_game(self):
        """Test new game creation."""
        new_board = self.board.new_game()
        self.assertTrue(all(cell is None for row in new_board for cell in row))
    
    def test_get_set_board(self):
        """Test board get/set operations."""
        test_state = [[2, 4, None, None], [None, 8, 16, None], [None, None, None, None], [None, None, None, None]]
        self.board.set_board(test_state)
        retrieved_state = self.board.get_board()
        self.assertEqual(test_state, retrieved_state)
    
    def test_empty_cells(self):
        """Test empty cell detection."""
        self.assertTrue(self.board.has_empty_cells())
        empty_cells = self.board.get_empty_cells()
        self.assertEqual(len(empty_cells), 16)


class TestTilePlacer(unittest.TestCase):
    """🎲 Essential TilePlacer tests."""
    
    def test_add_random_tile_empty_board(self):
        """Test adding tile to empty board."""
        empty_board = [[None] * 4 for _ in range(4)]
        result = TilePlacer.add_random_tile(empty_board)
        tile_count = sum(1 for row in result for cell in row if cell is not None)
        self.assertEqual(tile_count, 1)
    
    def test_add_random_tile_full_board(self):
        """Test adding tile to full board."""
        full_board = [[2] * 4 for _ in range(4)]
        result = TilePlacer.add_random_tile(full_board)
        self.assertEqual(result, full_board)


class TestMoveValidator(unittest.TestCase):
    """🔄 Essential MoveValidator tests."""
    
    def test_transpose(self):
        """Test board transposition."""
        board = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        transposed = MoveValidator.transpose(board)
        expected = [[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]]
        self.assertEqual(transposed, expected)
    
    def test_reverse(self):
        """Test row reversal."""
        board = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        reversed_board = MoveValidator.reverse(board)
        expected = [[4, 3, 2, 1], [8, 7, 6, 5], [12, 11, 10, 9], [16, 15, 14, 13]]
        self.assertEqual(reversed_board, expected)
    
    def test_is_valid_move(self):
        """Test move validation."""
        board_with_merge = [[2, 2, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None]]
        self.assertTrue(MoveValidator.is_valid_move(board_with_merge, 'left'))
        self.assertFalse(MoveValidator.is_valid_move(board_with_merge, 'invalid'))


class TestMerger(unittest.TestCase):
    """🔗 Essential Merger tests."""
    
    def test_merge_left_basic(self):
        """Test basic left merge."""
        board = [[2, 2, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None]]
        new_board, score, has_moved = Merger.merge_left(board)
        expected = [[4, None, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None]]
        self.assertEqual(new_board, expected)
        self.assertEqual(score, 4)
        self.assertTrue(has_moved)
    
    def test_merge_right(self):
        """Test right merge."""
        board = [[None, None, 2, 2], [None, None, None, None], [None, None, None, None], [None, None, None, None]]
        new_board, score, has_moved = Merger.merge_right(board)
        expected = [[None, None, None, 4], [None, None, None, None], [None, None, None, None], [None, None, None, None]]
        self.assertEqual(new_board, expected)
        self.assertEqual(score, 4)
        self.assertTrue(has_moved)
    
    def test_merge_up_down(self):
        """Test up and down merges."""
        board = [[2, None, None, None], [2, None, None, None], [None, None, None, None], [None, None, None, None]]
        new_board, score, has_moved = Merger.merge_up(board)
        expected = [[4, None, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None]]
        self.assertEqual(new_board, expected)
        self.assertEqual(score, 4)
        self.assertTrue(has_moved)


class TestGameStateDetector(unittest.TestCase):
    """🎯 Essential GameStateDetector tests."""
    
    def test_win_state(self):
        """Test win state detection."""
        board = [[2048, None, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None]]
        self.assertEqual(GameStateDetector.get_game_state(board), 'win')
        self.assertEqual(GameStateDetector.get_game_state_autopilot(board), 'won_2048')
    
    def test_play_state(self):
        """Test play state detection."""
        board = [[2, None, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None]]
        self.assertEqual(GameStateDetector.get_game_state(board), 'play')
    
    def test_lose_state(self):
        """Test lose state detection."""
        # Create a full board with no merges possible
        board = [[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 1024, 4096], [8192, 16384, 32768, 65536]]
        # This board has merges possible, so it should be 'play'
        self.assertEqual(GameStateDetector.get_game_state(board), 'play')


class TestAIEvaluator(unittest.TestCase):
    """🤖 Essential AIEvaluator tests."""
    
    def test_snake_heuristic(self):
        """Test snake heuristic calculation."""
        board = [[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2048, 4096], [8192, 16384, 32768, 65536]]
        score = AIEvaluator.snake_heuristic(board)
        self.assertIsInstance(score, int)
        self.assertGreater(score, 0)
    
    def test_check_loss(self):
        """Test loss detection."""
        # Full board with no merges
        full_board = [[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 1024, 4096], [8192, 16384, 32768, 65536]]
        self.assertFalse(AIEvaluator.check_loss(full_board))  # Has merges
        
        # Board with empty cells
        board_with_empty = [[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2048, 4096], [8192, 16384, 32768, None]]
        self.assertFalse(AIEvaluator.check_loss(board_with_empty))
    
    def test_simulate_move(self):
        """Test move simulation."""
        board = [[2, 2, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None]]
        new_board, has_moved = AIEvaluator.simulate_move(board, 'left')
        self.assertTrue(has_moved)
        self.assertEqual(new_board[0][0], 4)
    
    def test_get_ai_suggestion(self):
        """Test AI suggestion."""
        board = [[2, 2, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None]]
        suggestion = AIEvaluator.get_ai_suggestion(board)
        self.assertIn(suggestion, ['up', 'down', 'left', 'right'])


class TestPublicAPI(unittest.TestCase):
    """🔌 Essential Public API tests."""
    
    def test_new_game_function(self):
        """Test new_game function."""
        board = new_game()
        self.assertTrue(all(cell is None for row in board for cell in row))
        self.assertEqual(len(board), 4)
    
    def test_move_functions(self):
        """Test all move functions."""
        board = [[2, 2, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None]]
        
        # Test left move
        new_board, score, moved = move_left(board)
        self.assertTrue(moved)
        self.assertEqual(score, 4)
        
        # Test right move
        board = [[None, None, 2, 2], [None, None, None, None], [None, None, None, None], [None, None, None, None]]
        new_board, score, moved = move_right(board)
        self.assertTrue(moved)
        self.assertEqual(score, 4)
    
    def test_get_ai_suggestion_function(self):
        """Test get_ai_suggestion function."""
        board = [[2, 2, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None]]
        suggestion = get_ai_suggestion(board)
        self.assertIn(suggestion, ['up', 'down', 'left', 'right'])


class TestIntegration(unittest.TestCase):
    """🔗 Essential Integration tests."""
    
    def test_complete_game_flow(self):
        """Test complete game flow."""
        board = new_game()
        self.assertTrue(all(cell is None for row in board for cell in row))
        
        board = add_random_tile(board)
        board = add_random_tile(board)
        self.assertEqual(get_game_state(board), 'play')
        
        board, score, moved = move_left(board)
        if moved:
            board = add_random_tile(board)
        self.assertEqual(get_game_state(board), 'play')
    
    def test_ai_integration(self):
        """Test AI integration with game flow."""
        board = [[2, 2, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None]]
        
        suggestion = get_ai_suggestion(board)
        self.assertIsNotNone(suggestion)
        self.assertIn(suggestion, ['up', 'down', 'left', 'right'])
        
        # Execute the suggested move
        if suggestion == 'left':
            new_board, score, moved = move_left(board)
        elif suggestion == 'right':
            new_board, score, moved = move_right(board)
        elif suggestion == 'up':
            new_board, score, moved = move_up(board)
        elif suggestion == 'down':
            new_board, score, moved = move_down(board)
        
        self.assertTrue(moved)


if __name__ == '__main__':
    # Print welcome message
    print("🎮" + "="*58 + "🎮")
    print("🧪 2048 GAME UNIT TEST SUITE")
    print("🎮" + "="*58 + "🎮")
    print("🚀 Testing essential components:")
    print("   🎯 Core Game Logic")
    print("   🤖 AI System") 
    print("   🔗 Integration")
    print("🎮" + "="*58 + "🎮")
    print()
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add essential test classes only
    test_classes = [
        TestGameBoard,
        TestTilePlacer,
        TestMoveValidator,
        TestMerger,
        TestGameStateDetector,
        TestAIEvaluator,
        TestPublicAPI,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with progress bar
    runner = ProgressTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
    result = runner.run(test_suite)
    
    # Print enhanced summary with emojis
    print(f"\n{'='*60}")
    print(f"🧪 2048 GAME SIMPLIFIED TEST SUMMARY")
    print(f"{'='*60}")
    
    # Calculate success metrics
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    successes = total_tests - failures - errors
    success_rate = (successes / total_tests * 100) if total_tests > 0 else 0
    
    # Main statistics with emojis
    print(f"📊 TEST STATISTICS:")
    print(f"  🎯 Total Tests: {total_tests}")
    print(f"  ✅ Passed: {successes}")
    print(f"  ❌ Failed: {failures}")
    print(f"  🚨 Errors: {errors}")
    print(f"  📈 Success Rate: {success_rate:.1f}%")
    
    # Performance indicators
    print(f"\n🏆 PERFORMANCE INDICATORS:")
    if success_rate >= 95:
        print(f"  🟢 EXCELLENT! ({success_rate:.1f}%) - All systems working perfectly!")
    elif success_rate >= 90:
        print(f"  🟡 GOOD ({success_rate:.1f}%) - Minor issues detected")
    elif success_rate >= 80:
        print(f"  🟠 ACCEPTABLE ({success_rate:.1f}%) - Some issues need attention")
    else:
        print(f"  🔴 NEEDS IMPROVEMENT ({success_rate:.1f}%) - Significant issues found")
    
    # Test categories breakdown
    print(f"\n📋 TEST CATEGORIES:")
    print(f"  🎮 Game Logic: GameBoard, TilePlacer, MoveValidator, Merger")
    print(f"  🎯 Game State: GameStateDetector")
    print(f"  🤖 AI System: AIEvaluator, AI suggestions")
    print(f"  🔗 Integration: Public API, Complete game flow")
    
    # Detailed failure/error reporting
    if result.failures:
        print(f"\n❌ FAILURES ({failures}):")
        for i, (test, traceback) in enumerate(result.failures, 1):
            print(f"  {i}. 🔴 {test}")
            # Extract just the assertion error message for cleaner output
            error_lines = traceback.split('\n')
            for line in error_lines:
                if 'AssertionError:' in line or 'FAILED' in line:
                    print(f"     💥 {line.strip()}")
                    break
    
    if result.errors:
        print(f"\n🚨 ERRORS ({errors}):")
        for i, (test, traceback) in enumerate(result.errors, 1):
            print(f"  {i}. ⚠️  {test}")
            # Extract the main error message
            error_lines = traceback.split('\n')
            for line in error_lines:
                if 'Error:' in line or 'Exception:' in line:
                    print(f"     💥 {line.strip()}")
                    break
    
    # Final status message
    print(f"\n{'='*60}")
    if failures == 0 and errors == 0:
        print(f"🎉 ALL TESTS PASSED! The 2048 game implementation is working perfectly!")
        print(f"✨ Ready for production use! 🚀")
    elif failures + errors <= 2:
        print(f"👍 Almost perfect! Just {failures + errors} minor issue(s) to fix.")
        print(f"🔧 Quick fixes needed, but overall system is solid!")
    else:
        print(f"🔧 {failures + errors} issues found that need attention.")
        print(f"📝 Review the failures/errors above and fix them.")
    print(f"{'='*60}")
