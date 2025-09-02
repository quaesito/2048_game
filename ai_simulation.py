#!/usr/bin/env python3
"""
AI Simulation for 2048 Game
Tests the AI's ability to win the game by running automated games in async mode.

Usage:
    python ai_simulation.py [number_of_games]
    
Examples:
    python ai_simulation.py          # Run 50 games (default)
    python ai_simulation.py 100     # Run 100 games
    python ai_simulation.py 25      # Run 25 games
    python ai_simulation.py --help  # Show help
"""

import game_logic as game
import time
import sys
import argparse
from typing import List
from multiprocessing import Pool, cpu_count

class AISimulation:
    """AI simulation class for testing 2048 game AI performance."""
    def __init__(self):
        self.game_stats = {
            'games_played': 0,
            'games_won': 0,
            'games_lost': 0,
            'max_score': 0,
            'max_tile': 0,
            'total_moves': 0,
            'win_rate': 0.0,
            'total_ai_latency': 0.0,
            'ai_suggestions_count': 0,
            'total_game_time': 0.0,
            'min_ai_latency': float('inf'),
            'max_ai_latency': 0.0,
            'min_game_time': float('inf'),
            'max_game_time': 0.0,
            'all_max_tiles': [],  # Store all max tiles for distribution analysis
            'max_tile_distribution': {
                '2-4': 0,
                '8-16': 0,
                '32-64': 0,
                '128-256': 0,
                '512': 0,
                '1024': 0,
                '2048': 0,
                '4096+': 0
            }
        }
    
    def categorize_max_tile(self, max_tile):
        """Categorize max tile value into distribution buckets for statistics."""
        if max_tile <= 4:
            return '2-4'
        elif max_tile <= 16:
            return '8-16'
        elif max_tile <= 64:
            return '32-64'
        elif max_tile <= 256:
            return '128-256'
        elif max_tile == 512:
            return '512'
        elif max_tile == 1024:
            return '1024'
        elif max_tile == 2048:
            return '2048'
        else:
            return '4096+'
    
    def run_single_game(self, max_moves: int = 10000, verbose: bool = False) -> dict:
        """Run a single AI-controlled game and return detailed statistics."""
        # Initialize game using the backend
        board = game.new_game()
        game.add_random_tile(board)
        game.add_random_tile(board)
        
        moves_made = 0
        score = 0
        max_tile_achieved = 0
        consecutive_no_progress = 0
        
        # Timing metrics
        game_start_time = time.time()
        ai_latencies = []
        ai_suggestions_count = 0
        
        if verbose:
            print("Starting new game...")
            self.print_board(board)
        
        while moves_made < max_moves:
            # Check game state using backend
            game_status = game.get_game_state(board)
            
            if game_status == 'win':
                if verbose:
                    print(f"🎉 AI WON! Achieved 2048 tile!")
                game_end_time = time.time()
                return {
                    'status': 'win',
                    'moves': moves_made,
                    'score': score,
                    'max_tile': max_tile_achieved,
                    'final_board': [row[:] for row in board],
                    'game_time': game_end_time - game_start_time,
                    'ai_latencies': ai_latencies,
                    'ai_suggestions_count': ai_suggestions_count
                }
            elif game_status == 'lose':
                if verbose:
                    print(f"💀 AI LOST! No more moves available.")
                game_end_time = time.time()
                return {
                    'status': 'lose',
                    'moves': moves_made,
                    'score': score,
                    'max_tile': max_tile_achieved,
                    'final_board': [row[:] for row in board],
                    'game_time': game_end_time - game_start_time,
                    'ai_latencies': ai_latencies,
                    'ai_suggestions_count': ai_suggestions_count
                }
            
            # Get AI suggestion using the backend with timing
            ai_start_time = time.time()
            ai_move = game.get_ai_suggestion(board)
            ai_end_time = time.time()
            
            ai_latency = ai_end_time - ai_start_time
            ai_latencies.append(ai_latency)
            ai_suggestions_count += 1
            
            if ai_move is None:
                if verbose:
                    print("AI couldn't find a valid move!")
                game_end_time = time.time()
                return {
                    'status': 'lose',
                    'moves': moves_made,
                    'score': score,
                    'max_tile': max_tile_achieved,
                    'final_board': [row[:] for row in board],
                    'game_time': game_end_time - game_start_time,
                    'ai_latencies': ai_latencies,
                    'ai_suggestions_count': ai_suggestions_count
                }
            
            # Execute the move using backend functions
            move_functions = {
                'up': game.move_up,
                'down': game.move_down,
                'left': game.move_left,
                'right': game.move_right,
            }
            
            new_board, score_gain, has_moved = move_functions[ai_move](board)
            
            if has_moved:
                board = new_board
                score += score_gain
                moves_made += 1
                consecutive_no_progress = 0
                
                # CRITICAL: Add a new random tile after each successful move
                game.add_random_tile(board)
                
                # Track max tile
                for row in board:
                    for cell in row:
                        if cell is not None and cell > max_tile_achieved:
                            max_tile_achieved = cell
                
                if verbose and moves_made % 100 == 0:
                    print(f"Move {moves_made}: {ai_move}, Score: {score}, Max Tile: {max_tile_achieved}")
            else:
                consecutive_no_progress += 1
                if verbose:
                    print(f"Move {ai_move} didn't change the board! ({consecutive_no_progress} consecutive)")
                
                # If we haven't made progress for too long, break
                if consecutive_no_progress > 10:
                    if verbose:
                        print("Too many consecutive non-progress moves, ending game.")
                    break
        
        # Game ended due to max moves or no progress
        game_end_time = time.time()
        return {
            'status': 'timeout',
            'moves': moves_made,
            'score': score,
            'max_tile': max_tile_achieved,
            'final_board': [row[:] for row in board],
            'game_time': game_end_time - game_start_time,
            'ai_latencies': ai_latencies,
            'ai_suggestions_count': ai_suggestions_count
        }
    
    def run_multiple_games_async(self, num_games: int = 10, max_workers: int = None, max_moves: int = 5000) -> dict:
        """Run multiple AI games in parallel using multiprocessing for performance testing."""
        if max_workers is None:
            max_workers = min(cpu_count(), 5)  # Max 5 parallel games per batch
        
        print(f"🚀 Running AI simulation for {num_games} games in batches of {max_workers}...")
        print(f"⚡ Using {max_workers} CPU cores per batch")
        print("=" * 60)
        
        all_results = []
        total_execution_time = 0
        
        # Process games in batches of max_workers
        for batch_start in range(0, num_games, max_workers):
            batch_end = min(batch_start + max_workers, num_games)
            batch_size = batch_end - batch_start
            
            print(f"\n📦 Processing batch {batch_start // max_workers + 1}: Games {batch_start + 1}-{batch_end}")
            
            start_time = time.time()
            
            # Run current batch in parallel
            with Pool(processes=batch_size) as pool:
                # Create list of (game_number, max_moves) tuples for this batch
                game_args = [(game_num, max_moves) for game_num in range(batch_start + 1, batch_end + 1)]
                
                # Run batch games in parallel
                batch_results = pool.map(self._run_single_game_wrapper, game_args)
            
            end_time = time.time()
            batch_execution_time = end_time - start_time
            total_execution_time += batch_execution_time
            
            # Process batch results
            self._process_batch_results(batch_results, batch_execution_time)
            all_results.extend(batch_results)
        
        # Calculate final statistics
        if self.game_stats['games_played'] > 0:
            self.game_stats['win_rate'] = (self.game_stats['games_won'] / self.game_stats['games_played']) * 100
        
        print(f"\n⏱️  Total execution time: {total_execution_time:.2f} seconds")
        print(f"🎯 Games per second: {len(all_results) / total_execution_time:.1f}")
        
        return {
            'results': all_results,
            'stats': self.game_stats,
            'execution_time': total_execution_time
        }
    
    def _process_batch_results(self, results: List[dict], execution_time: float) -> None:
        """Process and display results from a batch of parallel game simulations."""
        print(f"📊 Processing {len(results)} game results from batch...")
        
        # Sort results by game number for consistent output
        results.sort(key=lambda x: x['game_num'])
        
        # Process each result
        for result in results:
            game_num = result['game_num']
            
            # Update statistics
            self.game_stats['games_played'] += 1
            self.game_stats['total_moves'] += result['moves']
            
            # Update max tile distribution
            max_tile = result['max_tile']
            self.game_stats['all_max_tiles'].append(max_tile)
            max_tile_category = self.categorize_max_tile(max_tile)
            self.game_stats['max_tile_distribution'][max_tile_category] += 1
            
            # Update timing statistics
            if 'ai_latencies' in result and result['ai_latencies']:
                for latency in result['ai_latencies']:
                    self.game_stats['total_ai_latency'] += latency
                    self.game_stats['ai_suggestions_count'] += 1
                    self.game_stats['min_ai_latency'] = min(self.game_stats['min_ai_latency'], latency)
                    self.game_stats['max_ai_latency'] = max(self.game_stats['max_ai_latency'], latency)
            
            if 'game_time' in result:
                game_time = result['game_time']
                self.game_stats['total_game_time'] += game_time
                self.game_stats['min_game_time'] = min(self.game_stats['min_game_time'], game_time)
                self.game_stats['max_game_time'] = max(self.game_stats['max_game_time'], game_time)
            
            if result['status'] == 'win':
                self.game_stats['games_won'] += 1
                avg_ai_latency = sum(result.get('ai_latencies', [0])) / max(len(result.get('ai_latencies', [1])), 1)
                print(f"✅ Game {game_num:2d}: WON!    Score: {result['score']:6,}, Moves: {result['moves']:4d}, Max Tile: {result['max_tile']}, Time: {result.get('game_time', 0):.2f}s, Avg AI: {avg_ai_latency:.3f}s")
            elif result['status'] == 'lose':
                self.game_stats['games_lost'] += 1
                avg_ai_latency = sum(result.get('ai_latencies', [0])) / max(len(result.get('ai_latencies', [1])), 1)
                print(f"❌ Game {game_num:2d}: LOST!   Score: {result['score']:6,}, Moves: {result['moves']:4d}, Max Tile: {result['max_tile']}, Time: {result.get('game_time', 0):.2f}s, Avg AI: {avg_ai_latency:.3f}s")
            else:
                avg_ai_latency = sum(result.get('ai_latencies', [0])) / max(len(result.get('ai_latencies', [1])), 1)
                print(f"⏰ Game {game_num:2d}: TIMEOUT Score: {result['score']:6,}, Moves: {result['moves']:4d}, Max Tile: {result['max_tile']}, Time: {result.get('game_time', 0):.2f}s, Avg AI: {avg_ai_latency:.3f}s")
            
            if result['score'] > self.game_stats['max_score']:
                self.game_stats['max_score'] = result['score']
            
            if result['max_tile'] > self.game_stats['max_tile']:
                self.game_stats['max_tile'] = result['max_tile']
        
        print(f"⏱️  Batch execution time: {execution_time:.2f} seconds")
        print(f"🎯 Batch speed: {len(results) / execution_time:.1f} games/second")
    
    @staticmethod
    def _run_single_game_wrapper(args_tuple) -> dict:
        """Wrapper function for multiprocessing to run a single game simulation."""
        game_num, max_moves = args_tuple
        # Create a new simulation instance for each process
        simulation = AISimulation()
        result = simulation.run_single_game(max_moves=max_moves, verbose=False)
        result['game_num'] = game_num
        return result
    
    def print_board(self, board: List[List]) -> None:
        """Print the game board in a formatted grid layout."""
        print("\n" + "=" * 25)
        for row in board:
            row_str = "|"
            for cell in row:
                if cell is None:
                    row_str += "    |"
                else:
                    row_str += f"{cell:4d}|"
            print(row_str)
        print("=" * 25)
    
    def calculate_max_tile_percentiles(self, max_tiles):
        """Calculate percentile statistics for max tile distribution analysis."""
        if not max_tiles:
            return {}
        
        sorted_tiles = sorted(max_tiles)
        n = len(sorted_tiles)
        
        return {
            'min': sorted_tiles[0],
            '25th': sorted_tiles[int(n * 0.25)],
            '50th': sorted_tiles[int(n * 0.5)],
            '75th': sorted_tiles[int(n * 0.75)],
            '90th': sorted_tiles[int(n * 0.90)],
            '95th': sorted_tiles[int(n * 0.95)],
            '99th': sorted_tiles[int(n * 0.99)],
            'max': sorted_tiles[-1]
        }
    
    def print_final_statistics(self, stats: dict, execution_time: float = None) -> None:
        """Print comprehensive simulation statistics with performance analysis."""
        print("\n" + "=" * 80)
        print("🏆 COMPREHENSIVE AI SIMULATION RESULTS")
        print("=" * 80)
        
        # Basic Game Statistics
        print("📊 GAME PERFORMANCE:")
        print(f"  Games Played: {stats['games_played']}")
        print(f"  Games Won: {stats['games_won']}")
        print(f"  Games Lost: {stats['games_lost']}")
        print(f"  Win Rate: {stats['win_rate']:.1f}%")
        print(f"  Highest Score: {stats['max_score']:,}")
        print(f"  Highest Tile: {stats['max_tile']}")
        print(f"  Average Moves per Game: {stats['total_moves'] / stats['games_played']:.1f}")
        
        # Max Tile Distribution Analysis
        print("\n📊 MAX TILE DISTRIBUTION ANALYSIS:")
        if stats['all_max_tiles']:
            percentiles = self.calculate_max_tile_percentiles(stats['all_max_tiles'])
            print(f"  Max Tile Percentiles:")
            print(f"    Min: {percentiles['min']}")
            print(f"    25th: {percentiles['25th']}")
            print(f"    50th (Median): {percentiles['50th']}")
            print(f"    75th: {percentiles['75th']}")
            print(f"    90th: {percentiles['90th']}")
            print(f"    95th: {percentiles['95th']}")
            print(f"    99th: {percentiles['99th']}")
            print(f"    Max: {percentiles['max']}")
            
            print(f"  Max Tile Distribution by Range:")
            total_games = stats['games_played']
            for range_name, count in stats['max_tile_distribution'].items():
                percentage = (count / total_games) * 100 if total_games > 0 else 0
                bar_length = int(percentage / 2)  # Scale for visual representation
                bar = "█" * bar_length + "░" * (25 - bar_length)
                print(f"    {range_name:>12}: {count:3d} games ({percentage:5.1f}%) {bar}")
            
            # Success rate analysis
            games_2048 = stats['max_tile_distribution']['2048']
            games_1024_plus = sum(stats['max_tile_distribution'][k] for k in ['1024', '2048', '4096+'])
            games_512_plus = sum(stats['max_tile_distribution'][k] for k in ['512', '1024', '2048', '4096+'])
            
            print(f"\n  🎯 SUCCESS RATE ANALYSIS:")
            print(f"    Games reaching 2048: {games_2048} ({games_2048/total_games*100:.1f}%)")
            print(f"    Games reaching 1024+: {games_1024_plus} ({games_1024_plus/total_games*100:.1f}%)")
            print(f"    Games reaching 512+: {games_512_plus} ({games_512_plus/total_games*100:.1f}%)")
        
        # AI Performance Metrics
        print("\n🤖 AI PERFORMANCE METRICS:")
        if stats['ai_suggestions_count'] > 0:
            avg_ai_latency = stats['total_ai_latency'] / stats['ai_suggestions_count']
            print(f"  Total AI Suggestions: {stats['ai_suggestions_count']:,}")
            print(f"  Average AI Latency: {avg_ai_latency:.3f} seconds")
            print(f"  Min AI Latency: {stats['min_ai_latency']:.3f} seconds")
            print(f"  Max AI Latency: {stats['max_ai_latency']:.3f} seconds")
            print(f"  AI Suggestions per Game: {stats['ai_suggestions_count'] / stats['games_played']:.1f}")
        else:
            print("  No AI latency data available")
        
        # Game Timing Statistics
        print("\n⏱️  GAME TIMING STATISTICS:")
        if stats['games_played'] > 0:
            avg_game_time = stats['total_game_time'] / stats['games_played']
            print(f"  Average Game Time: {avg_game_time:.2f} seconds")
            print(f"  Min Game Time: {stats['min_game_time']:.2f} seconds")
            print(f"  Max Game Time: {stats['max_game_time']:.2f} seconds")
            print(f"  Total Simulation Time: {stats['total_game_time']:.2f} seconds")
        
        # Execution Performance
        if execution_time:
            print("\n🚀 EXECUTION PERFORMANCE:")
            print(f"  Total Execution Time: {execution_time:.2f} seconds")
            print(f"  Games per Second: {stats['games_played'] / execution_time:.1f}")
            print(f"  Parallel Efficiency: {(stats['total_game_time'] / execution_time) * 100:.1f}%")
        
        # Performance Analysis
        print("\n📈 PERFORMANCE ANALYSIS:")
        if stats['ai_suggestions_count'] > 0:
            avg_ai_latency = stats['total_ai_latency'] / stats['ai_suggestions_count']
            if avg_ai_latency < 0.5:
                print("  🟢 AI Response Time: EXCELLENT (< 0.5s)")
            elif avg_ai_latency < 1.0:
                print("  🟡 AI Response Time: GOOD (0.5-1.0s)")
            elif avg_ai_latency < 2.0:
                print("  🟠 AI Response Time: ACCEPTABLE (1.0-2.0s)")
            else:
                print("  🔴 AI Response Time: SLOW (> 2.0s)")
        
        if stats['win_rate'] >= 95:
            print("  🟢 Win Rate: EXCELLENT (≥95%)")
        elif stats['win_rate'] >= 80:
            print("  🟡 Win Rate: GOOD (80-95%)")
        elif stats['win_rate'] >= 60:
            print("  🟠 Win Rate: ACCEPTABLE (60-80%)")
        else:
            print("  🔴 Win Rate: NEEDS IMPROVEMENT (<60%)")
        
        # Success Summary
        print("\n🎯 SUCCESS SUMMARY:")
        if stats['games_won'] > 0:
            print(f"  🎉 AI SUCCESSFULLY WON {stats['games_won']} GAME(S)!")
            print("  ✅ The AI can consistently achieve the 2048 tile!")
            if stats['win_rate'] >= 90:
                print("  🏆 WORLD-CLASS PERFORMANCE!")
            elif stats['win_rate'] >= 80:
                print("  🥇 EXCELLENT PERFORMANCE!")
            elif stats['win_rate'] >= 70:
                print("  🥈 GOOD PERFORMANCE!")
            else:
                print("  🥉 DECENT PERFORMANCE!")
        else:
            print(f"  🤔 AI didn't win any games, but achieved tile {stats['max_tile']}")
            print("  📈 The AI might need further tuning or more games to win.")
        
        print("=" * 80)

def parse_arguments():
    """Parse command-line arguments for configuring the AI simulation."""
    parser = argparse.ArgumentParser(
        description="2048 AI Simulation - Test AI performance with configurable number of games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ai_simulation.py          # Run 50 games (default)
  python ai_simulation.py 100     # Run 100 games
  python ai_simulation.py 25      # Run 25 games
  python ai_simulation.py --help  # Show this help message
        """
    )
    
    parser.add_argument(
        'games',
        nargs='?',
        type=int,
        default=50,
        metavar='GAMES',
        help='Number of games to simulate (default: 50)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        help='Maximum number of parallel workers (default: auto-detect)'
    )
    
    parser.add_argument(
        '--max-moves',
        type=int,
        default=5000,
        help='Maximum moves per game before timeout (default: 5000)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.games < 1:
        parser.error("Number of games must be at least 1")
    if args.games > 1000:
        print("⚠️  Warning: Large number of games may take a long time to complete")
    
    if args.max_workers is not None:
        if args.max_workers < 1:
            parser.error("Max workers must be at least 1")
        if args.max_workers > cpu_count():
            print(f"⚠️  Warning: Requested {args.max_workers} workers exceeds available CPU cores ({cpu_count()})")
    
    if args.max_moves < 100:
        parser.error("Max moves must be at least 100")
    if args.max_moves > 50000:
        print("⚠️  Warning: Very high max moves may cause games to run indefinitely")
    
    return args

def main():
    """Main function to run the AI simulation with command-line configuration."""
    # Parse command-line arguments
    args = parse_arguments()
    
    print("🎮 2048 AI Simulation - New AI Integration")
    print("Testing the new expectiminimax AI with snake heuristic...")
    print("Using game_logic.py backend with expectiminimax algorithm and snake pattern evaluation")
    print(f"📊 Configuration: {args.games} games, max {args.max_moves} moves per game")
    
    if args.max_workers:
        print(f"🔧 Custom max workers: {args.max_workers}")
    else:
        print(f"🔧 Auto-detected workers: {min(cpu_count(), 5)}")
    
    # Create simulation instance
    simulation = AISimulation()
    
    # Run simulation with command-line arguments
    results = simulation.run_multiple_games_async(
        num_games=args.games,
        max_workers=args.max_workers,
        max_moves=args.max_moves
    )
    
    # Print results
    simulation.print_final_statistics(results['stats'], results['execution_time'])
    
    # Show best game details
    best_game = max(results['results'], key=lambda x: x['score'])
    print(f"\n🏅 BEST GAME DETAILS:")
    print(f"Status: {best_game['status']}")
    print(f"Score: {best_game['score']:,}")
    print(f"Max Tile: {best_game['max_tile']}")
    print(f"Moves: {best_game['moves']}")
    
    if best_game['status'] == 'win':
        print("\n🎯 WINNING BOARD:")
        simulation.print_board(best_game['final_board'])
    
    # Show enhanced performance summary
    print(f"\n⚡ ENHANCED PERFORMANCE SUMMARY:")
    print(f"Total Games: {results['stats']['games_played']}")
    print(f"Execution Time: {results['execution_time']:.2f} seconds")
    print(f"Speed: {results['stats']['games_played'] / results['execution_time']:.1f} games/second")
    print(f"Win Rate: {results['stats']['win_rate']:.1f}%")
    
    if results['stats']['ai_suggestions_count'] > 0:
        avg_ai_latency = results['stats']['total_ai_latency'] / results['stats']['ai_suggestions_count']
        print(f"Average AI Latency: {avg_ai_latency:.3f} seconds")
        print(f"Total AI Suggestions: {results['stats']['ai_suggestions_count']:,}")
    
    if results['stats']['games_played'] > 0:
        avg_game_time = results['stats']['total_game_time'] / results['stats']['games_played']
        print(f"Average Game Time: {avg_game_time:.2f} seconds")
    
    # Show max tile distribution summary
    if results['stats']['all_max_tiles']:
        percentiles = simulation.calculate_max_tile_percentiles(results['stats']['all_max_tiles'])
        print(f"Max Tile Distribution:")
        print(f"  Median Max Tile: {percentiles['50th']}")
        print(f"  75th Percentile: {percentiles['75th']}")
        print(f"  90th Percentile: {percentiles['90th']}")
        print(f"  95th Percentile: {percentiles['95th']}")
        
        # Show success rates
        games_2048 = results['stats']['max_tile_distribution']['2048']
        games_1024_plus = sum(results['stats']['max_tile_distribution'][k] for k in ['1024', '2048', '4096+'])
        games_512_plus = sum(results['stats']['max_tile_distribution'][k] for k in ['512', '1024', '2048', '4096+'])
        total_games = len(results['stats']['all_max_tiles'])
        
        print(f"  Games reaching 2048: {games_2048} ({games_2048/total_games*100:.1f}%)")
        print(f"  Games reaching 1024+: {games_1024_plus} ({games_1024_plus/total_games*100:.1f}%)")
        print(f"  Games reaching 512+: {games_512_plus} ({games_512_plus/total_games*100:.1f}%)")
    
    # Show AI algorithm details
    print(f"\n🤖 NEW AI ALGORITHM DETAILS:")
    print("• Expectiminimax with Snake Heuristic (depth 2)")
    print("• Snake Pattern Evaluation: Perfect snake pattern weights")
    print("• Strategic Features:")
    print("  - Snake heuristic with exponential weights")
    print("  - Expectiminimax for stochastic tile placement")
    print("  - Depth 2 lookahead for balanced performance")
    print("  - Optimized for snake-like tile arrangement")
    print("• Target: High win rate with snake pattern strategy")
    print("• Performance tracking: AI latency, game timing, comprehensive stats")

if __name__ == "__main__":
    main()
