#!/usr/bin/env python3
"""
AI Simulation for 2048 Game
Tests the AI's ability to win the game by running automated games in async mode.

Usage:
    python ai_simulation.py [number_of_games]
    
Examples:
    python ai_simulation.py          # Run 50 games (default)
    python ai_simulation.py 100      # Run 100 games
    python ai_simulation.py --help   # Show help
"""

import game_logic as game
import time
import argparse
from typing import List, Dict
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
from datetime import datetime

# Configuration constants for simulation parameters
DEFAULT_GAMES = 50
DEFAULT_MAX_MOVES = 5000
DEFAULT_MAX_WORKERS = 5
# Tile value categories for distribution analysis
TILE_CATEGORIES = ['2-4', '8-16', '32-64', '128-256', '512', '1024', '2048', '4096+']

class AISimulator:
    """Unified simulator class handling all simulation functionality."""
    
    def __init__(self):
        self.reset_stats()
    
    def reset_stats(self):
        """Reset all statistics to initial values."""
        # Initialize comprehensive statistics tracking dictionary
        self.stats = {
            'games_played': 0, 'games_won': 0, 'games_lost': 0, 'max_score': 0, 'max_tile': 0,
            'total_moves': 0, 'win_rate': 0.0, 'total_ai_latency': 0.0, 'ai_suggestions_count': 0,
            'total_game_time': 0.0, 'min_ai_latency': float('inf'), 'max_ai_latency': 0.0,
            'min_game_time': float('inf'), 'max_game_time': 0.0, 'all_max_tiles': [],
            'max_tile_distribution': {cat: 0 for cat in TILE_CATEGORIES}
        }
    
    def categorize_max_tile(self, max_tile: int) -> str:
        """Categorize max tile value into distribution buckets."""
        # Map tile values to predefined categories for analysis
        if max_tile <= 4: return '2-4'
        elif max_tile <= 16: return '8-16'
        elif max_tile <= 64: return '32-64'
        elif max_tile <= 256: return '128-256'
        elif max_tile == 512: return '512'
        elif max_tile == 1024: return '1024'
        elif max_tile == 2048: return '2048'
        else: return '4096+'
    
    def update_stats(self, result: Dict):
        """Update statistics with a single game result."""
        # Increment basic counters
        self.stats['games_played'] += 1
        self.stats['total_moves'] += result['moves']
        
        # Update max tile distribution tracking
        max_tile = result['max_tile']
        self.stats['all_max_tiles'].append(max_tile)
        category = self.categorize_max_tile(max_tile)
        self.stats['max_tile_distribution'][category] += 1
        
        # Update AI performance timing statistics
        if 'ai_latencies' in result and result['ai_latencies']:
            for latency in result['ai_latencies']:
                self.stats['total_ai_latency'] += latency
                self.stats['ai_suggestions_count'] += 1
                self.stats['min_ai_latency'] = min(self.stats['min_ai_latency'], latency)
                self.stats['max_ai_latency'] = max(self.stats['max_ai_latency'], latency)
        
        # Update game timing statistics
        if 'game_time' in result:
            game_time = result['game_time']
            self.stats['total_game_time'] += game_time
            self.stats['min_game_time'] = min(self.stats['min_game_time'], game_time)
            self.stats['max_game_time'] = max(self.stats['max_game_time'], game_time)
        
        # Update win/loss outcome tracking
        if result['status'] == 'win':
            self.stats['games_won'] += 1
        elif result['status'] == 'lose':
            self.stats['games_lost'] += 1
        
        # Update maximum achievement values
        self.stats['max_score'] = max(self.stats['max_score'], result['score'])
        self.stats['max_tile'] = max(self.stats['max_tile'], result['max_tile'])
        
        # Recalculate win rate percentage
        if self.stats['games_played'] > 0:
            self.stats['win_rate'] = (self.stats['games_won'] / self.stats['games_played']) * 100
    
    def run_single_game(self, max_moves: int = DEFAULT_MAX_MOVES) -> Dict:
        """Run a single AI-controlled game and return detailed statistics."""
        # Initialize new game with two starting tiles
        board = game.new_game()
        game.add_random_tile(board)
        game.add_random_tile(board)
        
        # Game state tracking variables
        moves_made = 0
        score = 0
        max_tile_achieved = 0
        consecutive_no_progress = 0
        
        # Performance timing metrics
        game_start_time = time.time()
        ai_latencies = []
        
        # Define move functions for efficient lookup
        move_functions = {
            'up': game.move_up, 'down': game.move_down,
            'left': game.move_left, 'right': game.move_right,
        }
        
        while moves_made < max_moves:
            # Check current game state for win/lose conditions
            game_status = game.get_game_state(board)
            
            # Return results if game is finished
            if game_status in ['win', 'lose']:
                return {
                    'status': game_status, 'moves': moves_made, 'score': score, 'max_tile': max_tile_achieved,
                    'final_board': [row[:] for row in board], 'game_time': time.time() - game_start_time,
                    'ai_latencies': ai_latencies, 'ai_suggestions_count': len(ai_latencies)
                }
            
            # Get AI move suggestion with performance timing
            ai_start_time = time.time()
            ai_move = game.get_ai_suggestion(board)
            ai_latencies.append(time.time() - ai_start_time)
            
            # Handle case where AI has no valid moves
            if ai_move is None:
                return {
                    'status': 'lose', 'moves': moves_made, 'score': score, 'max_tile': max_tile_achieved,
                    'final_board': [row[:] for row in board], 'game_time': time.time() - game_start_time,
                    'ai_latencies': ai_latencies, 'ai_suggestions_count': len(ai_latencies)
                }
            
            # Execute the AI-suggested move
            new_board, score_gain, has_moved = move_functions[ai_move](board)
            
            # Update game state if move was successful
            if has_moved:
                board = new_board
                score += score_gain
                moves_made += 1
                consecutive_no_progress = 0
                
                # Add new random tile after successful move
                game.add_random_tile(board)
                
                # Track highest tile achieved
                for row in board:
                    for cell in row:
                        if cell is not None and cell > max_tile_achieved:
                            max_tile_achieved = cell
            else:
                # Track consecutive failed moves to prevent infinite loops
                consecutive_no_progress += 1
                if consecutive_no_progress > 10:
                    break
        
        # Game ended due to max moves reached or no progress made
        return {
            'status': 'timeout', 'moves': moves_made, 'score': score, 'max_tile': max_tile_achieved,
            'final_board': [row[:] for row in board], 'game_time': time.time() - game_start_time,
            'ai_latencies': ai_latencies, 'ai_suggestions_count': len(ai_latencies)
        }
    
    def run_parallel_games(self, num_games: int, max_workers: int = None, max_moves: int = DEFAULT_MAX_MOVES) -> Dict:
        """Run multiple AI games in parallel using multiprocessing."""
        # Auto-detect optimal worker count if not specified
        if max_workers is None:
            max_workers = min(cpu_count(), DEFAULT_MAX_WORKERS)
        
        print(f"🚀 Running AI simulation for {num_games} games in batches of {max_workers}...")
        print(f"⚡ Using {max_workers} CPU cores per batch")
        print("=" * 60)
        
        all_results = []
        total_execution_time = 0
        
        # Process games in parallel batches for optimal performance
        for batch_start in range(0, num_games, max_workers):
            batch_end = min(batch_start + max_workers, num_games)
            batch_size = batch_end - batch_start
            batch_number = batch_start // max_workers + 1
            
            print(f"\n📦 Processing batch {batch_number}: Games {batch_start + 1}-{batch_end}")
            
            start_time = time.time()
            
            # Execute current batch using multiprocessing pool
            with Pool(processes=batch_size) as pool:
                game_args = [(game_num, max_moves) for game_num in range(batch_start + 1, batch_end + 1)]
                batch_results = pool.map(self._run_game_wrapper, game_args)
            
            batch_time = time.time() - start_time
            total_execution_time += batch_time
            
            # Process and display batch results
            self._process_batch_results(batch_results, batch_time, batch_number)
            all_results.extend(batch_results)
        
        return {'results': all_results, 'stats': self.stats, 'execution_time': total_execution_time}
    
    def _process_batch_results(self, results: List[Dict], execution_time: float, batch_num: int) -> None:
        """Process and display results from a batch of parallel game simulations."""
        print(f"📊 Processing {len(results)} game results from batch...")
        
        # Sort results by game number for consistent display
        results.sort(key=lambda x: x['game_num'])
        
        # Process each game result in the batch
        for result in results:
            game_num = result['game_num']
            self.update_stats(result)  # Update cumulative stats
            
            # Calculate and display individual game results
            avg_ai_latency = sum(result.get('ai_latencies', [0])) / max(len(result.get('ai_latencies', [1])), 1)
            status_icon = "✅" if result['status'] == 'win' else "❌" if result['status'] == 'lose' else "⏰"
            status_text = "WON!" if result['status'] == 'win' else "LOST!" if result['status'] == 'lose' else "TIMEOUT"
            
            print(f"{status_icon} Game {game_num:2d}: {status_text:<7} Score: {result['score']:6,}, "
                  f"Moves: {result['moves']:4d}, Max Tile: {result['max_tile']}, "
                  f"Time: {result.get('game_time', 0):.2f}s, Avg AI: {avg_ai_latency:.3f}s")
        
        print(f"⏱️  Batch execution time: {execution_time:.2f} seconds")
        print(f"🎯 Batch speed: {len(results) / execution_time * 60:.1f} games/minute")
        
        # Log batch-specific results to CSV for detailed analysis
        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_batch{batch_num}"
        command_args = f"batch={batch_num}, games_in_batch={len(results)}"
        self.log_to_csv(session_id, command_args)
        print(f"📝 Batch {batch_num} results logged to CSV")
    
    @staticmethod
    def _run_game_wrapper(args_tuple) -> Dict:
        """Wrapper function for multiprocessing to run a single game simulation."""
        # Extract arguments and create isolated game runner for multiprocessing
        game_num, max_moves = args_tuple
        simulator = AISimulator()
        result = simulator.run_single_game(max_moves=max_moves)
        result['game_num'] = game_num
        return result
    
    def log_to_csv(self, session_id: str, command_args: str = ""):
        """Log simulation results to CSV format."""
        # Create logs directory if it doesn't exist
        os.makedirs("simulation_logs", exist_ok=True)
        csv_path = "simulation_logs/simulation_results.csv"
        file_exists = os.path.exists(csv_path)
        
        # Extract basic game statistics
        total_games = self.stats['games_played']
        games_won = self.stats['games_won']
        games_lost = self.stats['games_lost']
        win_rate = (games_won / total_games * 100) if total_games > 0 else 0
        
        # Extract max tile distribution counts
        max_tile_dist = self.stats['max_tile_distribution']
        
        # Calculate performance averages
        avg_moves = self.stats['total_moves'] / total_games if total_games > 0 else 0
        avg_ai_latency = self.stats['total_ai_latency'] / self.stats['ai_suggestions_count'] if self.stats['ai_suggestions_count'] > 0 else 0
        avg_game_time = self.stats['total_game_time'] / total_games if total_games > 0 else 0
        
        # Prepare CSV row data with all statistics
        row_data = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), session_id, total_games, games_won, games_lost,
            round(win_rate, 1), self.stats['max_score'], self.stats['max_tile'], self.stats['total_moves'],
            round(avg_moves, 1), round(avg_ai_latency, 3), round(avg_game_time, 2),
            self.stats['ai_suggestions_count'],
            max_tile_dist.get('2-4', 0), max_tile_dist.get('8-16', 0), max_tile_dist.get('32-64', 0),
            max_tile_dist.get('128-256', 0), max_tile_dist.get('512', 0), max_tile_dist.get('1024', 0),
            max_tile_dist.get('2048', 0), command_args
        ]
        
        # Write data to CSV file
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write column headers if creating new file
            if not file_exists:
                headers = [
                    'timestamp', 'session_id', 'total_games', 'games_won', 'games_lost',
                    'win_rate_percent', 'max_score', 'max_tile', 'total_moves',
                    'avg_moves_per_game', 'avg_ai_latency', 'avg_game_time',
                    'total_ai_suggestions', 'games_2', 'games_8', 'games_32',
                    'games_128', 'games_512', 'games_1024', 'games_2048', 'command_args'
                ]
                writer.writerow(headers)
            
            writer.writerow(row_data)
        
        print(f"📝 Results logged to: {csv_path}")
    
    def plot_distribution(self, save_plot: bool = True, show_plot: bool = True):
        """Plot the distribution of maximum tiles achieved across all games."""
        # Validate that we have data to plot
        has_data = self.stats.get('all_max_tiles') or any(self.stats.get('max_tile_distribution', {}).values())
        if not has_data:
            print("⚠️  No max tile data available for plotting")
            return
        
        # Create matplotlib figure and axis for bar plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle('2048 AI Simulation - Max Tile Distribution', fontsize=16, fontweight='bold')
        
        # Define tile categories and styling for visualization
        tile_categories = ['2-4', '8-16', '32-64', '128-256', '512', '1024', '2048']
        
        # Extract counts for each tile category
        max_tile_dist = self.stats['max_tile_distribution']
        counts = [max_tile_dist.get(category, 0) for category in tile_categories]
        
        # Create bar chart with custom styling
        bars = ax.bar(tile_categories, counts, color='black', edgecolor='black', linewidth=1, alpha=0.8)
        
        # Configure plot appearance and labels
        ax.set_xlabel('Maximum Tile Achieved', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Games', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on top of bars for better readability
        total_games = self.stats['games_played']
        for bar, count in zip(bars, counts):
            if count > 0:
                # Add count label above each bar
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       str(count), ha='center', va='bottom', fontweight='bold')
                
                # Add percentage label inside each bar
                percentage = (count / total_games) * 100
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                       f'{percentage:.1f}%', ha='center', va='center', 
                       fontweight='bold', color='white', fontsize=10)
        
        # Apply grid and layout formatting
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Add summary statistics box to the plot
        win_rate = (self.stats['games_won'] / total_games * 100) if total_games > 0 else 0
        stats_text = f'Total Games: {total_games}\nWin Rate: {win_rate:.1f}%'
        fig.text(0.02, 0.02, stats_text, fontsize=10, 
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Save plot to file if requested
        if save_plot:
            filename = f"simulation_logs/max_tile_distribution_{int(time.time())}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to: {filename}")
        
        # Display plot window if requested
        if show_plot:
            plt.show()
    
    @staticmethod
    def create_plot_from_csv():
        """Create a bar plot from all results in the CSV file."""
        csv_path = "simulation_logs/simulation_results.csv"
        if not os.path.exists(csv_path):
            print(f"❌ CSV file not found: {csv_path}")
            return
        
        try:
            # Read the data with error handling for malformed rows
            df = pd.read_csv(csv_path, on_bad_lines='skip')
            
            if df.empty:
                print("❌ CSV file is empty")
                return
            
            # Filter to only batch-specific rows (exclude final summary rows)
            batch_rows = df[df['session_id'].str.contains('_batch', na=False)]
            
            if batch_rows.empty:
                print("❌ No batch-specific data found in CSV")
                return
            
            print(f"📊 Found {len(batch_rows)} batch sessions in CSV")
            print(f"📈 Total games across all batches: {batch_rows['total_games'].sum()}")
            
            # Create aggregated stats dict from batch-specific data
            simulator = AISimulator()
            simulator.stats = {
                'games_played': batch_rows['total_games'].sum(),
                'games_won': batch_rows['games_won'].sum(),
                'max_score': batch_rows['max_score'].max(),
                'all_max_tiles': [],  # Not used in CSV-based plotting, but required by function
                'max_tile_distribution': {
                    '2-4': batch_rows['games_2'].sum() if 'games_2' in batch_rows.columns else 0,
                    '8-16': batch_rows['games_8'].sum() if 'games_8' in batch_rows.columns else 0,
                    '32-64': batch_rows['games_32'].sum() if 'games_32' in batch_rows.columns else 0,
                    '128-256': batch_rows['games_128'].sum() if 'games_128' in batch_rows.columns else 0,
                    '512': batch_rows['games_512'].sum() if 'games_512' in batch_rows.columns else 0,
                    '1024': batch_rows['games_1024'].sum() if 'games_1024' in batch_rows.columns else 0,
                    '2048': batch_rows['games_2048'].sum() if 'games_2048' in batch_rows.columns else 0,
                }
            }
            
            print(f"📊 Aggregated statistics:")
            print(f"  Total Games: {simulator.stats['games_played']}")
            print(f"  Games Won: {simulator.stats['games_won']}")
            print(f"  Win Rate: {(simulator.stats['games_won'] / simulator.stats['games_played'] * 100):.1f}%")
            print(f"  Max Score: {simulator.stats['max_score']}")
            
            # Use the existing plot function
            return simulator.plot_distribution(save_plot=True, show_plot=True)
            
        except Exception as e:
            print(f"❌ Error creating plot from CSV: {e}")
            return None
    
    def print_statistics(self, execution_time: float = None):
        """Print comprehensive simulation statistics."""
        print("\n" + "=" * 80)
        print("🏆 COMPREHENSIVE AI SIMULATION RESULTS")
        print("=" * 80)
        
        # Display basic game performance metrics
        print("📊 GAME PERFORMANCE:")
        print(f"  Games Played: {self.stats['games_played']}")
        print(f"  Games Won: {self.stats['games_won']}")
        print(f"  Games Lost: {self.stats['games_lost']}")
        print(f"  Win Rate: {self.stats['win_rate']:.1f}%")
        print(f"  Highest Score: {self.stats['max_score']:,}")
        print(f"  Highest Tile: {self.stats['max_tile']}")
        print(f"  Average Moves per Game: {self.stats['total_moves'] / self.stats['games_played']:.1f}")
        
        # Display detailed max tile distribution analysis
        print("\n📊 MAX TILE DISTRIBUTION ANALYSIS:")
        if self.stats['all_max_tiles']:
            # Calculate and display percentile statistics
            percentiles = self._calculate_percentiles(self.stats['all_max_tiles'])
            print(f"  Max Tile Percentiles:")
            for k, v in percentiles.items():
                print(f"    {k}: {v}")
            
            # Display visual distribution chart
            print(f"  Max Tile Distribution by Range:")
            total_games = self.stats['games_played']
            for range_name, count in self.stats['max_tile_distribution'].items():
                percentage = (count / total_games) * 100 if total_games > 0 else 0
                bar_length = int(percentage / 2)  # Scale for visual representation
                bar = "█" * bar_length + "░" * (25 - bar_length)
                print(f"    {range_name:>12}: {count:3d} games ({percentage:5.1f}%) {bar}")
            
            # Calculate and display success rate metrics
            games_2048 = self.stats['max_tile_distribution']['2048']
            games_1024_plus = sum(self.stats['max_tile_distribution'][k] for k in ['1024', '2048', '4096+'])
            games_512_plus = sum(self.stats['max_tile_distribution'][k] for k in ['512', '1024', '2048', '4096+'])
            
            print(f"\n  🎯 SUCCESS RATE ANALYSIS:")
            print(f"    Games reaching 2048: {games_2048} ({games_2048/total_games*100:.1f}%)")
            print(f"    Games reaching 1024+: {games_1024_plus} ({games_1024_plus/total_games*100:.1f}%)")
            print(f"    Games reaching 512+: {games_512_plus} ({games_512_plus/total_games*100:.1f}%)")
        
        # Display AI performance and timing metrics
        print("\n🤖 AI PERFORMANCE METRICS:")
        if self.stats['ai_suggestions_count'] > 0:
            avg_ai_latency = self.stats['total_ai_latency'] / self.stats['ai_suggestions_count']
            print(f"  Total AI Suggestions: {self.stats['ai_suggestions_count']:,}")
            print(f"  Average AI Latency: {avg_ai_latency:.3f} seconds")
            print(f"  Min AI Latency: {self.stats['min_ai_latency']:.3f} seconds")
            print(f"  Max AI Latency: {self.stats['max_ai_latency']:.3f} seconds")
            print(f"  AI Suggestions per Game: {self.stats['ai_suggestions_count'] / self.stats['games_played']:.1f}")
        
        # Display game timing and execution statistics
        print("\n⏱️  GAME TIMING STATISTICS:")
        if self.stats['games_played'] > 0:
            avg_game_time = self.stats['total_game_time'] / self.stats['games_played']
            print(f"  Average Game Time: {avg_game_time:.2f} seconds")
            print(f"  Min Game Time: {self.stats['min_game_time']:.2f} seconds")
            print(f"  Max Game Time: {self.stats['max_game_time']:.2f} seconds")
            print(f"  Total Simulation Time: {self.stats['total_game_time']:.2f} seconds")
        
        # Display parallel execution performance metrics
        if execution_time:
            print("\n🚀 EXECUTION PERFORMANCE:")
            print(f"  Total Execution Time: {execution_time:.2f} seconds")
            print(f"  Games per Minute: {self.stats['games_played'] / execution_time * 60:.1f}")
            print(f"  Parallel Efficiency: {(self.stats['total_game_time'] / execution_time) * 100:.1f}%")
        
        # Display performance analysis with color-coded ratings
        print("\n📈 PERFORMANCE ANALYSIS:")
        if self.stats['ai_suggestions_count'] > 0:
            avg_ai_latency = self.stats['total_ai_latency'] / self.stats['ai_suggestions_count']
            # Rate AI response time performance
            if avg_ai_latency < 0.5:
                print("  🟢 AI Response Time: EXCELLENT (< 0.5s)")
            elif avg_ai_latency < 1.0:
                print("  🟡 AI Response Time: GOOD (0.5-1.0s)")
            elif avg_ai_latency < 2.0:
                print("  🟠 AI Response Time: ACCEPTABLE (1.0-2.0s)")
            else:
                print("  🔴 AI Response Time: SLOW (> 2.0s)")
        
        # Rate win rate performance
        if self.stats['win_rate'] >= 95:
            print("  🟢 Win Rate: EXCELLENT (≥95%)")
        elif self.stats['win_rate'] >= 80:
            print("  🟡 Win Rate: GOOD (80-95%)")
        elif self.stats['win_rate'] >= 60:
            print("  🟠 Win Rate: ACCEPTABLE (60-80%)")
        else:
            print("  🔴 Win Rate: NEEDS IMPROVEMENT (<60%)")
        
        # Display final success summary with performance ratings
        print("\n🎯 SUCCESS SUMMARY:")
        if self.stats['games_won'] > 0:
            print(f"  🎉 AI SUCCESSFULLY WON {self.stats['games_won']} GAME(S)!")
            print("  ✅ The AI can consistently achieve the 2048 tile!")
            # Rate overall performance level
            if self.stats['win_rate'] >= 90:
                print("  🏆 WORLD-CLASS PERFORMANCE!")
            elif self.stats['win_rate'] >= 80:
                print("  🥇 EXCELLENT PERFORMANCE!")
            elif self.stats['win_rate'] >= 70:
                print("  🥈 GOOD PERFORMANCE!")
            else:
                print("  🥉 DECENT PERFORMANCE!")
        else:
            print(f"  🤔 AI didn't win any games, but achieved tile {self.stats['max_tile']}")
            print("  📈 The AI might need further tuning or more games to win.")
        
        print("=" * 80)
    
    def _calculate_percentiles(self, max_tiles):
        """Calculate percentile statistics for max tile distribution analysis."""
        if not max_tiles:
            return {}
        
        # Sort tiles and calculate percentile positions
        sorted_tiles = sorted(max_tiles)
        n = len(sorted_tiles)
        
        return {
            'Min': sorted_tiles[0],
            '25th': sorted_tiles[int(n * 0.25)],
            '50th (Median)': sorted_tiles[int(n * 0.5)],
            '75th': sorted_tiles[int(n * 0.75)],
            '90th': sorted_tiles[int(n * 0.90)],
            '95th': sorted_tiles[int(n * 0.95)],
            '99th': sorted_tiles[int(n * 0.99)],
            'Max': sorted_tiles[-1]
        }
    
    def print_board(self, board: List[List]) -> None:
        """Print the game board in a formatted grid layout."""
        # Print formatted board with borders and proper spacing
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

def parse_arguments():
    """Parse command-line arguments for configuring the AI simulation."""
    # Create argument parser with detailed help and examples
    parser = argparse.ArgumentParser(
        description="2048 AI Simulation - Test AI performance with configurable number of games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ai_simulation.py          # Run 50 games (default)
  python ai_simulation.py 100     # Run 100 games
  python ai_simulation.py 25      # Run 25 games
  python ai_simulation.py --create-plot    # Create plot from existing CSV
  python ai_simulation.py --help  # Show this help message
        """
    )
    
    # Define command-line arguments for simulation configuration
    parser.add_argument(
        'games', nargs='?', type=int, default=DEFAULT_GAMES, metavar='GAMES',
        help='Number of games to simulate (default: 50)'
    )
    
    parser.add_argument(
        '--max-workers', type=int,
        help='Maximum number of parallel workers (default: auto-detect)'
    )
    
    parser.add_argument(
        '--max-moves', type=int, default=DEFAULT_MAX_MOVES,
        help='Maximum moves per game before timeout (default: 5000)'
    )
    
    parser.add_argument(
        '--create-plot', action='store_true',
        help='Create bar plot from existing CSV data'
    )
    
    args = parser.parse_args()
    
    # Validate command-line arguments with appropriate error messages
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
    # Parse and validate command-line arguments
    args = parse_arguments()
    
    # Handle special plot generation command
    if hasattr(args, 'create_plot') and args.create_plot:
        print("📊 Creating plot from existing CSV...")
        AISimulator.create_plot_from_csv()
        return
    
    # Display simulation configuration and startup information
    print("🎮 2048 AI Simulation - Optimized Implementation")
    print("Testing the expectiminimax AI with snake heuristic...")
    print(f"📊 Configuration: {args.games} games, max {args.max_moves} moves per game")
    
    if args.max_workers:
        print(f"🔧 Custom max workers: {args.max_workers}")
    else:
        print(f"🔧 Auto-detected workers: {min(cpu_count(), DEFAULT_MAX_WORKERS)}")
    
    # Initialize simulation components
    simulator = AISimulator()
    
    # Execute parallel game simulation
    results = simulator.run_parallel_games(
        num_games=args.games,
        max_workers=args.max_workers,
        max_moves=args.max_moves
    )
    
    # Print comprehensive results
    simulator.print_statistics(results['execution_time'])
    
    # Show best game details
    best_game = max(results['results'], key=lambda x: x['score'])
    print(f"\n🏅 BEST GAME DETAILS:")
    print(f"Status: {best_game['status']}")
    print(f"Score: {best_game['score']:,}")
    print(f"Max Tile: {best_game['max_tile']}")
    print(f"Moves: {best_game['moves']}")
    
    if best_game['status'] == 'win':
        print("\n🎯 WINNING BOARD:")
        simulator.print_board(best_game['final_board'])
    
    # Show enhanced performance summary
    print(f"\n⚡ ENHANCED PERFORMANCE SUMMARY:")
    print(f"Total Games: {results['stats']['games_played']}")
    print(f"Execution Time: {results['execution_time']:.2f} seconds")
    print(f"Speed: {results['stats']['games_played'] / results['execution_time'] * 60:.1f} games/minute")
    print(f"Win Rate: {results['stats']['win_rate']:.1f}%")
    
    if results['stats']['ai_suggestions_count'] > 0:
        avg_ai_latency = results['stats']['total_ai_latency'] / results['stats']['ai_suggestions_count']
        print(f"Average AI Latency: {avg_ai_latency:.3f} seconds")
        print(f"Total AI Suggestions: {results['stats']['ai_suggestions_count']:,}")
    
    if results['stats']['games_played'] > 0:
        avg_game_time = results['stats']['total_game_time'] / results['stats']['games_played']
        print(f"Average Game Time: {avg_game_time:.2f} seconds")
    
    # Log final results to CSV
    session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    command_args = f"games={args.games}, max_moves={args.max_moves}"
    simulator.log_to_csv(session_id, command_args)
    
    # Generate plots
    print(f"\n📊 GENERATING VISUALIZATION PLOTS...")
    try:
        simulator.plot_distribution(save_plot=True, show_plot=False)
        print("✅ Plot generated and saved successfully!")
        print("📁 Check the 'simulation_logs/' directory for the generated plot.")
    except ImportError:
        print("⚠️  Matplotlib not available. Skipping plot generation.")
        print("   Install matplotlib with: pip install matplotlib")
    except Exception as e:
        print(f"⚠️  Error generating plots: {e}")
        print("   Plots will be skipped, but simulation results are still valid.")

if __name__ == "__main__":
    main()