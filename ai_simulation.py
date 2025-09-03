#!/usr/bin/env python3
"""
AI Simulation for 2048 Game
Tests the AI's ability to win the game by running automated games in async mode.

Usage:
    python ai_simulation.py [number_of_games]
    
Examples:
    python ai_simulation.py          # Run 50 games (default)
    python ai_simulation.py 100     # Run 100 games
    python ai_simulation.py --help  # Show help
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


class GameStats:
    """Handles game statistics tracking and management."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics to initial values."""
        self.stats = {
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
            'all_max_tiles': [],
            'max_tile_distribution': {
                '2-4': 0, '8-16': 0, '32-64': 0, '128-256': 0,
                '512': 0, '1024': 0, '2048': 0, '4096+': 0
            }
        }
    
    def categorize_max_tile(self, max_tile: int) -> str:
        """Categorize max tile value into distribution buckets."""
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
    
    def update_with_result(self, result: Dict):
        """Update statistics with a single game result."""
        self.stats['games_played'] += 1
        self.stats['total_moves'] += result['moves']
        
        # Update max tile distribution
        max_tile = result['max_tile']
        self.stats['all_max_tiles'].append(max_tile)
        category = self.categorize_max_tile(max_tile)
        self.stats['max_tile_distribution'][category] += 1
        
        # Update timing statistics
        if 'ai_latencies' in result and result['ai_latencies']:
            for latency in result['ai_latencies']:
                self.stats['total_ai_latency'] += latency
                self.stats['ai_suggestions_count'] += 1
                self.stats['min_ai_latency'] = min(self.stats['min_ai_latency'], latency)
                self.stats['max_ai_latency'] = max(self.stats['max_ai_latency'], latency)
        
        if 'game_time' in result:
            game_time = result['game_time']
            self.stats['total_game_time'] += game_time
            self.stats['min_game_time'] = min(self.stats['min_game_time'], game_time)
            self.stats['max_game_time'] = max(self.stats['max_game_time'], game_time)
        
        # Update win/loss counts
        if result['status'] == 'win':
            self.stats['games_won'] += 1
        elif result['status'] == 'lose':
            self.stats['games_lost'] += 1
        
        # Update max values
        if result['score'] > self.stats['max_score']:
            self.stats['max_score'] = result['score']
        if result['max_tile'] > self.stats['max_tile']:
            self.stats['max_tile'] = result['max_tile']
        
        # Update win rate
        if self.stats['games_played'] > 0:
            self.stats['win_rate'] = (self.stats['games_won'] / self.stats['games_played']) * 100


class GameRunner:
    """Handles running individual games and batch processing."""
    
    def __init__(self, stats: GameStats):
        self.stats = stats
    
    def run_single_game(self, max_moves: int = 10000) -> Dict:
        """Run a single AI-controlled game and return detailed statistics."""
        # Initialize game
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
        
        while moves_made < max_moves:
            # Check game state
            game_status = game.get_game_state(board)
            
            if game_status in ['win', 'lose']:
                game_end_time = time.time()
                return {
                    'status': game_status,
                    'moves': moves_made,
                    'score': score,
                    'max_tile': max_tile_achieved,
                    'final_board': [row[:] for row in board],
                    'game_time': game_end_time - game_start_time,
                    'ai_latencies': ai_latencies,
                    'ai_suggestions_count': ai_suggestions_count
                }
            
            # Get AI suggestion with timing
            ai_start_time = time.time()
            ai_move = game.get_ai_suggestion(board)
            ai_end_time = time.time()
            
            ai_latency = ai_end_time - ai_start_time
            ai_latencies.append(ai_latency)
            ai_suggestions_count += 1
            
            if ai_move is None:
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
            
            # Execute the move
            move_functions = {
                'up': game.move_up, 'down': game.move_down,
                'left': game.move_left, 'right': game.move_right,
            }
            
            new_board, score_gain, has_moved = move_functions[ai_move](board)
            
            if has_moved:
                board = new_board
                score += score_gain
                moves_made += 1
                consecutive_no_progress = 0
                
                # Add new random tile
                game.add_random_tile(board)
                
                # Track max tile
                for row in board:
                    for cell in row:
                        if cell is not None and cell > max_tile_achieved:
                            max_tile_achieved = cell
            else:
                consecutive_no_progress += 1
                if consecutive_no_progress > 10:
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
    
    def run_multiple_games_async(self, num_games: int, max_workers: int = None, max_moves: int = 5000) -> Dict:
        """Run multiple AI games in parallel using multiprocessing."""
        if max_workers is None:
            max_workers = min(cpu_count(), 5)
        
        print(f"🚀 Running AI simulation for {num_games} games in batches of {max_workers}...")
        print(f"⚡ Using {max_workers} CPU cores per batch")
        print("=" * 60)
        
        all_results = []
        total_execution_time = 0
        
        # Process games in batches
        for batch_start in range(0, num_games, max_workers):
            batch_end = min(batch_start + max_workers, num_games)
            batch_size = batch_end - batch_start
            batch_number = batch_start // max_workers + 1
            
            print(f"\n📦 Processing batch {batch_number}: Games {batch_start + 1}-{batch_end}")
            
            start_time = time.time()
            
            # Run current batch in parallel
            with Pool(processes=batch_size) as pool:
                game_args = [(game_num, max_moves) for game_num in range(batch_start + 1, batch_end + 1)]
                batch_results = pool.map(self._run_single_game_wrapper, game_args)
            
            end_time = time.time()
            batch_execution_time = end_time - start_time
            total_execution_time += batch_execution_time
            
            # Process batch results
            self._process_batch_results(batch_results, batch_execution_time, batch_number)
            all_results.extend(batch_results)
        
        return {
            'results': all_results,
            'stats': self.stats.stats,
            'execution_time': total_execution_time
        }
    
    def _process_batch_results(self, results: List[Dict], execution_time: float, batch_num: int) -> None:
        """Process and display results from a batch of parallel game simulations."""
        print(f"📊 Processing {len(results)} game results from batch...")
        
        # Sort results by game number
        results.sort(key=lambda x: x['game_num'])
        
        # Create batch-specific stats for logging
        batch_stats = GameStats()
        
        # Process each result
        for result in results:
            game_num = result['game_num']
            self.stats.update_with_result(result)  # Update cumulative stats
            batch_stats.update_with_result(result)  # Update batch-specific stats
            
            # Display result
            avg_ai_latency = sum(result.get('ai_latencies', [0])) / max(len(result.get('ai_latencies', [1])), 1)
            status_icon = "✅" if result['status'] == 'win' else "❌" if result['status'] == 'lose' else "⏰"
            status_text = "WON!" if result['status'] == 'win' else "LOST!" if result['status'] == 'lose' else "TIMEOUT"
            
            print(f"{status_icon} Game {game_num:2d}: {status_text:<7} Score: {result['score']:6,}, "
                  f"Moves: {result['moves']:4d}, Max Tile: {result['max_tile']}, "
                  f"Time: {result.get('game_time', 0):.2f}s, Avg AI: {avg_ai_latency:.3f}s")
        
        print(f"⏱️  Batch execution time: {execution_time:.2f} seconds")
        print(f"🎯 Batch speed: {len(results) / execution_time:.1f} games/second")
        
        # Log batch-specific results to CSV
        if batch_stats.stats['games_played'] > 0:
            session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_batch{batch_num}"
            command_args = f"batch={batch_num}, games_in_batch={len(results)}"
            CSVLogger.log_to_csv(batch_stats.stats, session_id, command_args)
            print(f"📝 Batch {batch_num} results logged to CSV (batch-specific stats)")
    
    @staticmethod
    def _run_single_game_wrapper(args_tuple) -> Dict:
        """Wrapper function for multiprocessing to run a single game simulation."""
        game_num, max_moves = args_tuple
        stats = GameStats()
        runner = GameRunner(stats)
        result = runner.run_single_game(max_moves=max_moves)
        result['game_num'] = game_num
        return result


class CSVLogger:
    """Handles CSV logging functionality."""
    
    @staticmethod
    def log_to_csv(stats: Dict, session_id: str, command_args: str = ""):
        """Log simulation results to CSV format."""
        os.makedirs("simulation_logs", exist_ok=True)
        csv_path = "simulation_logs/simulation_results.csv"
        file_exists = os.path.exists(csv_path)
        
        # Calculate statistics
        total_games = stats['games_played']
        games_won = stats['games_won']
        games_lost = stats['games_lost']
        win_rate = (games_won / total_games * 100) if total_games > 0 else 0
        
        # Count max tile distribution
        max_tile_dist = stats['max_tile_distribution']
        games_2 = max_tile_dist.get('2-4', 0)
        games_8 = max_tile_dist.get('8-16', 0)
        games_32 = max_tile_dist.get('32-64', 0)
        games_128 = max_tile_dist.get('128-256', 0)
        games_512 = max_tile_dist.get('512', 0)
        games_1024 = max_tile_dist.get('1024', 0)
        games_2048 = max_tile_dist.get('2048', 0)
        
        # Calculate averages
        avg_moves = stats['total_moves'] / total_games if total_games > 0 else 0
        avg_ai_latency = stats['total_ai_latency'] / stats['ai_suggestions_count'] if stats['ai_suggestions_count'] > 0 else 0
        avg_game_time = stats['total_game_time'] / total_games if total_games > 0 else 0
        
        # Prepare row data
        row_data = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # timestamp
            session_id,                                    # session_id
            total_games,                                   # total_games
            games_won,                                     # games_won
            games_lost,                                    # games_lost
            round(win_rate, 1),                           # win_rate_percent
            stats['max_score'],                           # max_score
            stats['max_tile'],                            # max_tile
            stats['total_moves'],                         # total_moves
            round(avg_moves, 1),                          # avg_moves_per_game
            round(avg_ai_latency, 3),                     # avg_ai_latency
            round(avg_game_time, 2),                      # avg_game_time
            stats['ai_suggestions_count'],                # total_ai_suggestions
            games_2,                                      # games_2
            games_8,                                      # games_8
            games_32,                                     # games_32
            games_128,                                    # games_128
            games_512,                                    # games_512
            games_1024,                                   # games_1024
            games_2048,                                   # games_2048
            command_args                                  # command_args
        ]
        
        # Write to CSV
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write headers if file is new
            if not file_exists:
                headers = [
                    'timestamp', 'session_id', 'total_games', 'games_won', 'games_lost',
                    'win_rate_percent', 'max_score', 'max_tile', 'total_moves',
                    'avg_moves_per_game', 'avg_ai_latency', 'avg_game_time',
                    'total_ai_suggestions',
                    'games_2', 'games_8', 'games_32', 'games_128', 'games_512',
                    'games_1024', 'games_2048', 'command_args'
                ]
                writer.writerow(headers)
            
            # Write data row
            writer.writerow(row_data)
        
        print(f"📝 Results logged to: {csv_path}")


class PlotGenerator:
    """Handles plot generation functionality."""
    
    @staticmethod
    def plot_max_tile_distribution(stats: Dict, save_plot: bool = True, show_plot: bool = True) -> None:
        """Plot the distribution of maximum tiles achieved across all games."""
        # Check if we have data
        has_data = stats.get('all_max_tiles') or any(stats.get('max_tile_distribution', {}).values())
        if not has_data:
            print("⚠️  No max tile data available for plotting")
            return
        
        # Create single bar plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle('2048 AI Simulation - Max Tile Distribution', fontsize=16, fontweight='bold')
        
        # Define tile categories up to 2048 only
        tile_categories = ['2-4', '8-16', '32-64', '128-256', '512', '1024', '2048']
        tile_colors = ['black'] * len(tile_categories)  # All bars black
        
        # Get counts for each category
        max_tile_dist = stats['max_tile_distribution']
        counts = [max_tile_dist.get(category, 0) for category in tile_categories]
        
        # Create bar plot
        bars = ax.bar(tile_categories, counts, color=tile_colors, edgecolor='black', linewidth=1, alpha=0.8)
        
        # Customize the plot
        ax.set_xlabel('Maximum Tile Achieved', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Games', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on top of bars
        total_games = stats['games_played']
        for bar, count in zip(bars, counts):
            if count > 0:
                # Add count label
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       str(count), ha='center', va='bottom', fontweight='bold')
                
                # Add percentage label
                percentage = (count / total_games) * 100
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                       f'{percentage:.1f}%', ha='center', va='center', 
                       fontweight='bold', color='white', fontsize=10)
        
        # Customize grid and layout
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Add summary statistics
        win_rate = (stats['games_won'] / total_games * 100) if total_games > 0 else 0
        stats_text = f'Total Games: {total_games}\nWin Rate: {win_rate:.1f}%'
        fig.text(0.02, 0.02, stats_text, fontsize=10, 
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Save plot if requested
        if save_plot:
            filename = f"simulation_logs/max_tile_distribution_{int(time.time())}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to: {filename}")
        
        # Show plot if requested
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
            # Read the data
            df = pd.read_csv(csv_path)
            
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
            stats = {
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
            print(f"  Total Games: {stats['games_played']}")
            print(f"  Games Won: {stats['games_won']}")
            print(f"  Win Rate: {(stats['games_won'] / stats['games_played'] * 100):.1f}%")
            print(f"  Max Score: {stats['max_score']}")
            
            # Use the existing plot function
            return PlotGenerator.plot_max_tile_distribution(stats, save_plot=True, show_plot=True)
            
        except Exception as e:
            print(f"❌ Error creating plot from CSV: {e}")
            return None


class StatisticsDisplay:
    """Handles displaying simulation statistics."""
    
    @staticmethod
    def print_final_statistics(stats: Dict, execution_time: float = None) -> None:
        """Print comprehensive simulation statistics."""
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
            percentiles = StatisticsDisplay._calculate_max_tile_percentiles(stats['all_max_tiles'])
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
    
    @staticmethod
    def _calculate_max_tile_percentiles(max_tiles):
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
    
    @staticmethod
    def print_board(board: List[List]) -> None:
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
  python ai_simulation.py --create-plot    # Create plot from existing CSV
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
    
    parser.add_argument(
        '--create-plot',
        action='store_true',
        help='Create bar plot from existing CSV data'
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
    
    # Handle special commands
    if hasattr(args, 'create_plot') and args.create_plot:
        print("📊 Creating plot from existing CSV...")
        PlotGenerator.create_plot_from_csv()
        return
    
    print("🎮 2048 AI Simulation - New AI Integration")
    print("Testing the new expectiminimax AI with snake heuristic...")
    print("Using game_logic.py backend with expectiminimax algorithm and snake pattern evaluation")
    print(f"📊 Configuration: {args.games} games, max {args.max_moves} moves per game")
    
    if args.max_workers:
        print(f"🔧 Custom max workers: {args.max_workers}")
    else:
        print(f"🔧 Auto-detected workers: {min(cpu_count(), 5)}")
    
    # Create simulation components
    stats = GameStats()
    runner = GameRunner(stats)
    
    # Run simulation
    results = runner.run_multiple_games_async(
        num_games=args.games,
        max_workers=args.max_workers,
        max_moves=args.max_moves
    )
    
    # Print results
    StatisticsDisplay.print_final_statistics(results['stats'], results['execution_time'])
    
    # Show best game details
    best_game = max(results['results'], key=lambda x: x['score'])
    print(f"\n🏅 BEST GAME DETAILS:")
    print(f"Status: {best_game['status']}")
    print(f"Score: {best_game['score']:,}")
    print(f"Max Tile: {best_game['max_tile']}")
    print(f"Moves: {best_game['moves']}")
    
    if best_game['status'] == 'win':
        print("\n🎯 WINNING BOARD:")
        StatisticsDisplay.print_board(best_game['final_board'])
    
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
    
    # Log final results to CSV
    session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    command_args = f"games={args.games}, max_moves={args.max_moves}"
    CSVLogger.log_to_csv(results['stats'], session_id, command_args)
    
    # Generate plots
    print(f"\n📊 GENERATING VISUALIZATION PLOTS...")
    try:
        PlotGenerator.plot_max_tile_distribution(results['stats'], save_plot=True, show_plot=False)
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