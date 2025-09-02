#!/usr/bin/env python3
"""
Test script for the logging system.
Run this to verify that the CSV logging works correctly.
"""

import os
import sys
import time

# Add current directory to path to import ai_simulation
sys.path.insert(0, '.')

try:
    from ai_simulation import AISimulation
    
    print("🧪 Testing logging system...")
    
    # Create simulation instance
    simulation = AISimulation()
    
    # Create dummy stats for testing
    dummy_stats = {
        'games_played': 10,
        'games_won': 8,
        'games_lost': 2,
        'max_score': 50000,
        'max_tile': 1024,
        'total_moves': 150,
        'win_rate': 80.0,
        'total_ai_latency': 2.5,
        'ai_suggestions_count': 25,
        'total_game_time': 45.0,
        'min_ai_latency': 0.1,
        'max_ai_latency': 0.3,
        'min_game_time': 3.0,
        'max_game_time': 6.0,
        'all_max_tiles': [512, 1024, 512, 256, 1024, 512, 256, 1024, 512, 256],
        'max_tile_distribution': {
            '2-4': 0,
            '8-16': 0,
            '32-64': 0,
            '128-256': 3,
            '512': 4,
            '1024': 3,
            '2048': 0,
            '4096+': 0
        }
    }
    
    dummy_args = {
        'games': 10,
        'max_moves': 1000
    }
    
    # Test logging
    print("📝 Testing CSV logging...")
    simulation.log_simulation_results(dummy_stats, 5.2, dummy_args)
    
    # Check if file was created
    csv_path = os.path.join(simulation.log_dir, "simulation_results.csv")
    if os.path.exists(csv_path):
        print(f"✅ CSV file created successfully: {csv_path}")
        
        # Read and display the CSV content
        import csv
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            print(f"📊 CSV contains {len(rows)} row(s)")
            if rows:
                print("📋 First row data:")
                for key, value in rows[0].items():
                    print(f"  {key}: {value}")
    else:
        print("❌ CSV file was not created")
    
    # Test log summary
    print("\n📊 Testing log summary display...")
    simulation.print_log_summary()
    
    print("\n🎉 Logging system test completed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the 2048_game directory")
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
