# 2048 Game with AI Suggestions

A web-based implementation of the classic 2048 puzzle game built with Flask, featuring an AI-powered move suggestion system to help players improve their gameplay.

## 🎮 Game Overview

2048 is a single-player sliding block puzzle game. The objective is to slide numbered tiles on a grid to combine them to create a tile with the number 2048. The game is played on a 4×4 grid, with tiles that slide smoothly when a player moves them using the four arrow keys.

## 🚀 Features

- **Classic 2048 Gameplay**: Slide tiles to merge identical numbers
- **Real-time Score Tracking**: Keep track of your score as you play
- **AI Move Suggestions**: Get intelligent move recommendations to improve your strategy
- **Responsive Web Interface**: Clean, modern UI that works on desktop and mobile
- **Game State Management**: Automatic win/lose detection and game status updates

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Setup
1. **Clone or navigate to the project directory:**
   ```bash
   cd 2048_game
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   If you don't have a requirements.txt file, install Flask manually:
   ```bash
   pip install flask
   ```

## 🎯 How to Play

1. **Start the game:**
   ```bash
   python main.py
   ```

2. **Open your browser and navigate to:**
   ```
   http://localhost:5000
   ```

3. **Game Controls:**
   - Use arrow keys or WASD to move tiles
   - Click "New Game" to start a fresh game
   - Click "AI Hint" to get move suggestions

4. **Objective:**
   - Combine identical tiles by sliding them together
   - Reach the 2048 tile to win
   - Avoid getting stuck with no possible moves

## 🤖 Advanced AI Suggestion System

The AI suggestion system uses a sophisticated **minimax algorithm with alpha-beta pruning** combined with multiple advanced heuristics to provide intelligent move recommendations. This represents a significant improvement over basic greedy algorithms.

### Core Algorithm

The AI uses a **multi-layered approach** that combines:

1. **Minimax with Alpha-Beta Pruning**: Looks 3 moves ahead to predict optimal strategies
2. **Advanced Heuristics**: Evaluates board positions using 5 weighted factors
3. **Corner Strategy**: Implements the proven "corner strategy" for high scores
4. **Monotonicity Optimization**: Maintains consistent tile value patterns

### Advanced Heuristics

The AI evaluates board positions using these weighted factors:

1. **Monotonicity (Weight: 1.0)**: Prefers boards where tile values increase consistently in one direction
2. **Smoothness (Weight: 0.1)**: Favors boards with similar adjacent tile values to enable merges
3. **Free Tiles (Weight: 2.0)**: Prioritizes moves that maintain more empty cells
4. **Max Tile Value (Weight: 1.0)**: Rewards achieving higher tile values
5. **Corner Strategy (Weight: 1.5)**: Bonus for keeping the largest tile in a corner

### Implementation Details

The advanced AI system includes these key functions:

- **`get_ai_suggestion(board)`**: Main function using minimax algorithm
- **`evaluate_board(board)`**: Multi-heuristic board evaluation
- **`minimax(board, depth, is_maximizing, alpha, beta)`**: Lookahead algorithm with pruning
- **`simulate_move_with_random_tile(board, direction)`**: Complete move simulation
- **`get_monotonicity_score(board)`**: Monotonicity pattern analysis
- **`get_smoothness_score(board)`**: Adjacent tile similarity scoring
- **`get_corner_strategy_score(board)`**: Corner placement optimization

### How the Advanced AI Works

1. **Lookahead Analysis**: The AI simulates 3 moves ahead using minimax algorithm
2. **Move Simulation**: For each possible move, it simulates the complete game state including random tile placement
3. **Multi-Heuristic Evaluation**: Each board position is scored using 5 different strategic factors
4. **Alpha-Beta Pruning**: Optimizes performance by eliminating unpromising branches
5. **Strategic Decision**: Selects the move that leads to the best long-term position

### Advanced Strategy Features

The AI implements proven 2048 strategies:

- **Corner Strategy**: Keeps the highest-value tile in a corner position
- **Snake Pattern**: Maintains a decreasing value pattern across the board
- **Merge Preservation**: Identifies and protects potential merge opportunities
- **Space Management**: Optimizes empty cell distribution for future moves
- **Long-term Planning**: Considers multiple future moves, not just immediate gains

### Performance Optimization

- **Depth Limiting**: Uses 3-move lookahead for optimal performance
- **Alpha-Beta Pruning**: Reduces computational complexity by ~50%
- **Weighted Scoring**: Balances different strategic factors for optimal decisions
- **Efficient Simulation**: Fast board state copying and evaluation

## 📁 Project Structure

```
2048_game/
├── main.py              # Flask application and API endpoints
├── game_logic.py        # Core game logic and AI algorithms
├── templates/
│   └── index.html      # Frontend HTML/CSS/JavaScript
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🔧 API Endpoints

- **`GET /`**: Main game page
- **`POST /start`**: Initialize a new game
- **`POST /move`**: Execute a player move
- **`GET /ai_move`**: Get AI move suggestion

---

**Happy gaming!** 🎮✨
