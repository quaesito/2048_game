# 2048 Game with AI Suggestions

A web-based implementation of the classic 2048 puzzle game built with Flask, featuring an AI-powered move suggestion system to help players improve their gameplay.

## 🎮 Game Overview

2048 is a single-player sliding block puzzle game. The objective is to slide numbered tiles on a grid to combine them to create a tile with the number 2048. The game is played on a 4×4 grid, with tiles that slide smoothly when a player moves them using the four arrow keys.

### 🎬 Demo

Watch the AI functionality in action:

<div align="center">

## 🤔 Get AI Hint

<img src="media/AI_hint.gif" alt="AI Hint Demo" width="400" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">

*Click "Get AI Hint" to receive intelligent move suggestions*

---

## 🤖 AI Autopilot Mode

<img src="media/AI_autopilot.gif" alt="AI Autopilot Demo" width="400" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">

*Watch the AI play automatically, continuing past 2048 for higher scores!*

</div>

## 🚀 Features

- **Classic 2048 Gameplay**: Slide tiles to merge identical numbers
- **Real-time Score Tracking**: Keep track of your score as you play
- **AI Move Suggestions**: Get intelligent move recommendations to improve your strategy
- **AI Autopilot Mode**: Watch the AI play automatically, continuing past 2048 until manually stopped
- **Responsive Web Interface**: Clean, modern UI that works on desktop and mobile
- **Game State Management**: Automatic win/lose detection and game status updates
- **Modular Architecture**: Clean separation of concerns with dedicated AI, game logic, and web components
- **Comprehensive Testing**: Unit tests, AI simulations, and performance analysis tools

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
   python -m venv 2048_env
   source 2048_env/bin/activate  # On Windows: 2048_env\Scripts\activate
   ```
   
   Or use the existing virtual environment:
   ```bash
   source 2048_env/bin/activate  # On Windows: 2048_env\Scripts\activate
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
   - Click "🤖 AI Autopilot" to let the AI play automatically (continues past 2048!)

4. **Objective:**
   - Combine identical tiles by sliding them together
   - Reach the 2048 tile to win
   - Avoid getting stuck with no possible moves

## 🤖 Advanced AI Suggestion System

The AI suggestion system uses a sophisticated **expectiminimax algorithm with snake heuristic** to provide intelligent move recommendations. This represents a significant improvement over basic greedy algorithms and is optimized for the snake pattern strategy.

### Core Algorithm

The AI uses a **multi-layered approach** that combines:

1. **Expectiminimax Algorithm**: Handles stochastic tile placement with depth 2 lookahead
2. **Snake Heuristic**: Uses exponential weights to encourage snake-like tile patterns
3. **Strategic Evaluation**: Optimizes for the proven snake pattern strategy
4. **Performance Optimization**: Balanced depth for speed and accuracy

### Snake Heuristic System

The AI evaluates board positions using a **snake pattern heuristic** with exponential weights:

```
Perfect Snake Pattern:
[2,    4,    8,   16]
[256, 128,  64,   32]
[512, 1024, 2048, 4096]
[8192, 4096, 2048, 1024]
```

**Key Features:**
- **Exponential Weights**: Higher values in corners get exponentially higher scores
- **Snake Pattern**: Encourages tiles to follow a snake-like decreasing pattern
- **Strategic Placement**: Optimizes for the most effective 2048 strategy

### Implementation Details

The advanced AI system is modularized across multiple files:

**`ai_backend.py` - AI Core:**
- **`AIEvaluator`**: Main AI evaluation class with snake heuristic
- **`snake_heuristic(board)`**: Snake pattern evaluation with exponential weights
- **`enhanced_evaluation(board)`**: Multi-heuristic evaluation function
- **`expectiminimax(board, depth, is_maximizing)`**: Stochastic lookahead algorithm
- **`get_ai_suggestion(board)`**: Main AI suggestion function

**`game_logic.py` - Game Logic:**
- **`move_up/down/left/right(board)`**: Core move functions
- **`get_game_state(board)`**: Game state evaluation
- **`add_random_tile(board)`**: Random tile placement
- **`new_game()`**: Game initialization

**`model.py` - Game Model:**
- **`Board`**: Game board representation and manipulation
- **`move(dir, addNextTile)`**: Model-based move execution
- **`checkLoss()`**: Loss condition checking

### How the Advanced AI Works

1. **Expectiminimax Analysis**: The AI simulates 2 moves ahead handling random tile placement
2. **Snake Pattern Evaluation**: Each board position is scored using exponential snake weights
3. **Stochastic Handling**: Properly accounts for 90% chance of 2-tiles and 10% chance of 4-tiles
4. **Strategic Decision**: Selects the move that leads to the best snake pattern position
5. **Performance Balance**: Depth 2 provides excellent performance with fast response times

### Advanced Strategy Features

The AI implements the proven snake pattern strategy:

- **Snake Pattern**: Maintains a snake-like decreasing value pattern across the board
- **Corner Optimization**: Places highest values in corners with exponential bonuses
- **Stochastic Planning**: Accounts for random tile placement probabilities
- **Long-term Planning**: Considers multiple future moves with proper probability weighting
- **High Win Rate**: Achieves 100% win rate in testing

### Performance Optimization

- **Depth 2 Lookahead**: Optimal balance between performance and accuracy
- **Snake Heuristic**: Fast evaluation using pre-computed exponential weights
- **Efficient Simulation**: Fast board state copying and evaluation
- **Stochastic Handling**: Proper probability weighting for random events
- **Response Time**: Average AI latency of ~0.07 seconds per suggestion

## 🧪 AI Simulation & Testing

The project includes comprehensive AI testing capabilities through `ai_simulation.py`:

### Simulation Features

- **Automated Game Testing**: Run hundreds of AI games automatically
- **Performance Metrics**: Track win rates, scores, and AI response times
- **Parallel Processing**: Multi-core simulation for faster testing
- **Statistical Analysis**: Detailed performance statistics and percentiles
- **Visualization**: Generate plots showing AI performance distribution

### Running Simulations

```bash
# Run 50 games (default)
python ai_simulation.py

# Run 100 games
python ai_simulation.py 100

# Run with custom settings
python ai_simulation.py 25 --max-moves 3000
```

### Simulation Output

- **Real-time Progress**: Live updates during simulation
- **Performance Plots**: Automatic generation of distribution charts
- **CSV Logs**: Detailed results saved to `simulation_logs/`
- **Comprehensive Stats**: Win rates, tile distributions, timing analysis

## 🤖 AI Autopilot Mode

The game features an advanced **AI Autopilot mode** that allows you to watch the AI play the entire game automatically:

### Autopilot Features

- **Fully Automated Play**: The AI makes all moves automatically
- **Continues Past 2048**: Unlike manual play, autopilot doesn't stop at 2048 - it keeps going for higher scores
- **Real-time Statistics**: Shows move count and elapsed time during autopilot
- **Manual Control**: Start/stop autopilot at any time with a single click
- **Visual Feedback**: Button changes color and text to indicate autopilot status
- **Smart Stopping**: Automatically stops when the game is lost or no moves are available

### How to Use Autopilot

1. **Start a Game**: Click "New Game" to begin
2. **Activate Autopilot**: Click the "🤖 AI Autopilot" button
3. **Watch the Magic**: The AI will play automatically, making moves every 100ms
4. **Monitor Progress**: Watch the real-time statistics (moves and time)
5. **Stop Anytime**: Click "⏹️ Stop Autopilot" to regain manual control

### Performance

- **Move Speed**: ~10 moves per second (100ms delay for visibility)
- **High Win Rate**: The AI typically achieves 2048 and continues to higher scores
- **Efficient**: Uses the same expectiminimax algorithm as the hint system
- **Responsive**: Can be stopped instantly at any time

## 📁 Project Structure

```
2048_game/
├── main.py              # Flask web application and API endpoints
├── game_logic.py        # Core game logic and board manipulation
├── ai_backend.py        # AI algorithms and evaluation heuristics
├── model.py             # Game model and board representation
├── ai_simulation.py     # AI performance testing and simulation
├── test_game_logic.py   # Unit tests for game logic
├── templates/
│   └── index.html      # Frontend HTML/CSS/JavaScript
├── media/               # Demo videos and images
│   ├── AI_hint.gif     # AI hint functionality demonstration
│   └── AI_autopilot.gif # AI autopilot mode demonstration
├── simulation_logs/     # AI simulation results and plots
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🔧 API Endpoints

- **`GET /`**: Main game page
- **`POST /start`**: Initialize a new game
- **`POST /move`**: Execute a player move
- **`GET /ai_suggestion`**: Get AI move suggestion and hint (unified endpoint)
- **`GET /game_state`**: Get current game state (for autopilot)
- **`POST /toggle_autopilot`**: Start/stop AI autopilot mode

## 🧪 Testing

The project includes comprehensive testing capabilities:

### Unit Tests
```bash
# Run game logic tests
python test_game_logic.py
```

### AI Performance Testing
```bash
# Run AI simulation
python ai_simulation.py 50
```

### Manual Testing
- Start the Flask server: `python main.py`
- Open browser to `http://localhost:5000`
- Test all game features and AI functionality

---

**Happy gaming!** 🎮✨
