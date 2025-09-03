# 🎮 2048 Game with AI

A modern web-based 2048 puzzle game featuring an intelligent AI that uses expectiminimax algorithm with snake heuristic to provide move suggestions and autopilot gameplay.

## 🚀 Quick Start

| Action | Command |
|--------|---------|
| **Install** | `pip install -r requirements.txt` |
| **Run Game** | `python main.py` |
| **Run Tests** | `python unit_tests.py` |
| **AI Simulation** | `python ai_simulation.py 100` |

## 🎯 Features

| Feature | Description |
|---------|-------------|
| **Classic 2048** | Slide tiles to merge identical numbers |
| **AI Hints** | Get intelligent move suggestions |
| **AI Autopilot** | Watch AI play automatically (continues past 2048!) |
| **Real-time Stats** | Track score, moves, and game state |
| **Responsive UI** | Works on desktop and mobile |

---

## 🎬 Live Demos

<div align="center">

### 🤖 AI Autopilot in Action
*Watch the AI play automatically, achieving high scores with strategic moves*

<img src="media/AI_autopilot.gif" alt="AI Autopilot Demo">

---

### 💡 Get AI Hint Feature  
*Get intelligent move suggestions from the AI to improve your gameplay*

<img src="media/Get_AI_Hint.gif" width="290" height="350" alt="Get AI Hint Demo">

</div>

---

## 🎮 How to Play

1. **Start**: `python main.py` → Open `http://localhost:5000`
2. **Controls**: Arrow keys or WASD to move tiles
3. **AI Help**: Click "Get AI Hint" for suggestions
4. **Autopilot**: Click "🤖 AI Autopilot" to watch AI play

## 🤖 AI System

### Algorithm Overview
- **Expectiminimax**: Handles stochastic tile placement with depth 2 lookahead
- **Snake Heuristic**: Uses exponential weights for optimal tile arrangement
- **Performance**: ~0.07s average response time, 94.9% win rate

### Snake Pattern Strategy
```
Perfect Snake Pattern:
[2,    4,    8,   16]
[256, 128,  64,   32]
[512, 1024, 2048, 4096]
[8192, 4096, 2048, 1024]
```

## 📁 Project Structure

| File | Purpose |
|------|---------|
| `main.py` | Flask web server and API endpoints |
| `game_logic.py` | Core game mechanics (board, moves, scoring) |
| `ai_backend.py` | AI algorithms and decision making |
| `ai_simulation.py` | AI performance testing and benchmarking |
| `unit_tests.py` | Comprehensive test suite |
| `templates/index.html` | Frontend UI |

## 🧪 Testing & Simulation

### Unit Tests
```bash
python unit_tests.py  # Run all tests
```

### AI Performance Testing
```bash
python ai_simulation.py                # 50 games (default)
python ai_simulation.py 1000           # 1000 games
python ai_simulation.py --create-plot  # Generate plots
```

### Performance Results (1,000 Games)
| Metric | Value |
|--------|-------|
| **Win Rate** | 94.9% |
| **AI Latency** | 0.07s |
| **Avg Game Time** | 66.65s |
| **Max Score** | 24,140 |
| **Avg Moves** | 953.4 |

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main game page |
| `/start` | POST | Initialize new game |
| `/move` | POST | Execute player move |
| `/ai_move` | GET | Get AI suggestion |
| `/autopilot/start` | POST | Start autopilot mode |
| `/autopilot/stop` | POST | Stop autopilot mode |
| `/autopilot/move` | POST | Execute AI move |

## 📊 AI Performance Analysis

### Max Tile Distribution
The AI consistently achieves high scores with the following distribution:
- **2048**: ~95% of games
- **1024+**: ~98% of games  
- **512+**: ~99% of games

### Monte Carlo Forecast (100,000 Games)
- **Expected Win Rate**: 96.8% ± 0.3%
- **Confidence**: 95% confidence interval
- **Improvement**: +1.9% from current performance

## 🛠️ Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| Flask | 3.1.2 | Web framework |
| pandas | 2.3.2 | Data analysis |
| matplotlib | 3.9.4 | Visualization |
| numpy | 2.0.2 | Numerical computing |
| pytest | ≥7.0.0 | Testing framework |

## 🎯 Game Modes

| Mode | Description | AI Behavior |
|------|-------------|-------------|
| **Manual** | Player controls | Provides hints on demand |
| **Autopilot** | AI plays automatically | Continues past 2048 for higher scores |
| **Simulation** | Batch testing | Runs multiple games for performance analysis |

## 🏆 Key Achievements

- ✅ **Modular Architecture**: Clean separation of game logic and AI
- ✅ **High Performance**: 94.9% win rate with fast response times
- ✅ **Comprehensive Testing**: 100% test success rate
- ✅ **Production Ready**: Robust error handling
- ✅ **Scalable**: Parallel processing

---

**Ready to play?** 🎮 Start with `python main.py` and enjoy the intelligent 2048 experience!