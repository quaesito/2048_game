"""
2048 Game Flask Web Application

This module provides a Flask web server for the 2048 puzzle game.
It handles HTTP requests for game interactions including:
- Game initialization and state management
- Player move processing
- AI move suggestions
- Real-time game state updates

The application uses a global game state to maintain the current board,
score, and game status across HTTP requests.
"""

from flask import Flask, jsonify, render_template, request
import game_logic as game
from ai_backend import get_ai_suggestion
import random

# Initialize Flask application
app = Flask(__name__)

# Global game state - maintains current game across HTTP requests
# This is a simple approach suitable for single-user gameplay
game_state = {
    'board': [],           # 2D list representing the game board
    'score': 0,            # Current player score
    'game_status': 'play', # Game state: 'play', 'win', 'lose', or 'won_2048'
    'autopilot_mode': False, # Whether autopilot is active
    'achieved_2048': False   # Whether 2048 tile has been achieved
}

@app.route('/')
def index():
    """Serve the main game HTML page."""
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_game():
    """Initialize a new game with empty board and two random tiles."""
    global game_state
    game_state['board'] = game.new_game()
    game.add_random_tile(game_state['board'])
    game.add_random_tile(game_state['board'])
    game_state['score'] = 0
    game_state['game_status'] = 'play'
    game_state['autopilot_mode'] = False
    game_state['achieved_2048'] = False
    return jsonify(game_state)

@app.route('/move', methods=['POST'])
def move():
    """Process a player move in the specified direction."""
    global game_state
    if game_state['game_status'] not in ['play', 'won_2048']:
        return jsonify(game_state)

    direction = request.json.get('direction')
    if not direction:
        return jsonify({'error': 'Direction not provided'}), 400

    move_functions = {
        'up': game.move_up,
        'down': game.move_down,
        'left': game.move_left,
        'right': game.move_right,
    }

    if direction in move_functions:
        move_func = move_functions[direction]
        new_board, score_gain, has_moved = move_func(game_state['board'])
        
        if has_moved:
            game_state['board'] = new_board
            game_state['score'] += score_gain
            game.add_random_tile(game_state['board'])
            
            # Check for 2048 achievement (when tile is first created)
            if not game_state['achieved_2048']:
                for row in game_state['board']:
                    for cell in row:
                        if cell == 2048:
                            game_state['achieved_2048'] = True
                            game_state['game_status'] = 'won_2048'
                            return jsonify(game_state)
            
            # Use appropriate game state function based on mode
            if game_state['autopilot_mode']:
                game_state['game_status'] = game.get_game_state_autopilot(game_state['board'])
            else:
                game_state['game_status'] = game.get_game_state(game_state['board'])

    return jsonify(game_state)

@app.route('/ai_move', methods=['GET'])
def ai_move():
    """Get AI move suggestion for the current board state."""
    global game_state
    if game_state['game_status'] != 'play':
         return jsonify({'suggestion': 'Game Over'})
         
    suggestion = game.get_ai_suggestion(game_state['board'])
    return jsonify({'suggestion': suggestion})

@app.route('/game_state', methods=['GET'])
def get_game_state():
    """Return current game state without modifying it."""
    global game_state
    return jsonify(game_state)

@app.route('/autopilot/start', methods=['POST'])
def start_autopilot():
    """Start autopilot mode."""
    global game_state
    game_state['autopilot_mode'] = True
    return jsonify({'status': 'autopilot_started', 'game_state': game_state})

@app.route('/autopilot/stop', methods=['POST'])
def stop_autopilot():
    """Stop autopilot mode."""
    global game_state
    game_state['autopilot_mode'] = False
    return jsonify({'status': 'autopilot_stopped', 'game_state': game_state})

@app.route('/autopilot/move', methods=['POST'])
def autopilot_move():
    """Execute an AI move in autopilot mode."""
    global game_state
    if not game_state['autopilot_mode']:
        return jsonify({'error': 'Autopilot not active'}), 400
    
    if game_state['game_status'] not in ['play', 'won_2048']:
        return jsonify(game_state)
    
    # Get AI suggestion
    ai_suggestion = get_ai_suggestion(game_state['board'])
    if not ai_suggestion:
        game_state['game_status'] = 'lose'
        return jsonify(game_state)
    
    # Execute the AI move
    move_functions = {
        'up': game.move_up,
        'down': game.move_down,
        'left': game.move_left,
        'right': game.move_right,
    }
    
    if ai_suggestion in move_functions:
        move_func = move_functions[ai_suggestion]
        new_board, score_gain, has_moved = move_func(game_state['board'])
        
        if has_moved:
            game_state['board'] = new_board
            game_state['score'] += score_gain
            game.add_random_tile(game_state['board'])
            
            # Check for 2048 achievement (when tile is first created)
            if not game_state['achieved_2048']:
                for row in game_state['board']:
                    for cell in row:
                        if cell == 2048:
                            game_state['achieved_2048'] = True
                            game_state['game_status'] = 'won_2048'
                            return jsonify(game_state)
            
            # Use autopilot game state function
            game_state['game_status'] = game.get_game_state_autopilot(game_state['board'])
    
    return jsonify(game_state)

if __name__ == '__main__':
    app.run(debug=True)
