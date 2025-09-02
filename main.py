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
import random

# Initialize Flask application
app = Flask(__name__)

# Global game state - maintains current game across HTTP requests
# This is a simple approach suitable for single-user gameplay
game_state = {
    'board': [],           # 2D list representing the game board
    'score': 0,            # Current player score
    'game_status': 'play'  # Game state: 'play', 'win', or 'lose'
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
    return jsonify(game_state)

@app.route('/move', methods=['POST'])
def move():
    """Process a player move in the specified direction."""
    global game_state
    if game_state['game_status'] != 'play':
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

if __name__ == '__main__':
    app.run(debug=True)
