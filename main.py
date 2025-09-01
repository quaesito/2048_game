from flask import Flask, jsonify, render_template, request
import game_logic as game
import random

app = Flask(__name__)

# Global state for the game board and score
game_state = {
    'board': [],
    'score': 0,
    'game_status': 'play'
}

@app.route('/')
def index():
    """Renders the main game page."""
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_game():
    """Initializes a new game."""
    global game_state
    game_state['board'] = game.new_game()
    game.add_random_tile(game_state['board'])
    game.add_random_tile(game_state['board'])
    game_state['score'] = 0
    game_state['game_status'] = 'play'
    return jsonify(game_state)

@app.route('/move', methods=['POST'])
def move():
    """Handles a player's move."""
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
    """Gets an AI-suggested move."""
    global game_state
    if game_state['game_status'] != 'play':
         return jsonify({'suggestion': 'Game Over'})
         
    suggestion = game.get_ai_suggestion(game_state['board'])
    return jsonify({'suggestion': suggestion})

if __name__ == '__main__':
    app.run(debug=True)
