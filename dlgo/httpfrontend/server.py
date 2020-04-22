# -*- coding: utf-8 -*-
import os

from flask import Flask, redirect
from flask import jsonify
from flask import request

from dlgo import agent
from dlgo import goboard_fast as goboard
from dlgo.utils import coords_from_point
from dlgo.utils import point_from_coords

from dlgo.scoring import compute_game_result as gr

__all__ = [
    'get_web_app',
]


def get_web_app(bot_map):
    """Create a flask application for serving bot moves.
    The bot_map maps from URL path fragments to Agent instances.
    The /static path will return some static content (including the
    jgoboard JS).
    Clients can get the post move by POSTing json to
    /select-move/<bot name>
    Example:
    >>> myagent = agent.RandomBot()
    >>> web_app = get_web_app({'random': myagent})
    >>> web_app.run()
    Returns: Flask application instance
    """
    here = os.path.dirname(__file__)
    static_path = os.path.join(here, 'static')
    app = Flask(__name__, static_folder=static_path, static_url_path='/static')

    @app.route('/')
    def redir():
        return redirect('http://localhost:5000/static/Human_vs_Bot_19size.html')

    @app.route('/select-move/<bot_name>', methods=['POST'])
    def select_move(bot_name):
        content = request.json
        board_size = content['board_size']
        game_state = goboard.GameState.new_game(board_size)
        board_ext = goboard.Board_Ext(game_state.board)  #Nail
        # Replay the game up to this point.
        for move in content['moves']:
            if move == 'pass':
                next_move = goboard.Move.pass_turn()
            elif move == 'resign':
                next_move = goboard.Move.resign()
            else:
                next_move = goboard.Move.play(point_from_coords(move))
            game_state = game_state.apply_move(next_move)

            p = next_move.point  # Nail
            board_ext.place_stone_ext(game_state.board, game_state.next_player.other.name[0], p)  #Nail
# Here need insert my encoder result  Nail
        bot_agent = bot_map[bot_name]

        bot_move = bot_agent.select_move(game_state,board_ext)  # Nail
        if bot_move.is_pass:
            bot_move_str = 'pass'
        elif bot_move.is_resign:
            bot_move_str = 'resign'
        else:
            bot_move_str = coords_from_point(bot_move.point)

        result_scoring = gr(game_state)
        print('Current Result = ', result_scoring, ' Bot_move = ', bot_move_str)
        return jsonify({
            'bot_move': bot_move_str,
            'diagnostics': bot_agent.diagnostics()
        })

    return app