from six.moves import input

from dlgo import goboard
from dlgo import gotypes
from dlgo import minimax
from dlgo.utils import print_board, print_move, point_from_coords
import h5py
from dlgo.agent import my_predict

import os
import time
from dlgo.scoring import my_compute_game_result as gr

path_model = r'/home/nail/Code_Go/checkpoints/20000_large_simple_bot.h5'
path_wav = r'/home/nail/Code_Go/checkpoints/w1.mp3'


from playsound import playsound

BOARD_SIZE = 19
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# tag::naive-board-heuristic[]
def capture_diff(game_state):
    black_stones = 0
    white_stones = 0
    for r in range(1, game_state.board.num_rows + 1):
        for c in range(1, game_state.board.num_cols + 1):
            p = gotypes.Point(r, c)
            color = game_state.board.get(p)
            if color == gotypes.Player.black:
                black_stones += 1
            elif color == gotypes.Player.white:
                white_stones += 1
    diff = black_stones - white_stones
    if game_state.next_player == gotypes.Player.black:
        return diff
    return -1 * diff
# end::naive-board-heuristic[]
def territory_diff(game_state):
    res, tb, tw = gr(game_state)
    #print('result = ',res, 'territory_diff = ', tb - tw)
    return tb - tw

def main():
    game = goboard.GameState.new_game(BOARD_SIZE)
    max_depth = int(input('Depth search = '))
    max_width = int(input('Width search = '))

    agnt = my_predict.load_prediction_agent(h5py.File(path_model, 'r'))

    bot = minimax.AlphaBetaAgent(max_depth=max_depth, max_width=max_width, agnt=agnt, eval_fn=territory_diff)

    while not game.is_over():
        print_board(game.board)
        if game.next_player == gotypes.Player.black:
            human_move = input('-- ').upper()    # Nail
            #os.system("mpg123 " + path_wav)
            point = point_from_coords(human_move.strip())
            move = goboard.Move.play(point)
        else:
            time_begin = time.time()
            move = bot.select_move(game, agnt)
            time_select = time.time() - time_begin
            print('Time selection move = ', time_select)
        print_move(game.next_player, move)

        game = game.apply_move(move)


if __name__ == '__main__':
    main()
