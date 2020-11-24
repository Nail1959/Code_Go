import random

from dlgo.agent import Agent
from dlgo.scoring import GameResult
from dlgo.agent import my_predict
import h5py   #Nail

__all__ = [
    'DepthPrunedAgent',
]

MAX_SCORE = 999999
MIN_SCORE = -999999
path_model = r'/home/nail/Code_Go/checkpoints/1000_small_bot.h5'

def reverse_game_result(game_result):
    if game_result == GameResult.loss:
        return game_result.win
    if game_result == GameResult.win:
        return game_result.loss
    return GameResult.draw


# tag::depth-prune[]
def best_result(game_state, max_depth, max_width, eval_fn):
    if game_state.is_over():                               # <1>
        if game_state.winner() == game_state.next_player:  # <1>
            return MAX_SCORE                               # <1>
        else:                                              # <1>
            return MIN_SCORE                               # <1>

    if max_depth == 0:                                     # <2>
        return eval_fn(game_state)                         # <2>

    best_so_far = MIN_SCORE
    #predict_moves = game_state.legal_moves()[:max_width]   # Nail
    agnt = my_predict.load_prediction_agent(h5py.File(path_model,'r'))
    predict_moves = agnt.select_ranked_move(game_state, max_width)
    predict_moves = predict_moves[:max_width]
    for candidate_move in predict_moves:        # <3>      # Nail
        next_state = game_state.apply_move(candidate_move) # <4>
        opponent_best_result = best_result(                # <5>
            next_state, max_depth - 1, max_width, eval_fn) # <5>  Nail
        our_result = -1 * opponent_best_result             # <6>
        if our_result > best_so_far:                       # <7>
            best_so_far = our_result                       # <7>

    return best_so_far
# end::depth-prune[]


# tag::depth-prune-agent[]
class DepthPrunedAgent(Agent):
    def __init__(self, max_depth, max_width,eval_fn):
        Agent.__init__(self)
        self.max_depth = max_depth
        self.eval_fn = eval_fn
        self.predict_moves = 0   #Nail
        self.max_width = max_width       #Nail

    def select_move(self, game_state):
        best_moves = []
        best_score = None
        #predict_moves = game_state.legal_moves()[:self.max_width]     # Nail
        agnt = my_predict.load_prediction_agent(h5py.File(path_model,'r'))
        predict_moves = agnt.select_ranked_move(game_state,self.max_width)
        predict_moves = predict_moves[:self.max_width]
        # Loop over all legal moves.
        for possible_move in predict_moves:              # Nail
            # Calculate the game state if we select this move.
            next_state = game_state.apply_move(possible_move)
            # Since our opponent plays next, figure out their best
            # possible outcome from there.
            opponent_best_outcome = best_result(next_state, self.max_depth, self.max_width, self.eval_fn)
            # Our outcome is the opposite of our opponent's outcome.
            our_best_outcome = -1 * opponent_best_outcome
            if (not best_moves) or our_best_outcome > best_score:
                # This is the best move so far.
                best_moves = [possible_move]
                best_score = our_best_outcome
            elif our_best_outcome == best_score:
                # This is as good as our previous best move.
                best_moves.append(possible_move)
        # For variety, randomly select among all equally good moves.
        return random.choice(best_moves)
# end::depth-prune-agent[]
