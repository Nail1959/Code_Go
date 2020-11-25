import random

from dlgo.agent import Agent
from dlgo.gotypes import Player

__all__ = [
    'AlphaBetaAgent',
]

MAX_SCORE = 999999
MIN_SCORE = -999999



# tag::alpha-beta-prune-1[]
def alpha_beta_result(game_state, max_depth, max_width, best_black, best_white, agnt, eval_fn):
    if game_state.is_over():                                   # <1>
        if game_state.winner() == game_state.next_player:      # <1>
            return MAX_SCORE                                   # <1>
        else:                                                  # <1>
            return MIN_SCORE                                   # <1>

    if max_depth == 0:                                         # <2>

        return eval_fn(game_state)                             # <2>

    best_so_far = MIN_SCORE

    predict_moves = agnt.select_ranked_move(game_state, max_width)
    #predict_moves = predict_moves[:max_width]
    for candidate_move in predict_moves:                       # <3>
        next_state = game_state.apply_move(candidate_move)     # <4>
        opponent_best_result = alpha_beta_result(              # <5>
            next_state, max_depth - 1, max_width,              # <5>
            best_black, best_white, agnt,                      # <5>
            eval_fn)                                           # <5>
        our_result = -1 * opponent_best_result                 # <6>

        if our_result > best_so_far:                           # <7>
            best_so_far = our_result                           # <7>
# end::alpha-beta-prune-1[]

# tag::alpha-beta-prune-2[]
        if game_state.next_player == Player.white:
            if best_so_far > best_white:                       # <8>
                best_white = best_so_far                       # <8>
            outcome_for_black = -1 * best_so_far               # <9>
            if outcome_for_black < best_black:                 # <9>
                return best_so_far                             # <9>
# end::alpha-beta-prune-2[]
# tag::alpha-beta-prune-3[]
        elif game_state.next_player == Player.black:
            if best_so_far > best_black:                       # <10>
                best_black = best_so_far                       # <10>
            outcome_for_white = -1 * best_so_far               # <11>
            if outcome_for_white < best_white:                 # <11>
                return best_so_far                             # <11>
# end::alpha-beta-prune-3[]
# tag::alpha-beta-prune-4[]
#     print('Depth = ', max_depth, ' best_so_far = ', best_so_far)
    return best_so_far
# end::alpha-beta-prune-4[]


# tag::alpha-beta-agent[]
class AlphaBetaAgent(Agent):
    def __init__(self, max_depth, max_width, agnt, eval_fn):
        Agent.__init__(self)
        self.max_depth = max_depth
        self.max_width = max_width
        self.eval_fn = eval_fn


    def select_move(self, game_state, agnt):
        best_moves = []
        best_score = None
        best_black = MIN_SCORE
        best_white = MIN_SCORE

        predict_moves = agnt.select_ranked_move(game_state, self.max_width)
        predict_moves = predict_moves[:self.max_width]
        # Loop over all legal moves.
        for possible_move in predict_moves:
            # Calculate the game state if we select this move.
            next_state = game_state.apply_move(possible_move)
            # Since our opponent plays next, figure out their best
            # possible outcome from there.
            opponent_best_outcome = alpha_beta_result(
                next_state, self.max_depth, self.max_width,
                best_black, best_white, agnt,
                self.eval_fn)
            # Our outcome is the opposite of our opponent's outcome.
            our_best_outcome = -1 * opponent_best_outcome
            if (not best_moves) or our_best_outcome > best_score:
                # This is the best move so far.
                best_moves = [possible_move]
                best_score = our_best_outcome
                if game_state.next_player == Player.black:
                    best_black = best_score
                elif game_state.next_player == Player.white:
                    best_white = best_score
            elif our_best_outcome == best_score:
                # This is as good as our previous best move.
                best_moves.append(possible_move)
        # For variety, randomly select among all equally good moves.
        return random.choice(best_moves)
# end::alpha-beta-agent[]
