# tag::dl_agent_imports[]
import numpy as np
import copy   # Nail
from dlgo.scoring import compute_game_result as gr   # Nail

from dlgo.agent.base import Agent
from dlgo.agent.helpers import is_point_an_eye
from dlgo import encoders
from dlgo import goboard
from dlgo import kerasutil
# end::dl_agent_imports[]
__all__ = [
    'DeepLearningAgent',
    'load_prediction_agent',
]


# tag::dl_agent_init[]
class DeepLearningAgent(Agent):
    def __init__(self, model, encoder):
        Agent.__init__(self)
        self.model = model
        self.encoder = encoder
# end::dl_agent_init[]

# tag::dl_agent_predict[]
    def predict(self, game_state, board_ext=None):
        if self.encoder.name()[:2] == 'my':
            #board_ext = goboard.Board_Ext(game_state.board)
            encoded_state = self.encoder.encode(game_state, board_ext )  #Nail
        else:
            encoded_state = self.encoder.encode(game_state)
        input_tensor = np.array([encoded_state])
        return self.model.predict(input_tensor)[0]

    def select_ranked_move(self, game_state,move_width=3,board_ext=None):
        num_moves = self.encoder.board_width * self.encoder.board_height
        move_probs = self.predict(game_state, board_ext)
# end::dl_agent_predict[]

# tag::dl_agent_probabilities[]
        move_probs=list(move_probs)
        move_pr = sorted(move_probs, reverse=True)[:move_width]  # Nail
        ranked_moves =list()
        for mr in move_pr:
            for mp in move_probs:
                if mp == mr:
                    ranked_moves.append(move_probs.index(mp))

        possible_moves = []


        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index(point_idx)
            if game_state.is_valid_move(goboard.Move.play(point)) and \
                    not is_point_an_eye(game_state.board, point, game_state.next_player):  #

                possible_moves.append(goboard.Move.play(point))

        if len(possible_moves) == 0:   # Нет допустимых ходов, тогда пас.
            return goboard.Move.pass_turn()  # <4>
        else:
            return possible_moves





    def select_move(self, game_state, board_ext=None):
        num_moves = self.encoder.board_width * self.encoder.board_height
        move_probs = self.predict(game_state, board_ext)
        # end::dl_agent_predict[]

        # tag::dl_agent_probabilities[]
        move_probs = move_probs ** 3  # <1>
        eps = 1e-6
        move_probs = np.clip(move_probs, eps, 1 - eps)  # <2>
        move_probs = move_probs / np.sum(move_probs)  # <3>
        # <1> Increase the distance between the move likely and least likely moves.
        # <2> Prevent move probs from getting stuck at 0 or 1
        # <3> Re-normalize to get another probability distribution.
        # end::dl_agent_probabilities[]

        # tag::dl_agent_candidates[]
        candidates = np.arange(num_moves)  # <1>
        ranked_moves = np.random.choice(
            candidates, num_moves, replace=False, p=move_probs)  # <2>
        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index(point_idx)
            if game_state.is_valid_move(goboard.Move.play(point)) and \
                    not is_point_an_eye(game_state.board, point, game_state.next_player):  # <3>
                return goboard.Move.play(point)
        return goboard.Move.pass_turn()  # <4>

    def my_select_move(self, game_state, board_ext=None):
        num_moves = self.encoder.board_width * self.encoder.board_height
        move_probs = self.predict(game_state, board_ext)
        # end::dl_agent_predict[]

        # tag::dl_agent_probabilities[]
        move_probs = move_probs ** 3  # <1>
        eps = 1e-6
        move_probs = np.clip(move_probs, eps, 1 - eps)  # <2>
        move_probs = move_probs / np.sum(move_probs)  # <3>
        # <1> Increase the distance between the move likely and least likely moves.
        # <2> Prevent move probs from getting stuck at 0 or 1
        # <3> Re-normalize to get another probability distribution.
        # end::dl_agent_probabilities[]

        # tag::dl_agent_candidates[]
        candidates = np.arange(num_moves)  # <1>
        ranked_moves = np.random.choice(
            candidates, num_moves, replace=False, p=move_probs)  # <2>

        possible_point = []  # Список всех доступных ходов предложенных сетью.
        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index(point_idx)
            if game_state.is_valid_move(goboard.Move.play(point)) and \
                    not is_point_an_eye(game_state.board, point, game_state.next_player):  # <3>
                possible_point.append(point)
        if len(possible_point) == 0:   # Нет допустимых ходов, тогда пас.
            return goboard.Move.pass_turn()  # <4>
        # Выбрать из всех возможных ходов
        score = 0   # Счет на доске, выбрать ход приносящий максимально допустимый счет #<5>
        for p in possible_point:
            game_state_copy = copy.deepcopy(game_state)
            game_state_copy = game_state_copy.apply_move(p)
            res = str(gr(game_state_copy))[1:]  # Отбрасываю B или W, оставляю знак
            res = float(res)
            if res > score:
               score = res
               point = p
        return goboard.Move.play(point)

# <1> Turn the probabilities into a ranked list of moves.
# <2> Sample potential candidates
# <3> Starting from the top, find a valid move that doesn't reduce eye-space.
# <4> If no legal and non-self-destructive moves are left, pass.
# end::dl_agent_candidates[]

# tag::dl_agent_serialize[]
    def serialize(self, h5file):
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self.encoder.name()
        h5file['encoder'].attrs['board_width'] = self.encoder.board_width
        h5file['encoder'].attrs['board_height'] = self.encoder.board_height
        h5file.create_group('model')
        kerasutil.save_model_to_hdf5_group(self.model, h5file['model'])
# end::dl_agent_serialize[]


# tag::dl_agent_deserialize[]
def load_prediction_agent(h5file):
    model = kerasutil.load_model_from_hdf5_group(h5file['model'])
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    encoder = encoders.get_encoder_by_name(
        encoder_name, (board_width, board_height))
    return DeepLearningAgent(model, encoder)
# tag::dl_agent_deserialize[]