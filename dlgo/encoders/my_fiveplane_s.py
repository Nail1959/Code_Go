import numpy as np

from dlgo.encoders.base import Encoder
from dlgo.goboard_fast import Move, Point,Board_Ext, Board


class MyFivePlaneEncoder_S(Encoder):
    def __init__(self, board_size):
        self.board_width, self.board_height = board_size
        self.num_planes = 5
        board = Board(num_rows=self.board_width, num_cols=self.board_height)
        board_ext = Board_Ext(board)

    def name(self):
        return 'my_fiveplane_s'
# end::fiveplane_init[]

# tag::fiveplane_encode[]
    def encode(self, game_state, board_ext):
        board_tensor = np.zeros(self.shape())
        #Modify Board_ext for stones where value cost == 1 or cost == -1

        try:
            if len(board_ext._grid_ext) > 4:
                max_value, point_black = board_ext.find_max_value()
                go_string_black = game_state.board.get_go_string(point_black)
                for point in go_string_black.stones:
                        board_ext._grid_ext[point][2] = 1
        except:

           pass


        try:
            if len(board_ext._grid_ext) > 4:
                min_value, point_white = board_ext.find_min_value()
                go_string_white = game_state.board.get_go_string(point_white)
                for point in go_string_white.stones:
                    board_ext._grid_ext[point][2] = -1
        except:
            pass

        for row in range(self.board_height):
            for col in range(self.board_width):
                p = Point(row=row + 1, col=col + 1)
                go_string = game_state.board.get_go_string(p)

                if go_string is None:
                    if game_state.does_move_violate_ko(game_state.next_player,
                                                       Move.play(p)):
                        board_tensor[4][row][col] = 1  # <1>  For Ko situation both is equal

                else:

                    if board_ext._grid_ext[p][2] < 0:   # for white stones
                        if board_ext._grid_ext[p][2] > -0.5:
                            board_tensor[2][row][col] = 0.5
                        else:
                            board_tensor[2][row][col] = -board_ext._grid_ext[p][2]

                        liberty_white =1 - 1.0 / (go_string.num_liberties+1)
                        board_tensor[3][row][col] = liberty_white

                    if board_ext._grid_ext[p][2] > 0:  # for black stones
                        if  board_ext._grid_ext[p][2] < 0.5:
                            board_tensor[0][row][col] = 0.5
                        else:
                            board_tensor[0][row][col] =  board_ext._grid_ext[p][2]

                        liberty_black = 1 - 1.0 / (go_string.num_liberties+1)
                        board_tensor[1][row][col] = liberty_black
        return board_tensor
# <1> Encoding moves prohibited by the ko rule
# <2> Encoding black and white stones with  liberties.
# end::myfivenplane_encode[]

# tag::myfivenplane_rest[]
    def encode_point(self, point):
        return self.board_width * (point.row - 1) + (point.col - 1)

    def decode_point_index(self, index):
        row = index // self.board_width
        col = index % self.board_width
        return Point(row=row + 1, col=col + 1)

    def num_points(self):
        return self.board_width * self.board_height

    def shape(self):
        return self.num_planes, self.board_height, self.board_width


def create(board_size):
    return MyFivePlaneEncoder_S(board_size)
