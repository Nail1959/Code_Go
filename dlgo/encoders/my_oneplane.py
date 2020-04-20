# tag::oneplane_imports[]
import numpy as np

from dlgo.encoders.base import Encoder
from dlgo.goboard import Point, Board_Ext
# end::oneplane_imports[]


# tag::oneplane_encoder[]
class MyOnePlaneEncoder(Encoder):
    def __init__(self, board_size):
        self.board_width, self.board_height = board_size
        self.num_planes = 1
        self. _grid_ext = {}


    def name(self):  # <1>
        return 'my_oneplane'

    def encode(self, game_state,board_ext=None):  # <2>
        #board_ext = Board_Ext(game_state.board)  #Nail inserted
        l = len(board_ext._grid_ext)
        board_matrix = np.zeros(self.shape())

        next_player = game_state.next_player
        color = 'w'
        if next_player.name == 'black':
            color = 'b'
        for r in range(self.board_height):
            for c in range(self.board_width):
                p = Point(row=r + 1, col=c + 1)

                try:
                    stone, cost = board_ext._grid_ext[p][1], board_ext._grid_ext[p][2]  #Nail
                except:
                    stone = None
                    cost = 0

                if stone is None:
                    continue
                if stone == 'b' :
                    board_matrix[0, r, c] = cost
                if stone == 'w':
                    board_matrix[0, r, c] = -cost
                # if stone.name == 'black' :
                #     board_matrix[0, r, c] = cost
                # if stone.name == 'white':
                #     board_matrix[0, r, c] = -cost

        #print(board_matrix)
        return board_matrix

# <1> We can reference this encoder by the name "oneplane".
# <2> To encode, we fill a matrix with 1 if the point contains one of the current player's stones, -1 if the point contains the opponent's stones and 0 if the point is empty.
# end::oneplane_encoder[]

# tag::oneplane_encoder_2[]
    def encode_point(self, point):  # <1>
        return self.board_width * (point.row - 1) + (point.col - 1)

    def decode_point_index(self, index):  # <2>
        row = index // self.board_width
        col = index % self.board_width
        return Point(row=row + 1, col=col + 1)

    def num_points(self):
        return self.board_width * self.board_height

    def shape(self):
        return self.num_planes, self.board_height, self.board_width

# <1> Turn a board point into an integer index.
# <2> Turn an integer index into a board point.
# end::oneplane_encoder_2[]


# tag::oneplane_create[]
def create(board_size):
    return MyOnePlaneEncoder(board_size)
# end::oneplane_create[]