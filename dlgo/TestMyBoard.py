from dlgo import goboard_slow
from dlgo import gotypes
from dlgo.utils import print_board, print_move
from goboard_slow import Board,GameState, GoString
from gotypes import Point, Player

ngame = GameState.new_game(9)
gobrd = Board(9,9)
p1 = Point(row=2,col=2)
p2 = Point(row=3,col=3)
player = Player.black

str = gobrd.place_stone(player, p1)
print(str)
str = gobrd.place_stone(player, p2)


print_board(ngame.board)