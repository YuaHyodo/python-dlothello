import creversi as reversi
import numpy.random as R

"""
テストとかに使うランダム局面を生成する
"""

def make_random_board(num):
    while True:
        try:
            board = reversi.Board()
            for i in range(num):
                board.move(R.choice(list(board.legal_moves)))
            break
        except:
            pass
    return board

if __name__ == '__main__':
    for i in range(6):
        board = make_random_board((i + 1) *10)
        print('')
        print('===board===')
        print(board)
        print('==========')
        print('')
