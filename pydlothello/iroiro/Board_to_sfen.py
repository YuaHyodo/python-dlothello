import creversi as reversi
"""
CreversiのBoardをsfenに変換する
"""

def board_to_sfen(board, turn):
    line = board.to_line()
    if board.turn:
        turn_of = 'B'
    else:
        turn_of = 'W'
    sfen = line + turn_of + str(turn)
    return sfen

if __name__ == '__main__':
    import numpy.random as R
    test = reversi.Board()
    print(test)
    print(board_to_sfen(test, 1))
    for i in range(30):
        print('')
        print(i)
        test.move(R.choice(list(test.legal_moves)))
        print(test)
        print(board_to_sfen(test, i + 2))
        print('')
