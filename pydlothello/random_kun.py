from base_player import BasePlayer
import creversi as reversi
import numpy as np

class random(BasePlayer):
    def __init__(self):
        super().__init__()
        self.name = 'random_kun'
        self.auther = 'Y_Hyodo'

    def usi(self):
        print('id name ' + self.name)
        print('id auther ' + self.auther)
        print('option name USI_Ponder type check default true')

    def set_sfen(self, sfen):
        d = {'B': reversi.BLACK_TURN, 'W': reversi.WHITE_TURN}
        line = sfen[0:65]
        turn_of = sfen[64]
        self.board = reversi.Board(line, d[turn_of])

    def position(self, sfen, moves):
        if sfen == 'startpos':
            self.board = reversi.Board()
        elif sfen[:5] == 'sfen ':
            self.set_sfen(sfen[5:])
        for i in range(len(moves)):
            self.board.move(reversi.move_from_str(moves[i]))
        return

    def score_scale_and_type(self):
        print('scoretype WP min 0 max 100')
        return

    def go(self):
        bestmove = np.random.choice(list(self.board.legal_moves))
        ponder_move = self.return_ponder_move(bestmove)
        return reversi.move_to_str(bestmove), reversi.move_to_str(ponder_move)

    def print_info(self, w):
        print('info score ' + str(np.random.randint(0, 100)) + ' string ' + w)
        return

    def return_ponder_move(self, move):
        board2 = self.board.copy()
        board2.move(move)
        return np.random.choice(list(board2.legal_moves))

if __name__ == '__main__':
    random = random()
    random.run()
