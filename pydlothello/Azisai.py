from base_player import BasePlayer
import creversi as reversi
import numpy as np
import time

"""
名前の由来: アジサイ(植物) 英語だとHydrangea
HydrangeaはしっくりこないのでAzisaiになった

理由: 今日(6/11)のbingの写真がアジサイのきれいな写真だったから(1日考えてもいい名前が思いつかなかった)
"""

DEF_MAX_DEPTH = 4#最大Depth

square_weights = np.array([[30, -12, 0, -1, -1, 0, -12, 30],
                                       [-12, -15, -3, -3, -3, -3, -15, -12],
                                       [0, -3, 0, -1, -1, 0, -3, 0],
                                       [-1, -3, -1, -1, -1, -1, -3, -1],
                                       [-1, -3,-1, -1, -1, -1, -3, -1],
                                       [0, -3, 0, -1, -1, 0, -3, 0],
                                       [-12, -15, -3, -3, -3, -3, -15, -12],
                                       [30, -12, 0, -1, -1, 0, -12, 30]])

class Azisai(BasePlayer):
    name = 'Azisai_v1'
    auther = 'Y_Hyodo'
    def __init__(self):
        super().__init__()
        self.MAX = 10000
        self.MIN = self.MAX * -1
        self.DRAW_V = 0
        self.max_depth = DEF_MAX_DEPTH

    def usi(self):
        print('id name ' + self.name)
        print('id auther ' + self.auther)
        print('option name max_depth type spin default ' + str(DEF_MAX_DEPTH) + ' min 1 max 8')

    def setoption(self, args):
        if args[1] == 'max_depth':
            self.max_depth = int(args[3])

    def score_scale_and_type(self):
        print('scoretype other min -100 max 100')
        return

    def set_sfen(self, sfen):
        #sfen => creversiのBoard
        d = {'B': reversi.BLACK_TURN, 'W': reversi.WHITE_TURN}
        line = sfen[0:65]
        turn_of = sfen[64]
        self.root_board = reversi.Board(line, d[turn_of])
        return

    def position(self, sfen, usi_moves):
        if sfen == 'startpos':#初期局面開始
            self.root_board = reversi.Board()
        elif sfen[:5] == 'sfen ':#sfenの局面開始
            self.set_sfen(sfen[5:])
        for usi_move in usi_moves:#movesの後に続くmoveを再生
            move = self.root_board.move_from_str(usi_move)

    def go(self):
        # 探索開始時刻の記録
        self.begin_time = time.time()

        if 64 in list(self.root_board.legal_moves):#パスしか合法手がない
            return 'pass', None

        # 最善手の取得
        bestmove = self.Search_main()

        return reversi.move_to_str(bestmove), None#ponder非対応

    def push(self, move):#打つ
        self.board.move(move)
        self.board_history.append(self.board.copy())#記録しておく
        self.moves += [move]#記録その2
        return

    def pop(self):#戻す
        self.board_history.pop(-1)#戻す
        self.board = self.board_history[-1].copy()#盤を戻す
        self.moves.pop(-1)#戻す
        return

    def eval(self, board):
        planes = np.empty((2, 8, 8), dtype=np.float32)
        board.piece_planes(planes)
        my = planes[0] * square_weights
        opponent = (planes[1] * square_weights) * -1
        my = sum(my.reshape((64,)))
        opponent = sum(opponent.reshape((64,)))
        return my + opponent

    def change(self, AB):
        return [-AB[1], -AB[0]]

    def return_winner(self):
        my = self.board.piece_num()
        opponent = self.board.opponent_piece_num()
        if my == opponent:
            v = self.DRAW_V
        elif my > opponent:
            v = self.MAX
        else:
            v = self.MIN
        return v

    def ordering(self):
        legal_moves = list(self.board.legal_moves)
        output = {}
        for i in range(len(legal_moves)):
            self.push(legal_moves[i])
            output[legal_moves[i]] = self.eval(self.board)
            self.pop()
        output = sorted(output.items(), reverse=False, key=lambda x: x[1])
        ordering_moves = []
        for i in range(len(output)):
            ordering_moves.append(output[i][0])
        return ordering_moves

    def Search(self, depth, alpha_beta):
        if self.board.is_game_over():
            return self.return_winner()
        if (depth >= self.max_depth) or (64 in list(self.board.legal_moves)):
            return self.eval(self.board)
        max_score = self.MIN
        Next_moves = self.ordering()
        for i in range(len(Next_moves)):
            self.push(Next_moves[i])
            result = self.Search(depth + 1, self.change(alpha_beta)) * -1
            if result >= alpha_beta[1]:
                self.pop()
                return result
            alpha_beta[0] = max([alpha_beta[0], result])
            max_score = max([max_score, result])
            self.pop()
        return max_score

    def Search_main(self):
        self.board = self.root_board.copy()
        self.board_history = [self.board.copy()]
        self.moves = []
        #以下、メイン
        alpha_beta = [self.MIN, self.MAX]
        Next_moves = self.ordering()
        for i in range(len(Next_moves)):
            self.push(Next_moves[i])
            result = self.Search(1, self.change(alpha_beta)) * -1
            if result >= alpha_beta[0]:
                alpha_beta[0] = result
                bestmove = Next_moves[i]
            print('info score ' + str(alpha_beta[0]) + ' pv ' + reversi.move_to_str(bestmove))
            self.pop()
        return bestmove

if __name__ == '__main__':
    azisai = Azisai()
    azisai.run()