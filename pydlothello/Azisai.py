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

square_weights = np.array([30, -12, 0, -1, -1, 0, -12, 30,
                                       -12, -15, -3, -3, -3, -3, -15, -12,
                                       0, -3, 0, -1, -1, 0, -3, 0,
                                       -1, -3, -1, -1, -1, -1, -3, -1,
                                       -1, -3,-1, -1, -1, -1, -3, -1,
                                       0, -3, 0, -1, -1, 0, -3, 0,
                                       -12, -15, -3, -3, -3, -3, -15, -12,
                                       30, -12, 0, -1, -1, 0, -12, 30])#マスの重み
square_weights += 15

class Azisai(BasePlayer):
    name = 'Azisai_v1'
    auther = 'Y_Hyodo'
    def __init__(self):#初期化
        super().__init__()
        self.MAX = 10000# +無限の代わり
        self.MIN = self.MAX * -1# -無限の代わり
        self.DRAW_V = 0#引き分けの値
        self.max_depth = DEF_MAX_DEPTH#探索深さ制限
        self.ordering_depth = int(self.max_depth / 2)
        self.NoSearch_table = {}
        self.ordering_moves_table = {}

    def usi(self):
        print('id name ' + self.name)
        print('id auther ' + self.auther)
        print('option name max_depth type spin default ' + str(DEF_MAX_DEPTH) + ' min 1 max 8')

    def usinewgame(self):
        self.NoSearch_table.clear()
        self.ordering_moves_table.clear()
        return
    
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

    def make_key(self, board):
        line = board.to_line()
        if board.turn:#黒番
            color = 'B'
        else:
            color = 'W'
        key = line + color
        return key

    def eval(self, board):#評価
        k =  self.make_key(board)
        if k in self.NoSearch_table.keys():
            return self.NoSearch_table[k]
        planes = np.empty((2, 8, 8), dtype=np.float32)
        board.piece_planes(planes)#石があるところは1,それ以外は0の配列がそれぞれの手番分得られる
        #以下、計算
        my_stones = planes[0].ravel()
        opponent_stones = planes[1].ravel()
        my = np.dot(my_stones, square_weights)
        opponent = np.dot(opponent_stones, square_weights)
        value = my - opponent
        self.NoSearch_table[k] = value
        return value

    def change(self, AB):
        return [-AB[1], -AB[0]]

    def return_winner(self):#手番側から見た数値が返ってくる
        my = self.board.piece_num()
        opponent = self.board.opponent_piece_num()
        if my == opponent:
            v = self.DRAW_V
        elif my > opponent:
            v = self.MAX
        else:
            v = self.MIN
        return v

    def ordering(self):#move ordering(手の並べ替え)
        key = self.make_key(self.board)
        if key in self.ordering_moves_table.keys():
            return self.ordering_moves_table[key]
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
        self.ordering_moves_table[key] = ordering_moves
        return ordering_moves

    def Search(self, depth, alpha_beta):
        if self.board.is_game_over():#終局
            return self.return_winner()
        if (depth >= self.max_depth):#評価
            return self.eval(self.board)
        max_score = self.MIN#最大スコアをリセット
        if depth <= self.ordering_depth:
            Next_moves = self.ordering()#並べ替え済み合法手を取得
        else:
            Next_moves = list(self.board.legal_moves)
        if 64 in Next_moves:
            self.push(64)
            result = self.Search(depth + 1, alpha_beta)
            self.pop()
            return result
        for i in range(len(Next_moves)):
            self.push(Next_moves[i])
            result = self.Search(depth + 1, self.change(alpha_beta)) * -1#自分目線の値
            if result >= alpha_beta[1]:#打ち切り
                self.pop()
                return result
            alpha_beta[0] = max([alpha_beta[0], result])#値を更新
            max_score = max([max_score, result])#値を更新
            self.pop()
        return max_score#最高スコアを返す

    def Search_main(self):
        #各種リセット
        self.board = self.root_board.copy()
        self.board_history = [self.board.copy()]
        self.moves = []
        alpha_beta = [self.MIN, self.MAX]
        print('info string no_search_score ' + str(self.eval(self.board)))
        #以下、メイン
        Next_moves = self.ordering()#並べ替え済み合法手を取得
        for i in range(len(Next_moves)):
            self.board = self.root_board.copy()
            self.board_history = [self.board.copy()]
            self.moves = []
            self.push(Next_moves[i])
            result = self.Search(1, self.change(alpha_beta)) * -1#自分目線にする
            if result >= alpha_beta[0]:#今までで最高の結果
                alpha_beta[0] = result#最高の結果を更新
                max_score = result
                bestmove = Next_moves[i]#最前手を更新
            print('info score ' + str(max_score) + ' pv ' + reversi.move_to_str(bestmove))#現在の最善手等を表示
            self.pop()
        return bestmove#最善手を返す

if __name__ == '__main__':
    azisai = Azisai()
    azisai.run()
