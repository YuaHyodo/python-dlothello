import numpy as np
import creversi as reversi

from base_player import BasePlayer
from input_features import make_feature
from tensorflow.keras.models import load_model
from iroiro.Board_to_sfen import board_to_sfen

from Endgame_AI import Endgame_AI

import time
import math

#==================
#邪魔な表示を消す
import tensorflow as tf
import warnings
import os
warnings.simplefilter('ignore')
os.environ[ 'TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel('ERROR')
#==================

#デフォルト必勝読み開始石数
DEFAULT_ENDGAME_SEARCH_ON = 48
#1手当たりの基本的な使用時間(秒)
DEF_USE_TIME = 3

def sort_dict(d, reverse=True):
    d2 = sorted(d.items(), reverse=reverse, key=lambda x: x[1])
    output = []
    for i in range(len(d2)):
        output.append(d2[i][0])
    return output

class MiniMaxPlayer(BasePlayer):
    name = 'python-dlothelo-minimax'
    auther = 'Y_Hyodo'
    #デフォルトのモデルファイルのパス
    DEFAULT_MODELFILE = './model/model_files/py-dlothello_model.h5'

    def __init__(self):
        super().__init__()
        # モデルファイルのパス
        self.modelfile = self.DEFAULT_MODELFILE
        # モデル
        self.model = None
        # 入力特徴量
        self.features = None
        # ルート局面
        self.root_board = reversi.Board()
        #読み切り探索部への切り替えスイッチ
        self.endgame_search_on = DEFAULT_ENDGAME_SEARCH_ON
        #
        self.use_time = DEF_USE_TIME

        self.ordering_moves_table = {}#すでに並べ替え済み
        self.ordering2_moves_table = {}

        self.MAX = 10000#最大を表す(+無限の代わり)
        self.MIN = self.MAX * -1#最小を表す(-無限の代わり)
        self.DRAW_V = 0#引き分けの値

        self.window_size = 0.01

    def usi(self):
        print('id name ' + self.name)
        print('id auther ' + self.auther)
        print('option name USI_Ponder type check default false')
        print('option name modelfile type string default ' + self.DEFAULT_MODELFILE)
        print('option name use_time type spin default ' + str(DEF_USE_TIME) + ' min 1 max 1000')
        print('option name endgame_serach_on type spin default ' + str(DEFAULT_ENDGAME_SEARCH_ON) + ' min 32 max 64')

    def setoption(self, args):
        if args[1] == 'modelfile':
            self.modelfile = args[3]
        elif args[1] == 'use_time':
            self.use_time = int(args[3])
        elif args[1] == 'endgame_search_on':
            self.endgame_search_on = int(args[3])
    
    def score_scale_and_type(self):
        print('scoretype other min -100 max 100')
        return

    # モデルのロード
    def load_model(self):
        self.model = load_model(self.modelfile)
        return

    def isready(self):
        # モデルをロード・準備
        self.load_model()

        self.endgame_ai = Endgame_AI()#終盤特化aiの準備

        # 局面初期化
        self.root_board = reversi.Board()
        return

    def set_sfen(self, sfen):
        """
        想定しているsfenの形式
        (UOIのofenの形式)
        空きマス: -, 黒石: X, 白石: Oで長さ64の盤を表す部分に
        手番を表すB(黒番), W(白番)をつけた、長さ65の配列
        その後ろに手数を書いてもいいと思う。(プログラムは無視する)
        """
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

        moves = []
        for usi_move in usi_moves:#movesの後に続くmoveを再生
            move = self.root_board.move_from_str(usi_move)
            moves.append(move)
        return

    def set_limits(self, btime=None, wtime=None, byoyomi=None, binc=None, winc=None, infinite=False, ponder=False):
        """
        とりあえず仮のやつ
        """
        if btime != None:
            btime /= 1000
        if wtime != None:
            wtime /= 1000
        if byoyomi != None:
            byoyomi /= 1000
        if binc != None:
            binc /= 1000
        if winc != None:
            winc /= 1000
        self.infinite_think = (infinite or ponder)
        self.STOP = False
        self.time_limit = self.use_time
        if self.root_board.turn:#黒番
            if btime < self.time_limit:
                self.time_limit = int(btime / 2)
            if binc != None:
                self.time_limit += (binc - 0.5)
        else:
            if wtime < self.time_limit:
                self.time_limit = int(wtime / 2)
            if winc != None:
                self.time_limit += (winc - 0.5)
        if byoyomi != None:
            self.time_limit += (byoyomi - 0.5)
        print('info string time_limit ' + str(self.time_limit))
        return

    def go(self):
        # 探索開始時刻の記録
        self.begin_time = time.time()

        if self.root_board.turn:
            t = reversi.BLACK_TURN
        else:
            t = reversi.WHITE_TURN
        self.board_history = [{'line': self.root_board.to_line(), 'turn_of': t}]#記録しておく

        if 64 in list(self.root_board.legal_moves):#パスしか合法手がない
            return 'pass', None

        if self.root_board.piece_sum() >= self.endgame_search_on:#読み切る
            print('info string mode endgame_ai')
            move = self.endgame_ai.main(self.root_board.copy())
            return move, None

        self.stop_OK = False
        bestmove = self.search_main()

        return reversi.move_to_str(bestmove), None

    def stop(self):
        # すぐに中断する
        self.STOP = True

    def ponderhit(self, last_limits):
        # 探索開始時刻の記録
        self.begin_time = time.time()
        self.last_pv_print_time = 0

        # 探索回数の閾値を設定
        self.set_limits(**last_limits)

    def quit(self):
        self.stop()

    def push(self, move):#打つ
        self.board.move(move)
        if self.board.turn:
            t = reversi.BLACK_TURN
        else:
            t = reversi.WHITE_TURN
        self.board_history.append({'line': self.board.to_line(), 'turn_of': t})#記録しておく
        return

    def pop(self):#戻す
        self.board_history.pop(-1)#戻す
        self.board = reversi.Board(self.board_history[-1]['line'], self.board_history[-1]['turn_of'])#盤を戻す
        return

    def make_K(self, board):
        return board_to_sfen(board, 0)

    def is_stop(self):#探索を打ち切るか？
        if not self.stop_OK:
            return False
        spend_time = time.time() - self.begin_time
        if (spend_time + 1) >= self.time_limit:
            return True
        return False

    def return_winner(self):
        #手番側から見た数字を返す
        Sum = self.board.piece_sum()
        my = self.board.piece_num()
        if (Sum - my) == my:
            return self.DRAW_V
        elif (Sum - my) < my:
            return self.MAX
        else:
            return self.MIN
        return

    def change(self, AB):#AlphaBetaの値をひっくり返す
        return [-AB[1], -AB[0]]

    def ordering(self, board):#ポリシー出力を使って手を並び変える
        K = self.make_K(board)#Key
        if K in self.ordering_moves_table.keys():#すでに登録されているか？
            return self.ordering_moves_table[K]#されているなら登録されている値を返す
        Next_moves = list(board.legal_moves)#合法手のリストを用意
        if len(Next_moves) == 1:
            return Next_moves
        F = [make_feature(board)]#入力特徴量を用意
        policy = self.model.predict(np.array(F), batch_size=len(F))[0][0]#推論
        #以下、スマートじゃない方法で並び替え
        output = {}
        for i in range(len(Next_moves)):
            output[Next_moves[i]] = policy[Next_moves[i]]
        ordering_moves = sort_dict(output, reverse=True)
        self.ordering_moves_table[K] = ordering_moves#次回のために登録する
        return ordering_moves

    def ordering2(self, board):#角を先に調べる
        K = self.make_K(board)
        if K in self.ordering_moves_table.keys():
            return self.ordering_moves_table[K]
        if K in self.ordering2_moves_table.keys():
            return self.ordering2_moves_table[K]
        legal_moves = list(board.legal_moves)
        if len(legal_moves) == 1:
            return legal_moves
        yusen = [0, 7, 56, 63]
        Next_moves = []
        for i in range(len(yusen)):
            if yusen[i] in legal_moves:
                Next_moves.append(yusen[i])
                legal_moves.remove(yusen[i])
        Next_moves += legal_moves
        self.ordering_moves_table[K] = Next_moves
        return Next_moves

    def eval(self, board):
        """
        ここが肝要
        兵頭作のAI Ari shogi ver1のMiniMax系探索部に搭載されている技術。
        合法手の数が多いほどGPUをうまく使える
        (ので、オセロではあまり効果がない)
        末端での評価の部分を書き換えるだけで実装できるのが良いポイントだと個人的に思う
        (実装が楽なのはいいことだと思う)
        """
        Next_moves = list(self.board.legal_moves)
        features = []
        for i in range(len(Next_moves)):
            b = board.copy()
            b.move(Next_moves[i])
            features.append(make_feature(b))
        pv = self.model.predict(np.array(features))
        values = pv[1]
        return max(values[0]) * -1

    def search(self, depth, AlphaBeta):
        if self.board.is_game_over():#終局
            return self.return_winner()
        if self.is_stop():
            return 'stop'
        if depth >= self.max_depth:#評価
            return self.eval(self.board)
        if depth <= int(self.max_depth / 4) or depth < 2:
            Next_moves = self.ordering(self.board)
        else:
            #Next_moves = list(self.board.legal_moves)
            Next_moves = self.ordering2(self.board)
        if 64 in Next_moves:
            self.push(64)
            result = self.search(depth + 1, AlphaBeta)
            self.pop()
            return result
        max_score = self.MIN#これまでの最大値
        for i in range(len(Next_moves)):
            self.push(Next_moves[i])#打つ
            result = self.search(depth + 1, self.change(AlphaBeta))#さらに深く読む
            if self.is_stop():
                self.pop()
                return 'stop'
            result *= -1
            if result == self.MAX:#勝った
                self.pop()
                return self.MAX#それ以上望ましい結果はないので打ち切る
            if result >= AlphaBeta[1]:#打ち切り
                self.pop()
                return result
            AlphaBeta[0] = max([AlphaBeta[0], result])
            max_score = max([result, max_score])#最高値を更新
            self.pop()#戻す
        return max_score

    def search_main(self):
        self.board = self.root_board.copy()
        Next_moves = self.ordering(self.board)
        self.max_depth = 2
        WIN = False
        bestmove = None
        bestscore = None
        while True:
            AlphaBeta = [self.MIN, self.MAX]
            self.board = self.root_board.copy()
            bestmove_kari = None
            bestscore_kari = self.MIN
            O = {}
            for i in range(len(Next_moves)):
                self.push(Next_moves[i])
                result = self.search(1, self.change(AlphaBeta))
                if self.is_stop():
                    break
                result *= -1
                if result == self.MAX:
                    print('info string I win.')
                    self.pop()
                    bestmove = Next_moves[i]
                    WIN = True
                    break
                if result >= AlphaBeta[0]:
                    AlphaBeta[0] = result
                    bestmove_kari = Next_moves[i]
                    bestscore_kari = result
                O[Next_moves[i]] = result
                self.pop()
                if WIN:
                    bestmove = bestmove_kari
                    bestscore = bestscore_kari
                    break
            if self.is_stop() or WIN:
                break
            bestmove = bestmove_kari
            bestscore = bestscore_kari
            print('info depth ' +  str(self.max_depth) + ' score ' + str(bestscore) + ' pv ' + reversi.move_to_str(bestmove))
            Next_moves = sort_dict(O)
            self.stop_OK = True
            self.max_depth += 1
        return bestmove

if __name__ == '__main__':
    player = MiniMaxPlayer()
    player.run()
