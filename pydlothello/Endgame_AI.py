from tensorflow.keras.models import load_model
from iroiro.Board_to_sfen import board_to_sfen
from input_features import make_feature
import creversi as reversi
import numpy as np

class Endgame_AI():
    """
    最終盤の読み切りAI(激おそ)
    シンプルな設計

    概要
    1: Policyネットワークでmove orderingを行う
    2: 石数は気にしない。勝てればいい
    3: パスが絡むと激よわ
    """
    def __init__(self):#初期化
        self.model_file = './model/model_files/py-dlothello_model.h5'
        self.model = load_model(self.model_file)#ニューラルネット
        self.make_input_feature = make_feature#入力特徴量を作る関数
        self.ordering_depth = 4#スタートしてからこのDepthまでしかmove orderingはしない
        self.ordering_moves_table = {}#すでに並べ替え済み
        self.ordering2_moves_table = {}

        self.MAX = 10000#最大を表す(+無限の代わり)
        self.MIN = self.MAX * -1#最小を表す(-無限の代わり)
        self.DRAW_V = 0#引き分けの値

    def reset(self, board):#ボードなどをリセットする
        self.root_board = board#復旧などに使う
        self.board = self.root_board.copy()#主にこちらを操作する
        self.moves = []#2連パスの検知など(creversiが検知しない)
        if self.board.turn:
            t = reversi.BLACK_TURN
        else:
            t = reversi.WHITE_TURN
        self.board_history = [{'line': self.board.to_line(), 'turn_of': t}]#手を戻す時に使う
        return

    def push(self, move):#打つ
        self.board.move(move)
        if self.board.turn:
            t = reversi.BLACK_TURN
        else:
            t = reversi.WHITE_TURN
        self.board_history.append({'line': self.board.to_line(), 'turn_of': t})#記録しておく
        self.moves += [move]#記録その2
        return

    def pop(self):#戻す
        self.board_history.pop(-1)#戻す
        self.board = reversi.Board(self.board_history[-1]['line'], self.board_history[-1]['turn_of'])#盤を戻す
        self.moves.pop(-1)#戻す
        return

    def make_K(self, board):
        return board_to_sfen(board, 0)

    def ordering(self, board):#ポリシー出力を使って手を並び変える
        K = self.make_K(board)
        if K in self.ordering_moves_table.keys():#すでに登録されているか？
            return self.ordering_moves_table[K]#されているなら登録されている値を返す
        Next_moves = list(board.legal_moves)#合法手のリストを用意
        if len(Next_moves) == 1:
            return Next_moves
        F = [self.make_input_feature(board)]#入力特徴量を用意
        policy = self.model.predict(np.array(F), batch_size=len(F))[0][0]#推論
        #以下、スマートじゃない方法で並び替え
        output = {}
        for i in range(len(Next_moves)):
            output[Next_moves[i]] = policy[Next_moves[i]]
        output = sorted(output.items(), reverse=True, key=lambda x: x[1])
        ordering_moves = []
        for i in range(len(output)):
            ordering_moves.append(output[i][0])
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

    def is_double_pass(self):
        """
        すでに石が置かれている場所には置けないので
        最後の２手が同じかどうか調べるだけで2連パスを簡単に検知できる。
        """
        return len(self.moves) >= 2 and self.moves[-1] == self.moves[-2]

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

    def search(self, now_depth):
        if self.board.is_game_over() or self.is_double_pass():#終局
            return self.return_winner()
        if now_depth <= self.ordering_depth:#move orderingをするか？
            Next_moves = self.ordering(self.board)# move orderingする
        else:#しない
            #Next_moves = list(self.board.legal_moves)#並び変えてないlistを使う
            Next_moves = self.ordering2(self.board)
        if 64 in Next_moves:#パス
            self.push(64)
            result = self.search(now_depth + 1)
            self.pop()
            return result
        max_score = self.MIN#これまでの最大値
        for i in range(len(Next_moves)):
            self.push(Next_moves[i])#打つ
            result = self.search(now_depth + 1) * -1#さらに深く読む
            if result == self.MAX:#勝った
                self.pop()
                return self.MAX#それ以上望ましい結果はないので打ち切る
            max_score = max([result, max_score])#最高値を更新
            self.pop()#戻す
        return max_score#max_scoreはDRAW_VかMINしかとらない

    def main(self, board):#メイン部
        self.reset(board)#リセット・準備
        max_score = self.MIN
        bestmove = None
        Next_moves = self.ordering(self.board)#並び替え済み合法手を用意
        for i in range(len(Next_moves)):
            self.push(Next_moves[i])
            result = self.search(0) * -1#調べる・自分視点にする
            if result >= max_score:#最高値更新
                max_score = result
                bestmove = Next_moves[i]
                if max_score == self.MAX:#勝ち確
                    print('info string I win.')#勝ち確メッセージを出力
                    break#打ち切る
            self.pop()
        return reversi.move_to_str(bestmove)

if __name__ == '__main__':
    from iroiro.make_random_board import make_random_board
    ed_ai = Endgame_AI()
    board = make_random_board(48)
    print(board)
    print('black_turn:', board.turn)
    print('bestmove:', ed_ai.main(board))
