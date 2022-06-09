from tensorflow.keras.models import load_model
from input_features import make_feature
import creversi as reversi
import numpy as np

class Endgame_AI():
    """
    最終盤の読み切りAI(激おそ)
    シンプルな設計
    """
    def __init__(self):#初期化
        self.model_file = './model/model_files/py-dlothello_model.h5'
        self.model = load_model(self.model_file)
        self.make_input_feature = make_feature#入力特徴量を作る関数
        self.ordering_depth = 4
        self.orded_moves_table = {}#すでに並べ替え済み

        self.MAX = 10000
        self.MIN = self.MAX * -1
        self.DRAW_V = 0

    def reset(self, board):#ボードなどをリセットする
        self.root_board = board#復旧などに使う
        self.board = self.root_board.copy()#主にこちらを操作する
        self.moves = []#2連パスの検知など(creversiが検知しない)
        if self.board.turn:
            t = reversi.BLACK_TURN
        else:
            t = reversi.WHITE_TURN
        self.board_history = [{'line': self.board.to_line(), 'turn_of': t}]
        return

    def push(self, move):#打つ
        self.board.move(move)
        if self.board.turn:
            t = reversi.BLACK_TURN
        else:
            t = reversi.WHITE_TURN
        self.board_history.append({'line': self.board.to_line(), 'turn_of': t})
        self.moves += [move]
        return

    def pop(self):#戻す
        self.board_history.pop(-1)
        self.board = reversi.Board(self.board_history[-1]['line'], self.board_history[-1]['turn_of'])
        self.moves.pop(-1)
        return

    def ordering(self, board):#ポリシー出力を使って手を並び変える
        K = board.to_line()
        if K in self.orded_moves_table.keys():
            return self.orded_moves_table[K]
        F = [self.make_input_feature(board)]
        policy = self.model.predict(np.array(F), batch_size=len(F))[0][0]
        Next_moves = list(board.legal_moves)
        output = {}
        for i in range(len(Next_moves)):
            output[Next_moves[i]] = policy[Next_moves[i]]
        output = sorted(output.items(), reverse=True, key=lambda x: x[1])
        orded_moves = []
        for i in range(len(output)):
            orded_moves.append(output[i][0])
        self.orded_moves_table[K] = orded_moves
        return orded_moves

    def is_double_pass(self):
        """
        すでに石が置かれている場所には置けないので
        最後の２手が同じかどうか調べるだけで2連パスを簡単に検知できる。
        """
        return len(self.moves) >= 2 and self.moves[-1] == self.moves[-2]

    def return_winner(self):
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
        if 64 in list(self.board.legal_moves):
            return self.return_winner()#仮の処理
        if now_depth <= self.ordering_depth:
            Next_moves = self.ordering(self.board)
        else:
            Next_moves = list(self.board.legal_moves)
        max_score = self.MIN
        for i in range(len(Next_moves)):
            self.push(Next_moves[i])
            result = self.search(now_depth + 1) * -1
            if result == self.MAX:#勝った
                self.pop()
                return self.MAX
            max_score = max([result, max_score])
            self.pop()
        return max_score

    def main(self, board):#メイン部
        self.reset(board)
        max_score = self.MIN - 10
        bestmove = None
        Next_moves = self.ordering(self.board)
        for i in range(len(Next_moves)):
            self.push(Next_moves[i])
            result = self.search(0) * -1
            if result >= max_score:
                max_score = result
                bestmove = Next_moves[i]
                if max_score == self.MAX:
                    print('info string I win.')
                    break
            self.pop()
        return reversi.move_to_str(bestmove)

if __name__ == '__main__':
    ed_ai = Endgame_AI()
    print('bestmove:', ed_ai.main(reversi.Board()))
