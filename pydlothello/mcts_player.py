"""
python-dlshogi2の改造
オセロに対応させ、兵頭好みの改造を施し、一部の機能を削った
USI-Xオセロ版対応(予定)

USI-Xの概要
やねうらお氏が提案する、対局ゲーム標準通信プロトコル
USIという将棋用の通信プロトコルがベースになっている
これを使うと、USI対応の将棋AI用のフレームワーク等の流用が簡単という大きなメリットがある

USI-Xについては、ここを参照のこと: http://yaneuraou.yaneu.com/2022/06/07/standard-communication-protocol-for-games/

上のページで「sfenをofenするなどの変更は絶対にしないでいただきたい」みたいなことが書いてあったり、
具体例の1つがオセロだったりするのは
我々が作っていたUOIプロトコルのせいだと思われる。(UOIではsfenにあたるものをofenと呼んでいた)
※やねうらお氏はブログに書き込む前に上記の意見を兵頭に直接伝えている。

我々が提案するUSI-Xのオセロ版の詳細な仕様ついては、ここを参照のこと: (まだ)
"""
import numpy as np
import creversi as reversi 

from base_player import BasePlayer
from uct.uct_node import UctNode
from input_features import make_feature
from tensorflow.keras.models import load_model

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

"""
以下では、パラメーターのデフォルト値を決定している

各種パラメータの説明(必要そうなものだけ)

DEFAULT_BATCH_SIZE: 1回の推論で何局面を同時に評価するかを決める。
値が大きいほど、1度に評価する局面が増える。
どの値が最適かはCPUとGPUのバランス等で決まるので各自で調べる必要がある。
慣習として2のN乗がよく使われるらしい。(1, 2, 4, 8, 16, 32, 64, 128, 256・・・)

==================================================

DEFAULT_C_PUCT: このパラメータの値を調整することで「利用」と「探索」のバランスを取る。
値が大きいほど「探索」を重視する(=広く浅く読む)

AlphaZeroなどではこの数字を探索の進み具合で動的に調整するようになっている。
本家dlshogi(探索部がC++のやつ)にも導入されていて、強くなることが確認されている。
気になる人は自分で導入してみてほしい。
=>参考: https://tadaoyamaoka.hatenablog.com/entry/2018/12/13/001953

==================================================

DEFAULT_TEMPERATURE: ボルツマン分布の「温度」というパラメータ
ポリシー出力にバラツキを加える。
値が大きい(=温度が高い)ほどバラツキが大きくなる(=広く浅く探索する)

学習用の教師データを生成する際は通常より温度を高めに設定することをお勧めする。
"""

# デフォルトバッチサイズ
DEFAULT_BATCH_SIZE = 32
# デフォルトPUCTの定数
DEFAULT_C_PUCT = 1.0
# デフォルト温度パラメータ
DEFAULT_TEMPERATURE = 1.0
# デフォルトPV表示間隔(ms)
DEFAULT_PV_INTERVAL = 500
# デフォルトプレイアウト数
DEFAULT_CONST_PLAYOUT = 1000
# 勝ちを表す定数（数値に意味はない）
VALUE_WIN = 10000
# 負けを表す定数（数値に意味はない）
VALUE_LOSE = -10000
# 引き分けを表す定数（数値に意味はない）
VALUE_DRAW = 20000
# キューに追加されたときの戻り値（数値に意味はない）
QUEUING = -1
# 探索を破棄するときの戻り値（数値に意味はない）
DISCARDED = -2
# Virtual Loss
VIRTUAL_LOSS = 3

# 温度パラメータを適用した確率分布を取得
def softmax_temperature_with_normalize(logits, temperature):
    # 温度パラメータを適用
    logits /= temperature

    # 確率を計算(オーバーフローを防止するため最大値で引く)
    max_logit = max(logits)
    probabilities = np.exp(logits - max_logit)

    # 合計が1になるように正規化
    sum_probabilities = sum(probabilities)
    probabilities /= sum_probabilities

    return probabilities

# ノード更新
def update_result(current_node, next_index, result):
    """
    ノードを更新する
    current_nodeは今のノード、
    next_indexは子ノードのインデックス、
    resultは探索結果

    現在のノードと子ノードの、累計訪問回数と
    累計価値に加算する。(バーチャルロスも考慮する)
    """
    current_node.sum_value += result
    current_node.move_count += 1 - VIRTUAL_LOSS
    current_node.child_sum_value[next_index] += result
    current_node.child_move_count[next_index] += 1 - VIRTUAL_LOSS

# 評価待ちキューの要素
class EvalQueueElement:
    def __init__(self, node, is_black_turn):
        self.node = node#nodeオブジェクト
        self.is_black_turn = is_black_turn#手番(Trueだと黒番)

class MCTSPlayer(BasePlayer):
    # USIエンジンの名前
    name = 'python-dlothello'
    #作者の名前
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
        # 評価待ちキュー
        self.eval_queue = None
        # バッチインデックス
        self.current_batch_index = 0

        # ルート局面
        self.root_board = reversi.Board()
        self.root_node = UctNode()

        # プレイアウト回数
        self.playout_count = 0
        # 中断するプレイアウト回数
        self.halt = None

        # バッチサイズ
        self.batch_size = DEFAULT_BATCH_SIZE

        # PUCTの定数
        self.c_puct = DEFAULT_C_PUCT
        # 温度パラメータ
        self.temperature = DEFAULT_TEMPERATURE
        # PV表示間隔
        self.pv_interval = DEFAULT_PV_INTERVAL

        self.debug = False

    def usi(self):
        print('id name ' + self.name)
        print('id auther ' + self.auther)
        print('option name USI_Ponder type check default false')
        print('option name modelfile type string default ' + self.DEFAULT_MODELFILE)
        print('option name batchsize type spin default ' + str(DEFAULT_BATCH_SIZE) + ' min 1 max 256')
        print('option name c_puct type spin default ' + str(int(DEFAULT_C_PUCT * 100)) + ' min 10 max 1000')
        print('option name temperature type spin default ' + str(int(DEFAULT_TEMPERATURE * 100)) + ' min 10 max 1000')
        print('option name pv_interval type spin default ' + str(DEFAULT_PV_INTERVAL) + ' min 0 max 10000')
        print('option name debug type check default false')

    def setoption(self, args):
        if args[1] == 'modelfile':
            self.modelfile = args[3]
        elif args[1] == 'batchsize':
            self.batch_size = int(args[3])
        elif args[1] == 'c_puct':
            self.c_puct = int(args[3]) / 100
        elif args[1] == 'temperature':
            self.temperature = int(args[3]) / 100
        elif args[1] == 'pv_interval':
            self.pv_interval = int(args[3])
        elif args[1] == 'debug':
            self.debug = args[3] == 'true'

    def score_scale_and_type(self):
        print('scoretype other min -100 max 100')
        return

    # モデルのロード
    def load_model(self):
        self.model = load_model(self.modelfile)
        return

    # 入力特徴量の初期化
    def init_features(self):
        """
        ニューラルネットに一気に複数の局面を送ったほうがGPUを効率的に使える
        python-dlshogi2などでは、キューに局面をためておいて、一度にたまった局面を評価することでそれを実現している

        αβ枝狩りなどと、局面をためて評価する工夫は相性がイマイチである
        Ari shogi(兵頭作の弱小将棋AI)のminimax系探索部(非公開)はその点を多少改善する工夫が施してあるので、
        余裕があればその技術を搭載した簡易探索部も公開する
        """
        self.eval_queue = []#評価待ちのノードが入る
        self.input_features = []#評価待ち局面の入力特徴量が入る
        return

    def isready(self):
        # モデルをロード・準備
        self.load_model()

        # 局面初期化
        self.root_board = reversi.Board()

        # 入力特徴量と評価待ちキューを初期化
        self.init_features()
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
        turn_of = sfen[65]
        self.board = reversi.Board(line, d[turn_of])
        return

    def position(self, sfen, usi_moves): 
        if sfen == 'startpos':#初期局面開始
            self.root_board = reversi.Board()
        elif sfen[:5] == 'sfen ':#sfenの局面開始
            self.set_sfen(sfen)

        moves = []
        for usi_move in usi_moves:#movesの後に続くmoveを再生
            move = self.root_board.move_from_str(usi_move)
            moves.append(move)
        self.root_node = UctNode()
        if self.debug:
            print(self.root_board)

    def set_limits(self, btime=None, wtime=None, byoyomi=None, binc=None, winc=None, infinite=False, ponder=False):
        """
        とりあえず仮のやつ
        """
        self.infinite_think = (infinite or ponder)
        self.STOP = False
        self.time_limit = 10
        return

    def go(self):
        # 探索開始時刻の記録
        self.begin_time = time.time()

        if len(list(self.root_board.legal_moves)) <= 1:#パスしか合法手がない
            return 'pass', None

        current_node = self.root_node

        # プレイアウト数をクリア
        self.playout_count = 0

        # ルートノードが未展開の場合、展開する
        if not current_node.is_expand:
            current_node.expand_node(self.root_board)

        # ルートノードが未評価の場合、評価する
        if current_node.policy is None:
            self.queue_node(self.root_board, current_node)
            self.eval_node()

        # 探索
        self.search()

        # 最善手の取得とPVの表示
        bestmove, bestvalue, ponder_move = self.get_bestmove_and_print_pv()

        # for debug
        if self.debug:
            for i in range(len(current_node.child_move)):
                print('{:3}:{:5} move_count:{:4} nn_rate:{:.5f} win_rate:{:.5f}'.format(
                    i, move_to_usi(current_node.child_move[i]),
                    current_node.child_move_count[i],
                    current_node.policy[i],
                    current_node.child_sum_value[i] / current_node.child_move_count[i] if current_node.child_move_count[i] > 0 else 0))

        return reversi.move_to_str(bestmove), reversi.move_to_str(ponder_move) if ponder_move else None

    def stop(self):
        # すぐに中断する
        self.STOP = True

    def ponderhit(self, last_limits):
        # 探索開始時刻の記録
        self.begin_time = time.time()
        self.last_pv_print_time = 0

        # プレイアウト数をクリア
        self.playout_count = 0

        # 探索回数の閾値を設定
        self.set_limits(**last_limits)

    def quit(self):
        self.stop()

    def is_win(self, board):
        """
        手番側が勝ったらTrue
        引き分けならNone
        負けたらFalseを返す
        """
        Sum = board.piece_sum()#合計石数
        my = board.piece_num()#手番側の石数
        if (Sum - my) == my:#引き分け
            return None
        if (Sum - my) < my:#勝ち
            return True
        return False

    def search(self):
        """
        探索は、以下のような手順で行われている
        1: ゲーム木を探索経路を記録しながら進み、評価する局面をキューにためる
        2: ニューラルネットで評価を行う
        3: ゲーム木の葉からrootに向かって評価値などを伝達する(ゲーム木を更新する)
        """
        self.last_pv_print_time = 0

        # 探索経路のバッチ
        trajectories_batch = []
        trajectories_batch_discarded = []

        # 探索回数が閾値を超える、または探索が打ち切られたらループを抜ける
        while True:
            trajectories_batch.clear()
            trajectories_batch_discarded.clear()
            self.current_batch_index = 0

            # バッチサイズの回数だけシミュレーションを行う
            for i in range(self.batch_size):
                # 盤面のコピー
                board = self.root_board.copy()

                # 探索
                trajectories_batch.append([])
                result = self.uct_search(board, self.root_node, trajectories_batch[-1])

                if result != DISCARDED:
                    # 探索回数を1回増やす
                    self.playout_count += 1
                else:
                    # 破棄した探索経路を保存
                    trajectories_batch_discarded.append(trajectories_batch[-1])
                    # 破棄が多い場合はすぐに評価する
                    if len(trajectories_batch_discarded) > self.batch_size // 2:
                        trajectories_batch.pop()
                        break

                # 評価中の葉ノードに達した、もしくはバックアップ済みため破棄する
                if result == DISCARDED or result != QUEUING:
                    trajectories_batch.pop()

            # 評価
            if len(trajectories_batch) > 0:
                self.eval_node()

            # 破棄した探索経路のVirtual Lossを戻す
            for trajectories in trajectories_batch_discarded:
                for i in range(len(trajectories)):
                    current_node, next_index = trajectories[i]
                    current_node.move_count -= VIRTUAL_LOSS
                    current_node.child_move_count[next_index] -= VIRTUAL_LOSS

            # バックアップ
            for trajectories in trajectories_batch:
                result = None
                for i in reversed(range(len(trajectories))):
                    current_node, next_index = trajectories[i]
                    if result is None:
                        # 葉ノード
                        result = current_node.child_node[next_index].value * -1
                    update_result(current_node, next_index, result)
                    result = result * -1

            # 探索を打ち切るか確認
            if self.check_interruption():
                return

            # PV表示
            if self.pv_interval > 0:
                elapsed_time = int((time.time() - self.begin_time) * 1000)
                if elapsed_time > self.last_pv_print_time + self.pv_interval:
                    self.last_pv_print_time = elapsed_time
                    self.get_bestmove_and_print_pv()

    # UCT探索
    def uct_search(self, board, current_node, trajectories):
        """
        resultの数字が本家と違ったり、
        「1.0 - result」と書いてあったところが「result * -1」になっていたりするのは、
        バリュー出力層の活性化関数が違うから。
        (本家はsigmoid、こちらはtanh)

        パス関連の部分はまだなので、とりあえずは負け扱いにしている
        """
        # 未展開なら展開
        if not current_node.is_expand:
            current_node.expand(self.board)
        # UCB値が最大の手を求める
        next_index = self.select_max_ucb_child(current_node)
        # 選んだ手を着手
        board.move(current_node.child_move[next_index])

        # Virtual Lossを加算
        current_node.move_count += VIRTUAL_LOSS
        current_node.child_move_count[next_index] += VIRTUAL_LOSS

        # 経路を記録
        trajectories.append((current_node, next_index))

        # ノードの展開の確認
        if current_node.child_node[next_index] is None:
            # ノードの作成
            child_node = current_node.create_child_node(next_index)

            if board.is_game_over():#終局チェック
                win = self.is_win(board)
                if win:#手番側が勝った
                    child_node.value = VALUE_WIN
                    result = -1
                elif win == None:#引き分け
                    child_node.value = VALUE_DRAW
                    result = 0
                else:#負けた
                    child_node.value = VALUE_LOSE
                    result = 1
            else:
                # 候補手を展開する
                child_node.expand_node(board)
                # 候補手がない場合
                if len(child_node.child_move) <= 1:
                    child_node.value = VALUE_LOSE
                    result = 1
                else:
                    # ノードを評価待ちキューに追加
                    self.queue_node(board, child_node)
                    return QUEUING
        else:
            # 評価待ちのため破棄する
            next_node = current_node.child_node[next_index]
            if next_node.value is None:
                return DISCARDED

            # 詰みと千日手チェック
            if next_node.value == VALUE_WIN:
                result = -1
            elif next_node.value == VALUE_LOSE:
                result = 1
            elif next_node.value == VALUE_DRAW:
                result = 0
            elif len(next_node.child_move) <= 0:
                result = 1
            else:
                # 手番を入れ替えて1手深く読む
                result = self.uct_search(board, next_node, trajectories)

        if result == QUEUING or result == DISCARDED:
            return result

        # 探索結果の反映
        update_result(current_node, next_index, result)

        return result * -1

    # UCB値が最大の手を求める
    def select_max_ucb_child(self, node):
        """
        注意: ここを変に改造するとかなり弱くなる(実験済み)
        """
        q = np.divide(node.child_sum_value, node.child_move_count,
            out=np.zeros(len(node.child_move), np.float32),
            where=node.child_move_count != 0)
        if node.move_count == 0:
            u = 1.0
        else:
            u = np.sqrt(np.float32(node.move_count)) / (1 + node.child_move_count)
        """
        下の式の解説
        
        1番目の項は「利用」である。
        比率が高いと、すでに勝率(実際は価値の合計)が高いノードが選ばれやすくなる
        
        2番目の項は「探索」である。
        比率が高いと、まだ選ばれていないノードが選ばれやすくなる
        ニューラルネットのpolicy出力も考慮している

        バランスの調整はc_puctというパラメータで行う
        """
        ucb = (q) + (self.c_puct * u * node.policy)

        return np.argmax(ucb)

    # 最善手取得とinfoの表示
    def get_bestmove_and_print_pv(self):
        # 探索にかかった時間を求める
        finish_time = time.time() - self.begin_time

        # 訪問回数最大の手を選択する
        current_node = self.root_node
        selected_index = np.argmax(current_node.child_move_count)

        # 選択した着手の勝率の算出
        bestvalue = current_node.child_sum_value[selected_index] / current_node.child_move_count[selected_index]
        score = int(bestvalue * 100)
        if score >= 100:
            score = 100
        if score <= -100:
            score = -100
        
        bestmove = current_node.child_move[selected_index]

        # PV
        pv = reversi.move_to_str(bestmove)
        ponder_move = None
        pv_node = current_node
        while pv_node.child_node:
            pv_node = pv_node.child_node[selected_index]
            if pv_node is None or pv_node.child_move is None or pv_node.move_count == 0:
                break
            selected_index = np.argmax(pv_node.child_move_count)
            pv += ' ' + reversi.move_to_str(pv_node.child_move[selected_index])
            if ponder_move is None:
                ponder_move = pv_node.child_move[selected_index]

        print('info nps {} time {} nodes {} score {} pv {}'.format(
            int(self.playout_count / finish_time) if finish_time > 0 else 0,
            int(finish_time),
            current_node.move_count,
            score, pv), flush=True)

        return bestmove, bestvalue, ponder_move

    # 探索を打ち切るか確認
    def check_interruption(self):
        """
        この関数がTrueを返すと探索を打ち切るので、それを考えて処理を書く。
        面倒なのでテキトーなやつを用意しておく
        """
        if self.STOP:#すぐに停止
            return True
        if self.infinite_think:#無制限に考えていい
            return False
        spend_time = int(time.time() - self.begin_time)
        if spend_time < self.time_limit:#まだ時間がある
            return False
        return True

    # 入力特徴量の作成
    def make_input_features(self, board):
        self.input_features.append(make_feature(board))
        return

    # ノードをキューに追加
    def queue_node(self, board, node):
        # 入力特徴量を作成
        self.make_input_features(board)
        # ノードをキューに追加
        self.eval_queue.append(EvalQueueElement(node, board.turn))

    # 推論
    def infer(self):
        nn_output = self.model.predict(np.array(self.input_features), batch_size=len(self.input_features))#推論
        return nn_output[0], nn_output[1]

    # 着手を表すラベル作成
    def make_move_label(self, move, color):
        return make_move_label(move, color)

    # 局面の評価
    def eval_node(self):
        # 推論
        policy_logits, values = self.infer()

        for i in range(len(values)):
            policy = policy_logits[i]
            value = values[i]
            current_node = self.eval_queue[i].node

            # 合法手一覧
            p = np.zeros(len(current_node.child_move))
            for j in range(len(current_node.child_move)):
                move = current_node.child_move[j]
                p[j] = policy[move]
                
            # ノードの値を更新
            current_node.policy = p
            current_node.value = float(value)

if __name__ == '__main__':
    player = MCTSPlayer()
    player.run()
