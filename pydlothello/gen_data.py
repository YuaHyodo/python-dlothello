from iroiro.Board_to_sfen import board_to_sfen
from USI_X_Engine import USI_X_Engine
from datetime import datetime
import creversi as reversi
import numpy as np
import pickle as pk
import time

"""
出てくる形式
{'sfen': [sfen], 'bestmove': [bestmove], 'winner': [winner]}
[sfen]にはsfenが
[bestmove]には実際に打たれたbestmove
[winner]は'B'(黒勝ち)か'W'(白勝ち)か'D'(引き分け)のいずれかが入る
"""

class gen_data():
    def __init__(self):
        self.dataset_num = 1000#生成するデータセットの数
        self.data_num = 50#1データセット当たりのデータ数(単位は局)
        self.start_random_move_num = 8#初手からこの手数までは絶対にランダムに行動
        self.think_time = 1#1手当たりの思考時間
        self.random_move = 5#ランダムムーブの割合を決める
        #self.set_engine()
        self.set_engine2()

    def set_engine(self):
        self.e1 = USI_X_Engine()
        self.e2 = USI_X_Engine()
        self.e1.print_info = True
        self.e2.print_info = True
        self.e1.Engine_path = 'MCTS.bat'
        self.e2.Engine_path = 'MCTS.bat'
        self.e1.options = ['setoption name USI_Ponder value false',
                 'setoption name temperature value 1000',
                           'setoption name endgame_search_on value 54',
                           'setoption name use_time value 3']
        self.e2.options = ['setoption name USI_Ponder value false',
                 'setoption name temperature value 1000',
                           'setoption name endgame_search_on value 54',
                           'setoption name use_time value 3']
        self.file_name = 'self_play'
        self.cool_time = 60#単位は秒

    def set_engine2(self):
        print('Playerをセット')
        self.e1 = USI_X_Engine()
        self.e2 = USI_X_Engine()
        self.e1.print_info = True
        self.e2.print_info = True
        self.e1.Engine_path = 'Azisai.bat'
        self.e2.Engine_path = 'Azisai.bat'
        self.e1.options = []
        self.e2.options = []
        self.file_name = 'Azisai'
        self.cool_time = 0

    def reset(self):#ボードとかをリセット
        self.board = reversi.Board()
        self.moves = []
        self.e1.NewGame()
        self.e2.NewGame()

    def save(self, data):#データをセーブ
        print('セーブ中・・・')
        now = datetime.now()
        path = './data/' + self.file_name + '{:04}{:02}{:02}{:02}{:02}{:02}.bin'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
        with open(path, 'wb') as f:
            pk.dump(data, f)
        print('セーブ完了')
        return
    
    def return_winner(self, board):#名前の通り
        Sum = board.piece_sum()
        my = board.piece_num()
        if (Sum - my) == my:#引き分け
            return 'D'
        if (Sum - my) > my:#自分負け
            if board.turn:#黒番
                return 'W'
            else:#白番
                return 'B'
        #以下、自分勝ち
        if board.turn:#黒番
            return 'B'
        return 'W'#白番

    def is_random(self):#ランダム行動をするか？
        return np.random.randint(0, 100) < self.random_move

    def play(self):#1局分
        self.reset()
        c = 1
        data = []
        moves = []
        while True:
            print('')
            print('===board===')
            print(self.board)
            print('is_black_turn:', self.board.turn)
            print('piece_sum:', self.board.piece_sum())
            print('==========')
            if c <= 8 or self.is_random():#ランダムムーブ
                print('random_move')
                move = np.random.choice(list(self.board.legal_moves))
                move = reversi.move_to_str(move)
            elif c % 2 == 1:#黒番
                print('Black')
                move = self.e1.go(board_to_sfen(self.board, c), [], self.think_time, use_sfen=True)
                if 'resign' in move:
                    winner = 'W'
                    break
            else:#白番
                print('White')
                move = self.e2.go(board_to_sfen(self.board, c), [], self.think_time, use_sfen=True)
                if 'resign' in move:
                    winner = 'B'
                    break
            data.append({'sfen': board_to_sfen(self.board, c),
                         'bestmove': move, 'winner': None})#データに加える
            self.board.move_from_str(move)#打つ
            moves += [move]#記録
            c += 1
            if c >= 64 or self.board.is_game_over() or (len(moves) >= 2 and moves[-1] == moves[-2]):#終局
                winner = self.return_winner(self.board)
                break
            print('')
        print('終局')
        print('===board===')
        print(self.board)
        print('is_black_turn:', self.board.turn)
        print('piece_sum:', self.board.piece_sum())
        print('手番側の石;', self.board.piece_num())
        print('==========')
        for i in range(len(data)):#winnerを書き込む
            data[i]['winner'] = winner
        output = []
        for i in range(len(data)):#passなどの条件を満たすデータを消す
            if data[i]['bestmove'] != 'pass' and i >= 8:
                output.append(data[i])
        return output

    def main(self, summer_mode=False):#メイン部
        for sets in range(self.dataset_num):
            data = []
            for i in range(self.data_num):
                print('#####################################')
                print(self.dataset_num, 'データセット中 |', sets, 'セット目')
                print(self.data_num, '局中 |', i, '局目')
                data.extend(self.play())
                if summer_mode:#夏のためのコード
                    print('夏モード')
                    print('PCが冷えるのを待機中・・・')
                    time.sleep(self.cool_time)
                    print('終わり')
                print('#####################################')
                print('')
            self.save(data)
        return

if __name__ == '__main__':
    """
    Q: summer_modeって何？
    A: PCが冷えるのを待つ時間
    連続で稼働するとPCの温度(主にGPU)が大変なことになるので設けている。(私の環境だと1分でも十分冷える)
    夏は個人的に一番好きな季節なのだが、AIの学習はしんどい。
    
    (時間がもったいないので)必要ない人はOFFにしよう。
    逆に、PCが燃えそうな人はtime.sleep()のカッコ内の数字を増やそう。
    """
    gen = gen_data()
    if input('self_play(y/n):') in ['n', 'N', 'no', 'No']:
        gen.set_engine2()
    else:
        gen.set_engine()
    gen.main(summer_mode=True)
