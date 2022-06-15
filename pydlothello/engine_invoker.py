from iroiro.Board_to_sfen import board_to_sfen
from USI_X_Engine import USI_X_Engine
import creversi as reversi
"""
自動で連戦してくれる
とりあえずデフォルトの設定で連戦させる
"""

class engine_invoker():
    def __init__(self):
        self.engine1 = USI_X_Engine()
        self.engine2 = USI_X_Engine()
        self.engine1.print_info = True
        self.engine2.print_info = True
        self.time_limits = [10, 10]

    def set_engine(self, path, options=[[], []]):
        self.engine1.Engine_path = path[0]
        self.engine2.Engine_path = path[1]
        self.engine1.options = options[0]
        self.engine2.options = options[1]
        return

    def reset(self):
        self.board = reversi.Board()
        self.moves = []
        self.engine1.NewGame()
        self.engine2.NewGame()

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

    def play(self):#1局分
        self.reset()
        c = 1
        moves = []
        while True:
            print('')
            print('===board===')
            print(self.board)
            print('is_black_turn:', self.board.turn)
            print('piece_sum:', self.board.piece_sum())
            print('==========')
            if c % 2 == 1:#黒番
                print('Black')
                move = self.engine1.go(board_to_sfen(self.board, c), [], self.time_limits[0], use_sfen=True)
                if 'resign' in move:
                    winner = 'W'
                    break
            else:#白番
                print('White')
                move = self.engine2.go(board_to_sfen(self.board, c), [], self.time_limits[1], use_sfen=True)
                if 'resign' in move:
                    winner = 'B'
            self.board.move_from_str(move)#打つ
            moves += [move]
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
        return winner

    def main(self, loops):
        wins = {'B': 0, 'D': 0, 'W': 0}
        for i in range(loops):
            print(loops, '戦中', i, '戦目')
            print('wins:', wins)
            result = self.play()
            wins[result] += 1
        print('連戦終了')
        print('engine1:', self.engine1.Engine_path)
        print('engine2:', self.engine2.Engine_path)
        print('wins:', wins)
        input('続ける:')
        return

if __name__ == '__main__':
    engine1 = input('エンジン1のpath:')
    engine2 = input('エンジン2のpath:')
    invoker = engine_invoker()
    invoker.set_engine([engine1, engine2])
    invoker.main(int(input('連戦回数: ')))
