from othello_sfen_viewer.othello_sfen_viewer1 import simple_window
from othello_sfen_viewer.Board_to_sfen import board_to_sfen
from USI_X_Engine import USI_X_Engine
import creversi as reversi

class vs_AI:
    def __init__(self):
        self.ai = USI_X_Engine()
        self.ai.print_info = False
        self.time_limit = 10
        self.window = simple_window()

    def reset(self):
        self.board = reversi.Board()
        self.moves = []
        self.ai.NewGame()
        self.window.init_window()

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

    def human(self):
        sfen = board_to_sfen(self.board, 1)
        self.window.update_window(sfen)
        while True:
            move = input('手を入力:')
            if reversi.move_from_str(move) in list(self.board.legal_moves):
                break
            print('その手は合法手ではありません')
        return move

    def play(self, player):#1局分
        self.reset()
        c = 1
        moves = []
        while True:
            print('')
            
            print('Black')
            print('')
            print(self.board)
            print('')
            if player[0] == 'human':
                move = self.human()
            else:
                move = self.ai.go(board_to_sfen(self.board, c), [], self.time_limit, use_sfen=True)
                if 'resign' in move:
                    winner = 'W'
                    break
            self.board.move_from_str(move)#打つ
            moves += [move]
            if self.board.is_game_over() or (len(moves) >= 2 and moves[-1] == moves[-2]):#終局
                winner = self.return_winner(self.board)
                break
            
            print('White')
            print('')
            print(self.board)
            print('')
            if player[1] == 'human':
                move = self.human()
            else:
                move = self.ai.go(board_to_sfen(self.board, c), [], self.time_limit, use_sfen=True)
                if 'resign' in move:
                    winner = 'B'
                    break
            self.board.move_from_str(move)#打つ
            moves += [move]
            if self.board.is_game_over() or (len(moves) >= 2 and moves[-1] == moves[-2]):#終局
                winner = self.return_winner(self.board)
                break
            
            print('')
        print('終局')
        sfen = board_to_sfen(self.board, 1)
        self.window.update_window(sfen)
        print('===board===')
        print(self.board)
        print('is_black_turn:', self.board.turn)
        print('piece_sum:', self.board.piece_sum())
        print('手番側の石;', self.board.piece_num())
        print('==========')
        self.ai.Kill()
        return winner

    def main(self, loops, player):
        wins = {'B': 0, 'D': 0, 'W': 0}
        for i in range(loops):
            print(loops, '戦中', i, '戦目')
            print('wins:', wins)
            result = self.play(player)
            wins[result] += 1
        print('連戦終了')
        print('wins:', wins)
        input('続ける:')
        return

if __name__ == '__main__':
    vs = vs_AI()
    vs.ai.Engine_path =  input('USI-Xエンジンのpath:')
    games = int(input('ゲーム数:'))
    vs.time_limit = int(input('時間制限:'))
    if input('AI_先手(y/n):') == 'y':
        player = ['ai', 'human']
    else:
        player = ['human', 'ai']
    vs.main(games, player)
