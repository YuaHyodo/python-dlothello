"""
USI-X対応エンジンを自分のプログラムで動かす際などにつかう

とりあえず動けばいい
"""
import subprocess as sp

class USI_X_Engine():
    def __init__(self):
        self.Engine_path = ''#エンジンファイルのパス
        self.options = []#ここにsetoptionコマンドを入れておけば自動でセットする
        self.print_info = True#infoコマンドの情報を表示するか？

    def command(self, word):#コマンドを送る
        self.engine.stdin.write(word + '\n')
        return

    def read(self, word):#コマンドを読む
        while True:
            line = self.engine.stdout.readline()
            if word in line:
                break
            if 'info' in line and self.print_info:
                print(line)
        return line

    def setup(self):#エンジンの起動とかをする
        self.engine = sp.Popen(self.Engine_path, stdin=sp.PIPE, stdout=sp.PIPE,
                          universal_newlines=True, bufsize=1)
        self.command('usi')
        self.read('usiok')
        for i in range(len(self.options)):#エンジンを設定する
            self.command(self.options[i])
        return

    def NewGame(self):#セットアップ等々をする
        self.setup()
        self.command('isready')
        self.read('readyok')
        self.command('usinewgame')
        return

    def go(self, sfen, moves, time_num):#メイン
        """
        sfen: sfenまたはstartposが入る
        moves: USI-Xオセロ版のmoveで表された手順
        time_num: 制限時間(秒)
        """
        m = ''
        for move in moves:
            m += ' '
            m += move
        self.command('position ' + sfen + ' moves' + m)
        
        to_engine = 'go btime ' + str(time_num * 1000) + ' wtime ' + str(time_num * 1000)
        self.command(to_engine)
        from_engine = self.read('bestmove')
        if 'resign' in from_engine:
            return 'resign'
        elif 'pass' in from_engine:
            return 'pass'
        move = from_engine[9] + from_engine[10]
        return move

    def Kill(self):#止める
        self.command('quit')
        return

if __name__ == '__main__':
    print('てすと')
    import creversi as reversi
    test = reversi.Board()
    e = USI_X_Engine()
    e.Engine_path = 'MCTS.bat'
    e.options = ['setoption name USI_Ponder value false',
                 'setoption name temperature value 1000']
    e.NewGame()
    moves = []
    while True:
        print(test)
        print(test.turn)
        print(test.piece_sum())
        if test.is_game_over() or (len(moves) >= 2 and moves[-1] == moves[-2]):
            break
        move = e.go('startpos', moves, 5)
        if move in ['resign']:
            break
        test.move_from_str(move)
        moves.append(move)
    e.Kill()
    print(test)
    print(test.turn)
    print(test.piece_sum())
