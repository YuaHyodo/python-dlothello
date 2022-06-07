import time
from concurrent.futures import ThreadPoolExecutor

"""
このBasePlayerを使ってプレーヤーを作ると便利
"""

class BasePlayer:
    def __init__(self):#初期化
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None

    def usi(self):#usiコマンドが来た時に動く部分
        pass

    def usinewgame(self):#usinewgameコマンドが来た際に動く部分
        pass
    
    def score_scale_and_type(self):
        """
        GUIなどが評価値のスケールや形式などをエンジンに問い合わせるための拡張コマンド
        形式は[WP(勝率), stone(石の差), other(その他)]から、
        スケールは最小値と最大値を返すことになっている。
        
        返答は
        scoretype [形式] min [最小値] max [最大値]になる
        例: scoretype WP min 0 max 100
        """
        pass

    def setoption(self, args):#setoptionコマンドが来た際に動く部分
        pass

    def isready(self):#isreadyコマンドが来た際に動く部分
        pass

    def position(self, sfen, usi_moves):#positionコマンドが来た際に動く部分
        pass

    def set_limits(self, btime=None, wtime=None, byoyomi=None, binc=None, winc=None, nodes=None, infinite=False, ponder=False):
        """
        goコマンドについてくる持ち時間設定から、今回の手番で使う時間を決める関数
        """
        pass

    def go(self):#goコマンドが来た時に動く関数
        pass

    def stop(self):#stopコマンドが来た際に動く関数
        pass

    def ponderhit(self, last_limits):#ponderhitコマンドが来た際に動く関数
        pass

    def quit(self):#quitコマンドが来た際に動く関数
        pass

    def run(self):#メイン
        while True:
            cmd_line = input().strip()
            cmd = cmd_line.split(' ', 1)

            if cmd[0] == 'usi':
                self.usi()
                print('usiok', flush=True)
                
            elif cmd[0] == 'setoption':
                option = cmd[1].split(' ')
                self.setoption(option)
                
            elif cmd[0] == 'score_scale_and_type':
                self.score_scale_and_type()
                
            elif cmd[0] == 'isready':
                self.isready()
                print('readyok', flush=True)
                
            elif cmd[0] == 'usinewgame':
                self.usinewgame()
                
            elif cmd[0] == 'position':
                args = cmd[1].split('moves')
                self.position(args[0].strip(), args[1].split() if len(args) > 1 else [])
                
            elif cmd[0] == 'go':
                kwargs = {}
                if len(cmd) > 1:
                    args = cmd[1].split(' ')
                    if args[0] == 'infinite':
                        kwargs['infinite'] = True
                    else:
                        if args[0] == 'ponder':
                            kwargs['ponder'] = True
                            args = args[1:]
                        for i in range(0, len(args) - 1, 2):
                            if args[i] in ['btime', 'wtime', 'byoyomi', 'binc', 'winc', 'nodes']:
                                kwargs[args[i]] = int(args[i + 1])
                self.set_limits(**kwargs)
                # ponderhitのために条件と経過時間を保存
                last_limits = kwargs
                need_print_bestmove = 'ponder' not in kwargs and 'infinite' not in kwargs

                def go_and_print_bestmove():
                    bestmove, ponder_move = self.go()
                    if need_print_bestmove:
                        print('bestmove ' + bestmove + (' ponder ' + ponder_move if ponder_move else ''), flush=True)
                    return bestmove, ponder_move
                
                self.future = self.executor.submit(go_and_print_bestmove)
                
            elif cmd[0] == 'stop':
                need_print_bestmove = False
                self.stop()
                bestmove, _ = self.future.result()
                print('bestmove ' + bestmove, flush=True)
                
            elif cmd[0] == 'ponderhit':
                last_limits['ponder'] = False
                self.ponderhit(last_limits)
                bestmove, ponder_move = self.future.result()
                print('bestmove ' + bestmove + (' ponder ' + ponder_move if ponder_move else ''), flush=True)
                
            elif cmd[0] == 'quit':
                self.quit()
                break
