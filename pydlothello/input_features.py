import creversi as reversi
import numpy as np

"""
入力特徴量のバグは致命的なのに気づきにくいのでみんなはちゃんとテストしようね。
兵頭はWCSC32(第32回世界コンピュータ将棋選手権)直前に致命的なバグを見つけて絶望した経験がある。
"""

def make_feature(board):
    planes = np.empty((2, 8, 8), dtype=np.float32)
    board.piece_planes(planes[0])
    return planes.transpose((1, 2, 0))

def make_feature_for_train(board):#教師データ水増し用
    """
    やっていることは普通のこと(盤を回してデータの水増しをする)だが、
    クソコードなので参考にはするな
    """
    planes = np.empty((4, 2, 8, 8), dtype=np.float32)
    board.piece_planes(planes[0])
    board.piece_planes_rotate90(planes[1])
    board.piece_planes_rotate180(planes[2])
    board.piece_planes_rotate270(planes[3])
    return planes.transpose((0, 2, 3, 1))

if __name__ == '__main__':
    """
    スマートではないが、人間が見て確認することにしている
    """
    test_board = reversi.Board()
    print(test_board)
    test1 = make_feature(test_board)
    print(test1.shape)
    for i in range(8):
        print(test1[i][0], test1[i][1], test1[i][2], test1[i][3], test1[i][4], test1[i][5], test1[i][6], test1[i][7])
    test2 = make_feature_for_train(test_board)
    print(test2.shape)
    for d in range(4):
        print('==============')
        for i in range(8):
            print(test2[d][i][0], test2[d][i][1], test2[d][i][2], test2[d][i][3], test2[d][i][4],
              test2[d][i][5], test2[d][i][6], test2[d][i][7])
        print('==============')
