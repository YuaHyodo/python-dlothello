from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from input_features import make_feature_for_train as make_feature
import matplotlib.pyplot as plt
import creversi as reversi
from model.NN import NN
import numpy as np
import pickle as pk
import time

class Train:
    def __init__(self):
        self.epochs = 10#エポック数
        self.batch_size = 512#バッチサイズ
        self.nn = NN()
        self.summer_mode_callback = LambdaCallback(on_epoch_end=lambda epoch,logs: time.sleep(60))#PCが冷えるのを待機

    def load_data(self, path):
        """
        データを読む
        """
        d = {'B': reversi.BLACK_TURN, 'W': reversi.WHITE_TURN}
        d2 = {'B': 1, 'W': -1, 'D': 0}
        with open(path, 'rb') as f:
            data = pk.load(f)
        features = None
        label_policy = []
        label_value = []
        for i in range(len(data)):
            if i % 1000 == 0:
                print(len(data), 'データ中', i, '番目まで処理')
            sfen = data[i]['sfen']
            line = sfen[0:65]
            turn_of = sfen[64]
            board = reversi.Board(line, d[turn_of])
            bestmove = data[i]['bestmove']
            winner = data[i]['winner']
            policy = np.zeros(64)
            policy[reversi.move_from_str(bestmove)] = d2[turn_of] * d2[winner]#負けた時にとった行動は-1、勝った時は1
            fea = make_feature(board)
            if type(features) == type(None):
                features = fea
            else:
                features = np.concatenate((features, fea))
            for i in range(4):
                label_policy.append(policy)
                label_value.append(d2[turn_of] * d2[winner])
        return features, np.array(label_policy), np.array(label_value)

    def show_Graph(self, history):#グラフを表示
        history = history.history
        plt.plot(history['loss'], label='Train_loss')
        plt.plot(history['policy_loss'], label='Train_policy_loss')
        plt.plot(history['value_loss'], label='Train_value_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc='best')
        plt.show()
        return

    def main(self, data_path, model_file):
        F, P, V = self.load_data(data_path)#データを読み込む
        print('data_shape:', F.shape, P.shape, V.shape)
        input_features = F#入力
        policy_labels = P#ポリシー教師データ
        value_labels = V#バリュー教師データ
        """
        #でばっぐ用
        input_features = np.random.uniform(1, -1, (1000, 8, 8, 2))
        policy_labels = np.random.uniform(1, -1, (1000, 64))
        value_labels = np.random.uniform(1, -1, (1000,))
        """
        model = load_model(model_file)
        model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam')#コンパイル

        print('学習開始')
        history = model.fit(input_features, [policy_labels, value_labels], batch_size=self.batch_size,
                            epochs=self.epochs, callbacks=[self.summer_mode_callback])#学習
        model.save(model_file)#せーぶ
        K.clear_session()
        print('学習完了')
        self.show_Graph(history)#グラフを表示
        return

if __name__ == '__main__':
    data_file = input('教師データのファイル:')
    model_file = './model/model_files/py-dlothello_model.h5'
    train = Train()
    if input('reset(y/n):') in ['Y', 'y']:
        model = train.nn.make()
        model.save(model_file)
        print('完了')
    train.main(data_file, model_file)
    input('終了:')
