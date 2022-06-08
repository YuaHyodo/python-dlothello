from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from model.NN import NN
import numpy as np

class Train:
    def __init__(self):
        self.epochs = 10#エポック数
        self.batch_size = 256#学習率
        self.nn = NN()

    def load_data(self, path):
        #まだ
        return

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
        data_set = load_data(data_path)#データを読み込む
        input_features = data_set['data']#入力
        policy_labels = data_set['policy']#ポリシー教師データ
        value_labels = data_set['value']#バリュー教師データ
        """
        #でばっぐ用
        input_features = np.random.uniform(1, -1, (1000, 8, 8, 2))
        policy_labels = np.random.uniform(1, -1, (1000, 64))
        value_labels = np.random.uniform(1, -1, (1000,))
        """
        model = self.nn.make()#未学習モデルを用意
        model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam')#コンパイル

        print('学習開始')
        history = model.fit(input_features, [policy_labels, value_labels], batch_size=self.batch_size,
                            epochs=self.epochs)#学習
        model.save(model_file)#せーぶ
        K.clear_session()
        print('学習完了')
        self.show_Graph(history)#グラフを表示
        return

if __name__ == '__main__':
    #まだ
    data_file = './data/train_data.bin'
    model_file = './model/model_files/py-dlothello_model.h5'
    train = Train()
    if input('reset(y/n):') in ['Y', 'y']:
        model = train.nn.make()
        model.save(model_file)
        print('完了')
    train.main(data_file, model_file)
    input('終了:')
