"""
このファイルでニューラルネットを作る
Tensorflowを使う
"""
import tensorflow as tf#2.2.0
import tensorflow.keras.layers as L

class NN:
  """
  シンプルな構造・シンプルな入力
  PolicyはMiniMax系の探索部でのmove ordering等でも使える
  """
  def __init__(self):
    self.input_shape = (8, 8, 2)#8 * 8の盤面が黒と白の分ある
    self.res_filters = 64#ResBlockのフィルター数
    self.res_blocks = 5#ResBlockの数
    self.dense_units = 64#全結合層のニューロン巣
    self.dense_layers = 3#全結合層の数
    self.output_policy = 60#ポリシー出力
    self.output_value = 1#バリュー出力
    
   def make(self):
    INPUT = L.Input(shape=self.input_shape)#入力
    x = L.Conv2D(self.res_filters, self.filter_size, padding='same', use_bias=False)(INPUT)
    x = L.BatchNormalization()(x)
    x = L.PReLU()(x)
    
    for i in range(self.res_blocks):
      #ResBlock始まり
      sc = x#ショートカットコネクション用
      x = L.Conv2D(self.res_filters, self.filter_size, padding='same', use_bias=False)(x)
      x = L.BatchNormalization()(x)
      x = L.PReLU()(x)
      x = L.Conv2D(self.res_filters, self.filter_size, padding='same', use_bias=False)(x)
      x = L.BatchNormalization()(x)
      x = L.Add([x, sc])#ショートカットコネクション
      x = L.PReLU()(x)
      #ResBlock終わり
     
    x = L.Flatten()(x)
    x = L.Dense(self.dense_units)(x)
    x = L.BatchNormalization()(x)
    x = L.PReLU()(x)
    
    #ここから分化する
    #ポリシーヘッド
    P = L.Dense(self.dense_units)(x)#ポリシーヘッドに接続
    P = L.BatchNormalization()(P)
    P = L.PReLU()(P)
    for i in range(self.dense_layers):
      P = L.Dense(self.dense_units)(P)
      P = L.BatchNormalization()(P)
      P = L.PReLU()(P)
    P = L.Dense(self.output)(P)
    P = L.Activation('softmax', name='policy')(P)#ポリシーヘッド終着点
    
    #バリューヘッド
    V = L.Dense(self.dense_units)(x)#バリューヘッドに接続
    V = L.BatchNormalization()(V)
    V = L.PReLU()(V)
    for i in range(self.dense_layers):
      V = L.Dense(self.dense_units)(V)
      V = L.BatchNormalization()(V)
      V = L.PReLU()(V)
    V = L.Dense(self.output_value)(V)
    V = L.Activation('tanh', name='value')(V)#バリューヘッド終着点
    
    #modelを作る
    model = tf.keras.models.Model(inputs=INPUT, outputs=[p, v])
    return model
