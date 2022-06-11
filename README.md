# python-dlothello
python-dlshogi2を改造して作ったUSI-Xオセロバージョンに対応したオセロAI。(未完成)<br> 
ランダムプレーヤーの"random_kun"と基準AIの"Azisai"も同梱してある。<br> 
クソコードアレルギーの人は見ないでください。<br>
関数名・変数名・実装方法等々に関するクレームは受け付けておりません。<br>
# python-dlothello詳細
python-dlshogi2というpythonで書かれた強い将棋AIを改造し、オセロに対応させた。<br>
やねうらお氏が提案しているUSI-Xプロトコルに対応したUSI-Xエンジンとして利用できる。(詳細な仕様は説明書.txtを見てください)<br>
USI-Xについてはこちらを参照のこと: http://yaneuraou.yaneu.com/2022/06/07/standard-communication-protocol-for-games/ <br>
# random_kun詳細
ランダムに手を選ぶUSI-Xエンジン。<br>
デバッグとかに使える。<br>
# Azisai詳細
「マスの重みの評価 + シンプルαβ探索」という構成の指標として使えるUSI-X対応エンジン。 <br>
python-dlothelloの学習の進捗を測るために用意した。<br>
マスの重みはリンク先のものを参考にした: https://uguisu.skr.jp/othello/5-1.html <br>
