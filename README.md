# python-dlothello
- python-dlshogi2を改造して作ったUSI-Xオセロバージョンに対応したオセロAI。(未完成)<br> 
- ランダムプレーヤーの"random_kun"と基準AIの"Azisai"も同梱してある。<br> 
- クソコードアレルギーの人は見ないでください。<br>
- 関数名・変数名・実装方法等々に関するクレームは受け付けておりません。<br>

# 作者自己紹介
- 名前: 兵頭優空
- 年齢: 16歳(高校1年生)
- A.I. Ari Shogi、A.I. AN Shogiなどの将棋AIを開発していて、大会に出たりしている(戦績は悲惨)
- 将棋AI以外も色々作っている

# python-dlothelloの詳細
- python-dlshogi2というPythonで書かれた強い将棋AIを改造し、オセロに対応させた。<br>
- python-dlshohi2は、「強い将棋ソフトの創りかた」という、このリポジトリを見るような人は絶対に読むべき名著(個人の感想です)のサンプルコードなので、まだ読んでない人は買って読みましょう。<br>
- python-dlothelloは、やねうらお氏が提案しているUSI-Xプロトコルに対応したUSI-Xエンジンとして利用できる。
- USI-X(オセロ)の詳細な仕様についてはこちらを参照のこと: https://github.com/YuaHyodo/USI-X-protocol_othello_version
- USI-Xについてはこちらを参照のこと: http://yaneuraou.yaneu.com/2022/06/07/standard-communication-protocol-for-games
- 作者(兵頭)が作ったUSI-X(オセロ)の関連ツール等へのリンクは、下の[USI-X(オセロ)の関連リポジトリ]にまとめてみた。

# random_kunの詳細
- ランダムに手を選ぶUSI-Xエンジン。<br>
- デバッグとかに使える。<br>

# Azisaiの詳細
- 「マスの重みの評価 + シンプルαβ探索」という構成の指標として使えるUSI-Xエンジン。 <br>
- python-dlothelloの学習の進捗を測るために用意した。<br>
- マスの重みはリンク先のものを利用させて頂いた: https://uguisu.skr.jp/othello/5-1.html <br>

# その他のコンテンツ
- 学習済みモデルファイル
- Azisaiで生成した教師データ
- dlreversiという作者が別で作っているオセロAIで生成した教師データ
- その他にもたくさんあります。

# USI-X(オセロ)の関連リポジトリ
- USI-Xプロトコル(オセロ)の詳細な仕様等についての文書が置いてあるリポジトリ: https://github.com/YuaHyodo/USI-X-protocol_othello_version

- USI-Xプロトコル(オセロ)に関するサンプルプログラムのリポジトリ: https://github.com/YuaHyodo/USI-X_Othello_Samples

- オンライン対局用プロトコル、Online Othello Protocolのリポジトリ: https://github.com/YuaHyodo/online_othello_protocol

- Online Othello Protocol対応の簡易対局サーバー、Ari-Othello-Serverのリポジトリ: https://github.com/YuaHyodo/Ari-Othello-Server

- USI-XエンジンとOnline Othello Protocolサーバーを中継するGUIを持ったツール、Azisai_Othello_GUIのリポジトリ: https://github.com/YuaHyodo/Azisai_Othello_GUI

- USI-XエンジンをPythonプログラムから呼び出すためのツール、USI_X_Engine_Bridgeのリポジトリ: https://github.com/YuaHyodo/USI_X_Engine_Bridge


# その他・お知らせ等
- 間違いを見つけた場合や作者に連絡事項がある場合はGitHub内で連絡してください。(作者はtwitterアカウントなどは持っていないので)
- 慢性的にリソース(時間・お金・計算機など)が不足しているので支援してくれる人を募集しています。
