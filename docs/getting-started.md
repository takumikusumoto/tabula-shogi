# 導入・利用ガイド（Getting Started）

TabulaShogi をダウンロード・ビルドし、将棋GUI（ShogiHome や 将棋所）に登録して対局を始めるための手順書です。

---

## 1. 動作環境

- **OS**: Windows 10 / 11 (x86_64), Linux (x86_64)
- **ビルド環境**: Rust 1.98 以上 (Rust 2024 Edition / Cargo)
- **将棋GUI**: [ShogiHome](https://sunfish-shogi.github.io/shogihome/)、[将棋所](http://shogidokoro.starfree.jp/) 等の USI 準拠 GUI

---

## 2. ビルド手順

リポジトリをクローンまたはダウンロードし、以下のコマンドでリリースビルドを行います：

```bash
cargo build --release
```

ビルドが完了すると、以下のパスに実行ファイルが生成されます：

- **Windows**: `target/release/tabula-shogi.exe`
- **Linux**: `target/release/tabula-shogi`

---

## 3. 将棋GUIへの登録方法

### ShogiHome の場合

1. **ShogiHome** を起動します。
2. 画面右上の **「設定」** ➜ **「エンジン設定」** を開きます。
3. **「エンジンを追加」** をクリックします。
4. ビルドされた `tabula-shogi.exe` の絶対パスを指定します。
5. エンジン名が自動的に **TabulaShogi <version>**（例: `TabulaShogi 0.1.0`）、作者が **Takumi Kusumoto** と認識されます。
6. 設定を保存すれば、通常の対局や検討、棋譜解析で選択可能になります。

### 将棋所（Shogidokoro）の場合

1. **将棋所** を起動します。
2. メニューの **「対局」** ➜ **「エンジン管理」** を開きます。
3. **「追加」** ボタンを押し、 `tabula-shogi.exe` を選択します。
4. エンジン一覧に TabulaShogi が追加されたら完了です。

---

## 4. USI主要オプションの設定

対局GUIのエンジン設定画面から以下のパラメータを調整できます：

| オプション名 | 型 | デフォルト値 | 調整範囲 | 説明 |
| :--- | :---: | :---: | :---: | :--- |
| **USI_Hash** | spin | 64 | 1 〜 8192 MB | 置換表（探索結果のメモリキャッシュ）のサイズ。PCの搭載メモリに応じて 256 や 1024 に増やすと読みが安定します。 |
| **Threads** | spin | 1 | 1 〜 64 | 探索に使用する並列スレッド数（Lazy SMP）。対局時は 2〜8、ベンチマーク時は 1 を推奨。 |
| **Eval_Type** | combo | HCE | HCE / NNUE | 評価関数の種類。HCE（手動評価関数）または NNUE（ニューラルネット）を選択。 |
| **NNUE_File** | string | <empty> | ファイルパス | 外部の量子化 NNUE 重みバイナリ（`nnue.bin`）のパス。指定するとモデルが即座に読み込まれます。 |

---

## 5. コマンドラインからの直接実行テスト

ターミナルやコマンドプロンプトから対話的に起動することも可能です：

```bash
.\target\release\tabula-shogi.exe
usi
# -> id name TabulaShogi 0.1.0
# -> id author Takumi Kusumoto
# -> option name USI_Hash type spin default 64 min 1 max 8192
# -> option name Threads type spin default 1 min 1 max 64
# -> option name Eval_Type type combo default HCE var HCE var NNUE
# -> option name NNUE_File type string default <empty>
# -> usiok
isready
# -> readyok
quit
```

---

## 6. 自己対局＆学習パイプラインの実行

TabulaShogi は、スタンドアロン CLI ツールとして自己対局・機械学習を実行できます：

### 6.1 自己対局の実行 (`selfplay`)

```bash
# 4スレッド・深さ3で100局の自己対局を実行し、CSA棋譜と学習データセットを出力
.\target\release\tabula-shogi.exe selfplay --games 100 --threads 4 --depth 3 --csa games.csa --data train.tsv
```

### 6.2 手動評価パラメータの最適化 (`tune`)

```bash
# 自己対局データを用いて Texel Tuning (Adam) により評価パラメータを最適化
.\target\release\tabula-shogi.exe tune --data train.tsv --epochs 50 --lr 1.0
```

### 6.3 スクラッチ NNUE の学習 (`train-nnue`)

```bash
# 自己対局データからスクラッチ NNUE ネットワークを学習し、量子化バイナリを出力
.\target\release\tabula-shogi.exe train-nnue --data train.tsv --out nnue.bin --epochs 20 --lr 0.001
```

### 6.4 探索ベンチマークの測定 (`bench`)

```bash
# 探索ノード数および NPS (Nodes Per Second) を計測
.\target\release\tabula-shogi.exe bench
```
