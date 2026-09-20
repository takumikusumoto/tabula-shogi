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
| **Eval_Type** | combo | HalfKP | HalfKP / HCE / NNUE | 評価関数の種類。HalfKP（スパースNNUE）、HCE（手動評価関数）、NNUE（小型残差NNUE）を選択。 |
| **HalfKP_File** | string | models/best_halfkp.bin | ファイルパス | HalfKP 評価重みバイナリのパス。存在する場合、自動読み込みされます。 |
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

### 6.4 アリーナ直接対戦＆SPRT検定 (`match`)

2つの異なるモデル（HCE vs NNUE、または世代の異なる `nnue.bin` 同士）を先後交代ペアマッチで並列対戦させ、勝率および SPRT（逐次確率比検定）による統計的有意性を判定します：

```bash
# HCE と学習済み NNUE の対局検定（20ペア = 全40局、4スレッド並列、探索深さ2）
.\target\release\tabula-shogi.exe match --engine1 HCE --engine2 nnue.bin --pairs 20 --threads 4 --depth 2

# 新旧 NNUE モデル同士のレーティング直接対戦
.\target\release\tabula-shogi.exe match --engine1 best_nnue.bin --engine2 candidate_nnue.bin --pairs 30 --threads 4 --depth 3
```

### 6.5 完全自律型自己改善ループ (`loop`)

「自己対局 ➜ IIZ深読み蒸留 ➜ HalfKP学習 ➜ アリーナ対決検定 ➜ 勝ち越し時の自動昇格」の進化サイクルを完全自動で指定世代数繰り返します：

```bash
# 5世代の自律改善ループを実行（1世代あたり1000局自己対局、20ペア検定、深さ2、6スレッド）
.\target\release\tabula-shogi.exe loop --iterations 5 --games 1000 --eval-pairs 20 --threads 6 --depth 2 --min-games 20

# またはPowerShellランチャースクリプトを使用
pwsh scripts/run_loop.ps1 -Iterations 5 -GamesPerIter 1000 -Threads 6
```

### 6.6 探索ベンチマークの測定 (`bench`)

```bash
# 初期局面からの深さ6探索によるノード数および NPS (Nodes Per Second) を計測
.\target\release\tabula-shogi.exe bench
```
