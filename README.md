# TabulaShogi (`tabula-shogi`)

外部のモデル・重み・棋譜を一切持ち込まず、ゼロからの自己対局（Tabula Rasa: 白紙）によって進化する、完全スクラッチ開発の USI（Universal Shogi Interface）準拠将棋エンジンです。  
**ShogiHome** などの主要な将棋GUIに登録して、人間との対局や検討、AI同士の自己対局を行うことができます。

---

## 特徴

- **完全スクラッチ＆白紙からの自己進化（Tabula Rasa Philosophy）**:
  - やねうら王、Apery、Stockfish等の外部OSSコードや探索ルーチンの流用は一切行っていません。
  - 外部の学習済み重みファイル等に依存せず、ゼロベースで設計された駒得・Piece-Square Tables (PST)・玉の安全度・大局観（HCE）に加え、本格的な **HalfKP ニューラルネットワーク評価関数**（204,120 スパース特徴行 × 128 隠れ層 ClippedReLU、約2,613万重みパラメータ、IIZ 多重反復雑巾絞り蒸留、スパース AdamW ストリーミング学習）およびスクラッチ Residual Baseline NNUE を完全内蔵。
  - 自己対局生成データを用いたバックプロパゲーション＆Adamオプティマイザによる強化学習・評価関数自己進化パイプラインを完全内蔵。
- **超高速・堅牢なRust実装**:
  - 外部クレート依存ゼロ（Zero External Dependencies）。標準ライブラリのみで完結。
  - リリースビルド時で **250万〜300万 NPS (Nodes Per Second)** の超高速探索性能を実現。
- **本格的なゲーム木探索アルゴリズム**:
  - **Negamax + Alpha-Beta枝刈り**
  - **反復深化 (Iterative Deepening Search)** & **Aspiration Windows**
  - **静止探索 (Quiescence Search)**: 駒の取り合いや成り手を重点探索し、地平線効果を抑制
  - **置換表 (Transposition Table)**: ロックフリーAtomic 64-bitパッキング、Zobrist Hash活用
  - **Move Ordering**: 置換表の手、MVV-LVA、キラー手、応手（Countermove）、歴史ヒューリスティックによる探索枝刈り効率化
  - **マルチスレッド並列探索 (Lazy SMP)**: TT共有・探索ツリー分散によるスケーラビリティ
  - **千日手検出 (Repetition)**: 同一局面4回検知
- **将棋ルールの厳密な実装**:
  - 王手放置・自殺手の排除
  - 二歩の禁止
  - 行き所のない駒の配置禁止（1・2段目の香、1段目の桂/歩）
  - **打ち歩詰めの禁止**: 先手・後手双方の王手回避探索と連携した厳密判定
  - **王手回避 (Check Evasions) 専用生成器**: 王手されている局面での王手放置を物理的に防止
- **USIプロトコル完全準拠**:
  - リアルタイムな `info depth ... score cp ... nodes ... nps ... pv ...` の標準出力により、ShogiHome等のGUIで読み筋や評価値グラフ、消費時間がグラフィカルに表示されます。
  - `btime`, `wtime`, `byoyomi`, `binc`, `winc` に対応した安全な時間管理（Time Manager）。
  - `Eval_Type` (HalfKP / HCE / NNUE)、`HalfKP_File`、`NNUE_File` オプションによる動的評価エンジン切り替えに対応。

---

## ビルド方法

### 必要環境

- Rust 1.98 以上 (Rust 2024 Edition / Cargo)

### ビルドコマンド

```bash
# リリース最適化ビルド
cargo build --release
```

ビルドが完了すると、以下のパスに実行ファイルが生成されます：

- Windows: `target/release/tabula-shogi.exe`

---

## CLIモード & サブコマンド

TabulaShogi は、将棋GUIからの USI エンジン起動に加えて、自律的自己対局や機械学習チューニングを実行できるスタンドアロン CLI ツールチェーンを備えています：

```bash
# 1. デフォルト: USI プロトコル通信ループ (引数なし、または 'usi')
tabula-shogi
tabula-shogi usi

# 2. 自己対局パイプライン (CSA棋譜および学習用TSVの自動生成)
tabula-shogi selfplay --games 100 --threads 4 --depth 3 --csa games.csa --data train.tsv

# 3. 評価関数パラメータの自己最適化 (Texel Tuning / Adam)
tabula-shogi tune --data train.tsv --epochs 50 --lr 1.0

# 4. 自己対局データを用いたスクラッチ NNUE バックプロパゲーション学習
tabula-shogi train-nnue --data train.tsv --out nnue.bin --epochs 20 --lr 0.001

# 5. アリーナ対戦＆SPRT検定（2モデル間の先後ペア並列対戦と勝率・Elo差測定）
tabula-shogi match --engine1 hce --engine2 nnue.bin --pairs 20 --threads 4 --depth 2

# 6. 完全自律型自己改善ループ（自己対局 ➜ IIZ深読み蒸留 ➜ HalfKP学習 ➜ アリーナ検定 ➜ 自動昇格）
tabula-shogi loop --iterations 5 --games 1000 --eval-pairs 20 --threads 6 --depth 2 --min-games 20

# または専用ランチャースクリプトを使用
pwsh scripts/run_loop.ps1 -Iterations 5 -GamesPerIter 1000 -Threads 6

# 7. float AdamWチェックポイントから有倍率TABU_HK2モデルを再エクスポート
tabula-shogi export-halfkp --checkpoint models/candidate_halfkp_ckpt.bin --out models/candidate_halfkp.bin --samples 512

# 8. 探索ベンチマークの実行
tabula-shogi bench
```

---

## ShogiHomeへの登録方法

1. **ShogiHome** を起動します。
2. 画面上部またはメニューの **「設定」**（または **「エンジン設定」**）を開きます。
3. **「エンジンを追加」** をクリックします。
4. エンジンの実行ファイルとして、ビルドされた `tabula-shogi.exe` の絶対パスを指定します：
   - 例: `C:\path\to\tabula-shogi\target\release\tabula-shogi.exe`
5. エンジン名が自動的に `TabulaShogi <version>`（例: `TabulaShogi 0.1.0`）と認識されます。
6. 設定を保存すれば、通常の対局や検討、棋譜解析で選択可能になります。

---

## USIプロトコル対応コマンド一覧

| コマンド | 説明 |
| :--- | :--- |
| `usi` | エンジン名・作者・設定オプションを出力し、`usiok` を返答 |
| `isready` | エンジンの初期化完了を確認し、`readyok` を返答 |
| `setoption name USI_Hash value <N>` | 置換表（Transposition Table）のメモリサイズ（MB単位）を設定 |
| `setoption name Threads value <N>` | 並列探索スレッド数を設定 (1〜64) |
| `setoption name Eval_Type value <HalfKP\|HCE\|NNUE>` | 評価関数モードを切り替え (HalfKP: スパースNNUE, HCE: 手動評価関数, NNUE: 小型残差NNUE) |
| `setoption name HalfKP_File value <PATH>` | HalfKP評価重みバイナリ (`models/best_halfkp.bin`) を読み込み |
| `setoption name NNUE_File value <PATH>` | 外部量子化NNUE重みバイナリ (`nnue.bin`) を読み込み |
| `usinewgame` | 新規対局開始に伴う置換表および局面履歴のクリア |
| `position [startpos \| sfen <SFEN>] moves ...` | 盤面局面のセットおよび着手履歴の適用 |
| `go [btime ...] [wtime ...] [byoyomi ...] [binc ...] [winc ...] [infinite]` | 指定の時間制限・秒読みルールで探索を開始し、定期的なPV infoと `bestmove` を出力 |
| `stop` | 実行中の探索を即時中断し、直ちに現在の最善手を出力 |
| `eval` | 現在の局面の静的評価値（HCE内訳またはNNUEスコア）を出力 |
| `quit` | プロセスを安全に終了 |

---

## テストと品質管理

```bash
# 単体テストの実行（ルール適合、打ち歩詰め、二歩、王手回避、HalfKP、探索等の全テスト）
cargo test

# 静的解析リンターの実行
cargo clippy -- -D warnings

# フォーマッタの確認
cargo fmt -- --check
```

---

## ドキュメント

- [システムアーキテクチャ設計書](ARCHITECTURE.md): モジュール構造、盤面表現、探索アルゴリズム詳細
- [導入・利用ガイド](docs/getting-started.md): ビルド手順、ShogiHome / 将棋所への登録、USIオプション解説
- [開発・品質管理ガイド](docs/development.md): トランクベース開発、テスト、リンター、SemVer規則
- [開発レギュレーション（公式規定）](docs/regulations.md): 外部データ・モデル持ち込み禁止、完全スクラッチ原則
- [内蔵定跡ツリー解説書](docs/opening-book.md): 角換わり、相掛かり、矢倉、横歩取り、対抗形の全定跡分岐Mermaid図

---

## ライセンス

[MIT License](LICENSE) © 2026 Takumi Kusumoto
