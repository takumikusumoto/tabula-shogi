# TabulaShogi (`tabula-shogi`)

外部のモデル・重み・棋譜を一切持ち込まず、ゼロからの自己対局（Tabula Rasa: 白紙）によって進化する、完全スクラッチ開発の USI（Universal Shogi Interface）準拠将棋エンジンです。  
**ShogiHome** などの主要な将棋GUIに登録して、人間との対局や検討、AI同士の自己対局を行うことができます。

---

## 特徴

- **完全スクラッチ＆白紙からの自己進化（Tabula Rasa Philosophy）**:
  - やねうら王、Apery、Stockfish等の外部OSSコードや探索ルーチンの流用は一切行っていません。
  - 外部の学習済みNNUE重みファイル等に依存せず、ゼロベースで設計された駒得・Piece-Square Tables (PST)・玉の安全度・大局観（HCE）を採用。自己対局による強化学習パイプラインは Phase 3 で順次構築中。
- **超高速・堅牢なRust実装**:
  - 外部クレート依存ゼロ（Zero External Dependencies）。標準ライブラリのみで完結。
  - リリースビルド時で **250万〜300万 NPS (Nodes Per Second)** の超高速探索性能を実現。
- **本格的なゲーム木探索アルゴリズム**:
  - **Negamax + Alpha-Beta枝刈り**
  - **反復深化 (Iterative Deepening Search)** & **Aspiration Windows**
  - **静止探索 (Quiescence Search)**: 駒の取り合いや成り手を重点探索し、地平線効果を抑制
  - **置換表 (Transposition Table)**: ロックフリーAtomic 64-bitパッキング、Zobrist Hash活用
  - **Move Ordering**: 置換表の手、MVV-LVA、キラー手、応手（Countermove）、歴史ヒューリスティックによる探索枝刈り効率化
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

| コマンド                                                                    | 説明                                                                            |
| :-------------------------------------------------------------------------- | :------------------------------------------------------------------------------ |
| `usi`                                                                       | エンジン名・作者・設定オプションを出力し、`usiok` を返答                        |
| `isready`                                                                   | エンジンの初期化完了を確認し、`readyok` を返答                                  |
| `setoption name USI_Hash value <N>`                                         | 置換表（Transposition Table）のメモリサイズ（MB単位）を設定                     |
| `usinewgame`                                                                | 新規対局開始に伴う置換表および局面履歴のクリア                                  |
| `position [startpos \| sfen <SFEN>] moves ...`                              | 盤面局面のセットおよび着手履歴の適用                                            |
| `go [btime ...] [wtime ...] [byoyomi ...] [binc ...] [winc ...] [infinite]` | 指定の時間制限・秒読みルールで探索を開始し、定期的なPV infoと `bestmove` を出力 |
| `stop`                                                                      | 実行中の探索を即時中断し、直ちに現在の最善手を出力                              |
| `quit`                                                                      | プロセスを安全に終了                                                            |

---

## テストと品質管理

```bash
# 単体テストの実行（ルール適合、打ち歩詰め、二歩、王手回避、探索等の全テスト）
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
- [設計決定記録 (ADR)](docs/adr/): アーキテクチャ上の重要決定の履歴ログ

---

## ライセンス

[MIT License](LICENSE) © 2026 Takumi Kusumoto
