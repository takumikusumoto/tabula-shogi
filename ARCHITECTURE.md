# TabulaShogi Architecture & Design Guide

本書では、TabulaShogiの内部設計、モジュール構造、アルゴリズムの詳細、および将来の機能拡張方針について解説します。

---

## 1. モジュール分離と依存関係

```text
src/
├── main.rs          # USIループ起動用エントリポイント
├── lib.rs           # 各モジュールの公開と統合
├── types/           # 将棋の基礎型定義
│   ├── color.rs     # 先手・後手 (Black / White)
│   ├── piece.rs     # 駒種 (生駒, 成駒, 持ち駒, SFEN)
│   ├── square.rs    # 9x9マス (0..81, USI変換)
│   └── moves.rs     # 指し手 (通常移動, 駒打ち, 成り)
├── board/           # 盤面状態と履歴
│   ├── position.rs  # 盤面配列, 持ち駒, do/undo_move, 利き計算
│   └── zobrist.rs   # XorShift64による決定論的Zobrist Hash
├── book/            # 純粋な内蔵定跡
│   └── tree.rs      # 手書き定跡木（最大26手・探索時間0秒で着手）
├── movegen/         # 合法手生成
│   └── generator.rs # 移動手, 駒打ち手, 二歩/行き所なし除外, 打ち歩詰め判定
├── eval/            # スクラッチ評価関数
│   ├── evaluator.rs # 駒得, PST, 玉の囲い度, 手番ボーナス, 浮き駒ペナルティ
│   └── nnue.rs      # 完全スクラッチNNUE推論モジュール
├── search/          # ゲーム木探索
│   ├── engine.rs    # Negamax + Alpha-Beta, 反復深化, 静止探索, Null Move Pruning
│   ├── tt.rs        # 置換表 (Transposition Table)
│   ├── ordering.rs  # Move Ordering (TT, MVV-LVA, Killer, Countermove, History)
│   ├── see.rs       # 静的交換探索 (Static Exchange Evaluation)
│   ├── dfpn.rs      # df-pnアルゴリズムによる超高速詰将棋探索
│   └── time_mgr.rs  # 時間配分管理 (持ち時間, 秒読み)
├── selfplay/        # 完全自律型自己対局パイプライン
│   ├── config.rs    # 自己対局設定 (探索深さ, 乱数序盤手, スレッド数, 投了閾値)
│   ├── game.rs      # 1局の対局シミュレーション (詰み, 千日手, 最大手数)
│   ├── manager.rs   # マルチスレッド並行対局制御 & リアルタイム統計
│   ├── csa.rs       # 標準CSA形式 (Version 2.2) 棋譜シリアライザ
│   └── dataset.rs   # 機械学習用局面データセット記録 (SFEN, 評価値, 結果)
├── tune/            # スクラッチ強化学習・評価パラメータ自己最適化
│   ├── params.rs    # PST & 駒価値のパラメータベクトル表現
│   └── texel.rs     # Texel Tuning (Adamオプティマイザ / Elo勝率モデル)
└── usi/             # プロトコル通信層
    ├── parse.rs     # USIコマンドパース
    └── protocol.rs  # Lazy SMP並列探索スレッド制御, 入出力ループ
```

---

## 2. 盤面表現と利き計算

### 座標系

- 将棋盤は 9×9 = 81マス。
- 内部表現: `Square(0..=80)`
  - `file` (筋): 0 (1筋) 〜 8 (9筋)
  - `rank` (段): 0 (1段目/a) 〜 8 (9段目/i)
  - `index = file * 9 + rank`
- この設計により、同一筋上の連続アクセスや段ごとのオフセット計算が整数の乗加算で高速に行われます。

### 利き計算 (`attacks_to`)

- 飛び駒（香車、飛車、竜、角、馬）はレイスキャン方式を採用。
- 近接駒（歩、桂、銀、金、成駒、玉）はオフセットテーブルによる定数時間参照。
- 自玉の王手判定は、玉の位置に対して相手側の利きが存在するかを走査することで、極めて軽量に判定します。

---

## 3. 合法手生成の完全性

1. **疑似合法手の生成**:
   - 盤上の駒の利き先（味方の駒があるマスは除外）。
   - 持ち駒の空マスへの打込み（二歩チェック、行き所のないマスへの打込みチェック）。
   - 成りの強制チェック（1段目の歩・香、1〜2段目の桂）と任意成りの分岐。
2. **厳密な合法手フィルタリング**:
   - 候補手を実際に `pos.do_move()` して自玉が王手されていないか（自殺手・王手放置）を検証。
   - **打ち歩詰め（反則）の検出**:
     - 歩を打って王手になった場合、相手玉に逃げ手（回避可能な合法手）が1手でも存在するかをチェック。
     - 逃げ手が存在しない（即詰み）の場合は打ち歩詰めとして候補から除外。

---

## 4. 評価関数の設計 (スクラッチ数理モデル)

外部のNNUEやOSSの学習済み重みを一切使わない、完全スクラッチの評価関数です。

$$Score = (Material_{Black} - Material_{White}) + (PST_{Black} - PST_{White}) + (Safety_{Black} - Safety_{White}) + Tempo$$

1. **駒得 (Material)**:
   - 各駒の基礎価値 + 持ち駒ボーナス（盤上より自由度が高いため +15% 程度の加算）。
2. **Piece-Square Tables (PST)**:
   - 序盤での重要マス（7六、2六等の角道・飛先歩、中央の位取り）へのボーナス。
   - 玉の囲い位置（8八、2八等）への加点と、居玉（5九）放置への減点。
3. **玉の安全度 (King Safety)**:
   - 自玉の周囲8マスに存在する守備駒（金・銀・成駒）の枚数と連携度を評価。
4. **手番ボーナス (Tempo Bonus)**:
   - 手番側の主導権ボーナス (+25 cp) を付与。

---

## 5. 探索アルゴリズム

### 反復深化 (Iterative Deepening)

- 深さ1から開始し、深さを1ずつ上げながら最善手と評価値を更新。
- 設定された目標時間（Time Manager）に達した段階で深さ探索を打ち切り、確実に時間内に最善手を返答。

### 静止探索 (Quiescence Search)

- 深さ0の葉ノードにおいて、駒の取り合いや成り手のみを再帰探索。
- 地平線効果（大駒が取られる変化を直前で見失う現象）を防止。

### 置換表 (Transposition Table)

- XorShift64による決定論的Zobrist Hashキーを使用。
- 各ノードの評価値の種類（Exact, LowerBound, UpperBound）を区別して枝刈り。
- PV（Principal Variation）読み筋の復元にも利用。

### Move Ordering

- 置換表の手（TT Move）を最優先。
- MVV-LVA（Most Valuable Victim - Least Valuable Attacker: 捕獲される駒の価値 × 10 - 取る駒の価値）で駒取り手を優先。
- キラー手ヒューリスティックによるベータカット手の優先。

---

## 6. 開発ロードマップと機能拡張

- **自己対局強化学習パイプライン (Phase 3: 完了)**:
  - スタンドアロンCLI自己対局エンジン（`tabula-shogi selfplay`）の構築。
  - 外部棋譜・重みを一切使わず、ゼロから自己対局棋譜（CSA形式 Version 2.2）および学習用データセット（TSV形式）を自動生成・蓄積。
  - Adamオプティマイザによる **Texel Tuning** ソルバー（`tabula-shogi tune`）の実装。
- **自己対局生成データによるNNUE学習 (Phase 4)**:
  - 自己対局で得られた数万局の棋譜データを用いたスクラッチNNUE重みの学習と自動更新ループ。
  - 評価関数パラメータとNNUE推論のシームレスな統合。
