# 開発・コントリビューションガイド（Development Guide）

TabulaShogi の開発・テスト・品質基準に関するガイドラインです。

---

## 1. 開発の基本原則

TabulaShogi は **「外部のモデル・重み・定跡・棋譜を一切持ち込まず、ゼロからの自己生成で強くなる」** という厳格な基本理念のもとで開発されています。
開発にあたっては、必ず [docs/regulations.md](regulations.md) を事前にご確認ください。

---

## 2. 開発ワークフロー（Trunk-Based Development）

本プロジェクトは **トランクベース開発（Trunk-Based Development）** を採用しています：

- 変更は小さく分割し、直接 `main` ブランチにコミットしていきます。
- コミットメッセージは **Conventional Commits** 規約に従います：
  - `feat:` 新機能・アルゴリズム追加
  - `fix:` バグ修正
  - `perf:` 高速化・パフォーマンス改善
  - `refactor:` 動作を変えないリファクタリング
  - `docs:` ドキュメント整備
  - `chore:` ビルド・設定・雑務

---

## 3. テストと静的解析

コミット前に必ず以下のコマンドを実行し、エラーや警告がゼロであることを確認してください：

```bash
# 全テストスイート（全32件の統合・機能回帰テスト）の実行
cargo test

# 静的解析リンターの実行（警告はエラーとして扱う）
cargo clippy -- -D warnings

# フォーマッタの確認
cargo fmt -- --check
```

### テストスイート構成（全32件）

1. **`tests/integration_tests.rs` (16件)**: 基本将棋ルール、王手回避生成、二歩・打ち歩詰め判定、反復深化・PVS探索、置換表（TT）、SEE駒得オーダリング、df-pn詰将棋探索、USI通信プロトコル。
2. **`tests/phase3_selfplay_tune_tests.rs` (7件)**: 自己対局生成、CSA形式棋譜シリアライズ、SFEN変換往復性、Texel Tuning（Adam）損失収束テスト。
3. **`tests/phase4_nnue_tests.rs` (5件)**: スクラッチNNUEバックプロパゲーション学習器、16bit整数量子化、モデルバイナリ（`TABU_NN1`）読み書き往復性、USIオプション動的切り替え。
4. **`tests/phase5_loop_tests.rs` (4件)**: アリーナ先後ペア対戦、ランダム序盤局面生成、SPRT（逐次確率比検定）対数尤度比計算、自律的自己改善ループ（`LoopPipeline`）統合テスト。

※ 本リポジトリには Pre-commit Gitフックが設定されており、コミット時に `cargo fmt -- --check` および `markdownlint-cli2` による静的検証が自動実行されます。

---

## 4. バージョニング規則（Semantic Versioning）

バージョン管理は [SemVer 2.0.0](https://semver.org/lang/ja/)（0.x番台の規則）に準拠します：

- `0.y.0` (MINOR): 破壊的変更、大規模機能（自己対局パイプライン等）の導入
- `0.y.z` (PATCH): バグ修正、軽微な機能追加、ドキュメント更新
- `1.0.0` (MAJOR): プロ越えレベルの安定版完成

---

## 5. 開発ロードマップと現在地

- **Phase 1: 基本基盤（完了）**
  - 盤面表現、合法手生成、王手回避生成、打ち歩詰め判定、USIプロトコル通信ループ、HCE評価関数。
- **Phase 2: ゲーム木探索と品質基盤（完了）**
  - αβ探索（PVS/反復深化）、置換表（TT）、Lazy SMPマルチスレッド並列探索、df-pn詰将棋探索、SEEオーダリング、モダンOSSリポジトリ構造、Rust 2024 Edition、本格探索回帰テスト。
- **Phase 3: 自己対局＆自己学習パイプライン（完了）**
  - スタンドアロンCLI自己対局エンジン（`tabula-shogi selfplay`）。
  - 外部棋譜・重みを一切使わない自律的自己対局棋譜（CSA形式 v2.2）および学習用データセット（TSV形式）の生成・蓄積。
  - Adamオプティマイザによる評価パラメータ自己最適化（`tabula-shogi tune` / Texel Tuning）ループの構築。
- **Phase 4: 自己対局生成データによるスクラッチNNUE学習＆強化学習（完了）**
  - 自己対局データを用いたゼロ外部依存のスクラッチNNUEバックプロパゲーション学習器（`tabula-shogi train-nnue`）。
  - 16bit整数量子化（355 KB）・高速SIMD親和バイナリシリアライザ (`TABU_NN1`)。
  - USI動的切り替え（`Eval_Type` / `NNUE_File`）および探索エンジンへのシームレスな統合。
- **Phase 5: 自律的自己改善ループ & レーティング自動検定（完了）**
  - 先後交代ペア並列対戦アリーナ（`tabula-shogi match`）。
  - Wald の逐次確率比検定（SPRT）ソルバーによる統計的有意性の判定。
  - 自己対局 ➜ データ蓄積 ➜ NNUE 学習 ➜ アリーナ対決 ➜ モデル自動昇格（`tabula-shogi loop`）の完全自律パイプライン化。
- **次のマイルストーン: 自律強化学習による「プロ越えレベル」の達成（当面の目標）**
