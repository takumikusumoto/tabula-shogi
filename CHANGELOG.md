# Changelog

All notable changes to TabulaShogi are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (`0.y.z` rules).

## [Unreleased]

## [0.2.0] - 2026-11-26

> **Note**: 本リリースの同梱バイナリは **HCE（手動評価関数）** が既定で動作します。HalfKP 評価エンジンおよび自律学習パイプライン（`loop` / `export-halfkp`）は完全内蔵されており、`models/best_halfkp.bin`（`TABU_HK2` 形式）を配置すると自動で読み込まれます。学習済み HalfKP 重みファイルは SPRT 昇格完了後に OS 非依存の別アセットとして配布予定です。

### Added

- **本格 HalfKP ニューラルネットワーク評価関数（204,120 特徴量 × 128 隠れ層 ClippedReLU）**: 差分アキュムレータ、Q6/Q9 固定小数点量子化（バイナリ形式 `TABU_HK2`）、スパース AdamW ストリーミング学習、検証付き `export-halfkp` サブコマンドを追加。
- **スクラッチ Residual Baseline NNUE（バイナリ形式 `TABU_NN5`）**: 自己対局データからの学習コマンド `train-nnue` および USI 動的切替（`Eval_Type`, `NNUE_File`, `HalfKP_File`）を追加。
- **自己対局・チューニング・アリーナ検定・自律改善パイプライン**: `selfplay`（CSA v2.2 / TSV）、`tune`（Texel Tuning / Adam）、`match`（先後交代ペア並列対戦＋SPRT 逐次検定・動的延長）、`loop`（自己対局 → 深読み再評価 → 学習 → アリーナ検定 → SPRT 昇格ゲート）を追加。
- **探索・詰み機能の強化**: df-pn 詰将棋探索および攻勢プリチェック、SEE 駒得オーダリング、Lazy SMP 並列探索、KIF 形式定跡読込（`use_book` 分離により自己対局では定跡を完全無効化）を追加。

### Changed

- **破壊的変更 — HalfKP バイナリ形式を `TABU_HK2` へ移行**: Q6/Q9 量子化対応に伴い旧 `TABU_HKP` 形式の読み込みを Fail-Closed で明示拒否するよう変更。
- **全永続化・モデル読込・自己対局の Fail-Closed 化**: アトミック書き出し＋`.bak` バックアップ、3 者／全ストレージパス衝突検証、8 バイトヘッダ先読みによる多態モデル解決、`LoopWorkspace` / `DatasetBatchReader` への統合を実施。
- **メモリ規律の徹底**: HalfKP モデルのストリーミング読込およびミニバッチ上限（`MAX_BATCH_SIZE = 4096`）により、実行時 RSS を 500 MiB 以内に維持。

### Fixed

- USI `Eval_Type` / `NNUE_File` 設定時に破損 NNUE ファイルでサイレントに初期重みへフォールバックしていた不具合を修正。
- 固定深さ探索・時間制限探索の中断時に未完了反復のスコアや定跡ゼロ値がラベル／置換表を汚染する不具合を修正。
- 親ディレクトリ無しパスでのデータセット追記失敗や、深読みプール永続化・読込エラーのサイレント無視を修正。

## [0.1.0] - 2026-09-12

- 初期開発版（Private リポジトリ分割前マイルストーン）：盤面表現、合法手・王手回避・打ち歩詰め判定、基本 PVS/反復深化探索、HCE 評価関数、USI プロトコル基盤。
