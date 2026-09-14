/// 自己対局の設定パラメータ
#[derive(Debug, Clone)]
pub struct SelfPlayConfig {
    /// 生成する総対局数
    pub num_games: usize,
    /// 並行実行スレッド数
    pub threads: usize,
    /// 探索深さ (0の場合は時間制限などを利用可能だが、基本は固定深さ)
    pub depth: u8,
    /// 多様性を生むための序盤ランダム着手手数
    pub random_opening_plies: usize,
    /// 1対局あたりの最大手数 (これを超えると引き分け)
    pub max_plies: usize,
    /// 連続してこの評価値以下になった場合に投了と判定する閾値 (cp)
    pub resign_threshold: i32,
    /// CSA形式棋譜の出力先ファイルパス (Noneの場合は出力しない)
    pub csa_output: Option<String>,
    /// 強化学習用局面データセットの出力先ファイルパス (Noneの場合は出力しない)
    pub data_output: Option<String>,
    /// スレッドあたりの置換表サイズ (MB)
    pub tt_size_mb: usize,
    /// 乱数シード値 (完全自律的乱数生成用)
    pub seed: u64,
    /// 評価関数の動作モード (HCE または NNUE)
    pub eval_mode: crate::eval::EvalMode,
    /// ソフトマックス温度サンプリングを行う手数上限 (これ以降は決定論的最善手)
    pub temperature_plies: usize,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        SelfPlayConfig {
            num_games: 10,
            threads: 1,
            depth: 3,
            random_opening_plies: 8,
            max_plies: 320,
            resign_threshold: -2500,
            csa_output: None,
            data_output: None,
            tt_size_mb: 16,
            seed: 0x9E3779B97F4A7C15, // 黄金比基底シード
            eval_mode: crate::eval::EvalMode::Hce,
            temperature_plies: 24,
        }
    }
}
