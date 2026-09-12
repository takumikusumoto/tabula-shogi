use crate::board::Position;
use crate::movegen::MoveGenerator;
use crate::types::Move;
use std::collections::HashMap;

pub const DFPN_INFINITY: u32 = 100_000_000;
pub const DFPN_DEFAULT_MAX_NODES: u32 = 300_000;

#[derive(Clone, Copy, Debug)]
pub struct DfpnNode {
    pub pn: u32,
    pub dn: u32,
    pub best_move: Option<Move>,
}

pub struct DfpnSolver {
    table: HashMap<u64, DfpnNode>,
    max_nodes: u32,
    nodes_searched: u32,
}

impl Default for DfpnSolver {
    fn default() -> Self {
        Self::new(DFPN_DEFAULT_MAX_NODES)
    }
}

impl DfpnSolver {
    pub fn new(max_nodes: u32) -> Self {
        DfpnSolver {
            table: HashMap::with_capacity(32768),
            max_nodes,
            nodes_searched: 0,
        }
    }

    /// 詰み探索のエントリポイント
    /// 詰みがあれば (true, Some(初回王手)) を返し、詰みがなければ (false, None) を返す
    pub fn solve(&mut self, pos: &mut Position) -> (bool, Option<Move>) {
        self.table.clear();
        self.nodes_searched = 0;

        let root_hash = pos.hash;
        self.mid(pos, DFPN_INFINITY, DFPN_INFINITY, true);

        if let Some(entry) = self.table.get(&root_hash)
            && entry.pn == 0
        {
            return (true, entry.best_move);
        }
        (false, None)
    }

    /// Multiple Iterative Deepening (MID) による証明数・反証数探索
    /// is_or_node: 王手をかける側（攻め方）が手番のとき true, 王手を外す側（受け方）が手番のとき false
    fn mid(&mut self, pos: &mut Position, target_pn: u32, target_dn: u32, is_or_node: bool) {
        self.nodes_searched += 1;
        if self.nodes_searched > self.max_nodes {
            return;
        }

        let hash = pos.hash;

        // 子節点の生成
        let moves = if is_or_node {
            // OR節点: 相手玉に王手をかける手のみ
            MoveGenerator::generate_checks(pos)
        } else {
            // AND節点: 自玉の王手を回避する手
            MoveGenerator::generate_evasions(pos)
        };

        // 終端状態の判定
        if moves.is_empty() {
            if is_or_node {
                // 王手がない = 攻め失敗 (反証)
                self.table.insert(
                    hash,
                    DfpnNode {
                        pn: DFPN_INFINITY,
                        dn: 0,
                        best_move: None,
                    },
                );
            } else {
                // 王手回避手がない = 詰み成立！ (証明)
                self.table.insert(
                    hash,
                    DfpnNode {
                        pn: 0,
                        dn: DFPN_INFINITY,
                        best_move: None,
                    },
                );
            }
            return;
        }

        // 千日手検出
        if pos.repetition_count() >= 3 {
            self.table.insert(
                hash,
                DfpnNode {
                    pn: DFPN_INFINITY,
                    dn: 0,
                    best_move: None,
                },
            );
            return;
        }

        // テーブルに初期エントリがなければ登録
        self.table.entry(hash).or_insert(DfpnNode {
            pn: 1,
            dn: 1,
            best_move: None,
        });

        loop {
            if self.nodes_searched > self.max_nodes {
                break;
            }

            // 子ノードの (pn, dn) を評価・集約
            let best_m;
            let mut current_pn: u32;
            let mut current_dn: u32;

            if is_or_node {
                // OR節点: pn = min(child.pn), dn = sum(child.dn)
                current_dn = 0;
                let mut best_child_move = moves[0];
                let mut min_child_pn = DFPN_INFINITY;
                let mut second_min_pn = DFPN_INFINITY;
                let mut min_child_pn_dn = 1;

                for &mv in &moves {
                    pos.do_move(mv);
                    let child_entry = self.table.get(&pos.hash).copied().unwrap_or(DfpnNode {
                        pn: 1,
                        dn: 1,
                        best_move: None,
                    });
                    pos.undo_move();

                    if child_entry.pn < min_child_pn {
                        second_min_pn = min_child_pn;
                        min_child_pn = child_entry.pn;
                        best_child_move = mv;
                        min_child_pn_dn = child_entry.dn;
                    } else if child_entry.pn < second_min_pn {
                        second_min_pn = child_entry.pn;
                    }

                    current_dn = current_dn.saturating_add(child_entry.dn);
                }
                current_pn = min_child_pn;
                best_m = Some(best_child_move);

                // 終了条件チェック
                if current_pn >= target_pn || current_dn >= target_dn || current_pn == 0 {
                    self.table.insert(
                        hash,
                        DfpnNode {
                            pn: current_pn,
                            dn: current_dn,
                            best_move: best_m,
                        },
                    );
                    break;
                }

                // 最善手の子ノードを展開 (目標証明数・反証数を設定)
                let sub_target_pn = target_pn.min(second_min_pn.saturating_add(1));
                let sub_target_dn =
                    target_dn.saturating_sub(current_dn.saturating_sub(min_child_pn_dn));

                pos.do_move(best_child_move);
                self.mid(pos, sub_target_pn, sub_target_dn, false);
                pos.undo_move();
            } else {
                // AND節点: pn = sum(child.pn), dn = min(child.dn)
                current_pn = 0;
                let mut min_child_dn = DFPN_INFINITY;
                let mut second_min_dn = DFPN_INFINITY;
                let mut best_child_move = moves[0];
                let mut min_child_dn_pn = 1;

                for &mv in &moves {
                    pos.do_move(mv);
                    let child_entry = self.table.get(&pos.hash).copied().unwrap_or(DfpnNode {
                        pn: 1,
                        dn: 1,
                        best_move: None,
                    });
                    pos.undo_move();

                    if child_entry.dn < min_child_dn {
                        second_min_dn = min_child_dn;
                        min_child_dn = child_entry.dn;
                        best_child_move = mv;
                        min_child_dn_pn = child_entry.pn;
                    } else if child_entry.dn < second_min_dn {
                        second_min_dn = child_entry.dn;
                    }

                    current_pn = current_pn.saturating_add(child_entry.pn);
                }
                current_dn = min_child_dn;
                best_m = Some(best_child_move);

                if current_pn >= target_pn || current_dn >= target_dn || current_dn == 0 {
                    self.table.insert(
                        hash,
                        DfpnNode {
                            pn: current_pn,
                            dn: current_dn,
                            best_move: best_m,
                        },
                    );
                    break;
                }

                let sub_target_dn = target_dn.min(second_min_dn.saturating_add(1));
                let sub_target_pn =
                    target_pn.saturating_sub(current_pn.saturating_sub(min_child_dn_pn));

                pos.do_move(best_child_move);
                self.mid(pos, sub_target_pn, sub_target_dn, true);
                pos.undo_move();
            }

            self.table.insert(
                hash,
                DfpnNode {
                    pn: current_pn,
                    dn: current_dn,
                    best_move: best_m,
                },
            );
        }
    }
}
