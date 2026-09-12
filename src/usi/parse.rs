use crate::search::TimeControl;
use crate::types::Move;

#[derive(Debug, PartialEq)]
pub enum UsiCommand {
    Usi,
    IsReady,
    SetOption {
        name: String,
        value: String,
    },
    UsiNewGame,
    Position {
        sfen: Option<String>,
        moves: Vec<Move>,
    },
    Go(TimeControl),
    GoMate(Option<u64>),
    Stop,
    Quit,
    GameOver(String),
    Eval,
    Unknown(String),
}

impl UsiCommand {
    pub fn parse(line: &str) -> Self {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return UsiCommand::Unknown("".to_string());
        }

        let mut parts = trimmed.split_whitespace();
        let cmd = match parts.next() {
            Some(c) => c,
            None => return UsiCommand::Unknown("".to_string()),
        };

        match cmd {
            "usi" => UsiCommand::Usi,
            "isready" => UsiCommand::IsReady,
            "usinewgame" => UsiCommand::UsiNewGame,
            "stop" => UsiCommand::Stop,
            "quit" => UsiCommand::Quit,
            "eval" => UsiCommand::Eval,
            "gameover" => {
                let res = parts.next().unwrap_or("").to_string();
                UsiCommand::GameOver(res)
            }
            "setoption" => {
                // setoption name USI_Hash value 64
                let mut name = String::new();
                let mut value = String::new();
                let mut is_name = false;
                let mut is_value = false;

                for token in parts {
                    if token == "name" {
                        is_name = true;
                        is_value = false;
                    } else if token == "value" {
                        is_name = false;
                        is_value = true;
                    } else if is_name {
                        if !name.is_empty() {
                            name.push(' ');
                        }
                        name.push_str(token);
                    } else if is_value {
                        if !value.is_empty() {
                            value.push(' ');
                        }
                        value.push_str(token);
                    }
                }
                UsiCommand::SetOption { name, value }
            }
            "position" => {
                // position startpos [moves 7g7f 3c3d ...]
                // position sfen <sfen> [moves 7g7f ...]
                let kind = parts.next().unwrap_or("");
                let mut sfen_str = None;
                let mut moves = Vec::new();

                if kind == "startpos" {
                    // startpos
                    if let Some(moves_token) = parts.next()
                        && moves_token == "moves"
                    {
                        for m_str in parts {
                            if let Some(m) = Move::from_usi(m_str) {
                                moves.push(m);
                            }
                        }
                    }
                } else if kind == "sfen" {
                    // sfen board turn hand ply
                    let mut sfen_parts = Vec::new();
                    let mut reading_moves = false;

                    for token in parts {
                        if token == "moves" {
                            reading_moves = true;
                            continue;
                        }

                        if reading_moves {
                            if let Some(m) = Move::from_usi(token) {
                                moves.push(m);
                            }
                        } else {
                            sfen_parts.push(token);
                        }
                    }
                    sfen_str = Some(sfen_parts.join(" "));
                }

                UsiCommand::Position {
                    sfen: sfen_str,
                    moves,
                }
            }
            "go" => {
                let mut tc = TimeControl::default();
                let mut is_mate = false;
                let mut mate_limit = None;

                while let Some(sub) = parts.next() {
                    match sub {
                        "mate" => {
                            is_mate = true;
                            mate_limit = parts.next().and_then(|v| v.parse().ok());
                        }
                        "btime" => tc.btime = parts.next().and_then(|v| v.parse().ok()),
                        "wtime" => tc.wtime = parts.next().and_then(|v| v.parse().ok()),
                        "byoyomi" => tc.byoyomi = parts.next().and_then(|v| v.parse().ok()),
                        "binc" => tc.binc = parts.next().and_then(|v| v.parse().ok()),
                        "winc" => tc.winc = parts.next().and_then(|v| v.parse().ok()),
                        "infinite" => tc.infinite = true,
                        _ => {}
                    }
                }

                if is_mate {
                    UsiCommand::GoMate(mate_limit)
                } else {
                    UsiCommand::Go(tc)
                }
            }
            _ => UsiCommand::Unknown(trimmed.to_string()),
        }
    }
}
