// SparseSolver — Copyright (C) 2026 Jerzy Slaby (DeepBorys).
// Uses the postflop-solver engine by Wataru Inariba; see NOTICE.
// SPDX-License-Identifier: AGPL-3.0-or-later
// Modified for SparseSolver inference and benchmarks; see NOTICE and LICENSE.

#![allow(dead_code)] // Shared by several small example entry points.
//! Bind the published board/hand K=1 mapping to this game's betting histories.
use postflop_solver::*;

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

#[derive(Clone)]
pub(self) struct Decision {
    pub(self) id: u64,
    history: Vec<usize>,
    pub(self) key: String,
    pub(self) board: String,
    pub(self) signature: String,
    pub(self) player: usize,
    pub(self) flop_frontier: Option<u64>,
    pub(self) river_frontier: Option<u64>,
}

pub(self) fn decisions(game: &mut PostFlopGame) -> Vec<Decision> {
    assert!(!game.is_compression_enabled());
    fn collect(
        node: &PostFlopNode,
        history: Vec<usize>,
        key: String,
        frontier: Option<u64>,
        river_frontier: Option<u64>,
        out: &mut Vec<Decision>,
    ) {
        if node.is_terminal() {
            return;
        }
        let cards = if node.is_chance() {
            node.chance_cards()
        } else {
            Vec::new()
        };
        if !node.is_chance() {
            out.push(Decision {
                id: node.id(),
                history: history.clone(),
                key: key.clone(),
                board: String::new(),
                signature: String::new(),
                player: node.player(),
                flop_frontier: frontier,
                river_frontier,
            });
        }
        let frontier = if node.is_chance() && node.board_turn() == NOT_DEALT {
            Some(node.id())
        } else {
            frontier
        };
        let river_frontier = if node.is_chance()
            && node.board_turn() != NOT_DEALT
            && node.board_river() == NOT_DEALT
        {
            Some(node.id())
        } else {
            river_frontier
        };
        for a in 0..node.num_actions() {
            let mut h = history.clone();
            h.push(if node.is_chance() {
                cards[a] as usize
            } else {
                a
            });
            let token = if node.is_chance() {
                "C".to_string()
            } else {
                format!("A{a}")
            };
            collect(
                &node.play(a),
                h,
                format!("{key}/{token}"),
                frontier,
                river_frontier,
                out,
            );
        }
    }
    let mut out = Vec::new();
    collect(&game.root(), vec![], "ROOT".into(), None, None, &mut out);
    for d in &mut out {
        game.apply_history(&d.history);
        assert_eq!(
            game.node_id(),
            d.id,
            "isomorphic remapping is not supported"
        );
        d.board = game
            .current_board()
            .iter()
            .skip(3)
            .map(|&c| card_to_string(c).unwrap())
            .collect::<Vec<_>>()
            .join(",");
        d.signature = format!(
            "{:?}|{:?}|{:?}",
            game.available_actions(),
            game.total_bet_amount(),
            game.current_player()
        );
    }
    game.back_to_root();
    out
}

pub(self) fn hand_names(game: &PostFlopGame, p: usize) -> Vec<String> {
    game.private_cards(p)
        .iter()
        .map(|&(a, b)| {
            format!(
                "{}{}",
                card_to_string(a).unwrap(),
                card_to_string(b).unwrap()
            )
        })
        .collect()
}

pub(self) fn canonical(h: &str) -> String {
    let (a, b) = (&h[..2], &h[2..]);
    if a < b {
        format!("{a}{b}")
    } else {
        format!("{b}{a}")
    }
}

pub fn prepare(game: &mut PostFlopGame, assets: &Path) -> K1SharedTransferPlan {
    prepare_family(game, assets, "2x2")
}

pub fn prepare_family(
    game: &mut PostFlopGame,
    assets: &Path,
    family: &str,
) -> K1SharedTransferPlan {
    prepare_target(game, assets, family, "49x48")
}

pub fn prepare_target(
    game: &mut PostFlopGame,
    assets: &Path,
    family: &str,
    target: &str,
) -> K1SharedTransferPlan {
    assert_eq!(
        game.card_config().flop,
        flop_from_str("QsJh2h").unwrap(),
        "Bundled K1 transfer maps currently cover QJ2 only"
    );
    let map = if family == "1x1" && target == "2x2" {
        "1x1_to_2x2"
    } else {
        family
    };
    let path = assets.join(format!("models/maps/QsJh2h_{map}_k1.tsv.gz"));
    let unpack = std::process::Command::new("gzip")
        .arg("-dc")
        .arg(&path)
        .output()
        .expect("gzip is required for K=1 maps");
    assert!(unpack.status.success(), "Cannot read {}", path.display());
    let text = String::from_utf8(unpack.stdout).unwrap();
    let indices: Vec<BTreeMap<_, _>> = (0..2)
        .map(|p| {
            hand_names(game, p)
                .iter()
                .enumerate()
                .map(|(i, h)| (canonical(h), i))
                .collect()
        })
        .collect();
    let mut kept = BTreeSet::from([String::new()]);
    let turns: &[&str] = match family {
        "1x1" => &["3d"],
        "2x2" => &["3d", "4h"],
        "6x6" => &["3d", "4h", "9c", "Ks", "6d", "Tc"],
        _ => panic!("K1 transfer map not supplied for family {family}"),
    };
    let rivers: &[&str] = if family == "1x1" {
        &["7s"]
    } else if family == "6x6" {
        &["7s", "8c", "Th", "5d", "Qc", "2d"]
    } else {
        &["7s", "8c"]
    };
    for &t in turns {
        kept.insert(t.into());
        for &r in rivers {
            kept.insert(format!("{t},{r}"));
        }
    }
    let boards: Vec<_> = kept.iter().cloned().collect();
    let mut mappings = BTreeMap::new();
    for line in text.lines() {
        let row: Vec<_> = line.split('\t').collect();
        assert_eq!(row.len(), 6);
        let p = row[1].parse::<usize>().unwrap();
        assert_eq!(row[5].parse::<f64>().unwrap(), 1.0);
        assert!(kept.contains(row[3]));
        assert!(mappings
            .insert(
                (row[0].to_string(), p, canonical(row[2])),
                (
                    boards.iter().position(|b| b == row[3]).unwrap(),
                    indices[p][&canonical(row[4])]
                )
            )
            .is_none());
    }
    let info = decisions(game);
    let nodes: BTreeMap<_, _> = info
        .iter()
        .map(|d| ((d.key.clone(), d.board.clone(), d.player), d))
        .collect();
    let mut table_ids = BTreeMap::new();
    let mut tables = Vec::new();
    let mut copies = Vec::new();
    let board_cards = |b: &str| {
        let c: Vec<_> = b
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| card_from_str(s).unwrap())
            .collect();
        (
            c.first().copied().unwrap_or(NOT_DEALT),
            c.get(1).copied().unwrap_or(NOT_DEALT),
        )
    };
    let target_boards = if target == "49x48" {
        None
    } else {
        let (ts, rs): (&[&str], &[&str]) = match target {
            "2x2" => (&["3d", "4h"], &["7s", "8c"]),
            "6x6" => (
                &["3d", "4h", "9c", "Ks", "6d", "Tc"],
                &["7s", "8c", "Th", "5d", "Qc", "2d"],
            ),
            _ => panic!("K1 target not supported: {target}"),
        };
        let mut boards = BTreeSet::new();
        for t in ts {
            boards.insert(t.to_string());
            for r in rs {
                if t != r {
                    boards.insert(format!("{t},{r}"));
                }
            }
        }
        Some(boards)
    };
    for d in &info {
        if target_boards
            .as_ref()
            .map_or(false, |b| !b.contains(&d.board))
        {
            continue;
        }
        if kept.contains(&d.board) {
            continue;
        }
        let table = *table_ids
            .entry((d.board.clone(), d.player))
            .or_insert_with(|| {
                let blocked: Vec<_> = d.board.split(',').collect();
                let hands = hand_names(game, d.player)
                    .iter()
                    .map(|h| {
                        if blocked.iter().any(|c| *c == &h[..2] || *c == &h[2..]) {
                            None
                        } else {
                            Some(mappings[&(d.board.clone(), d.player, canonical(h))])
                        }
                    })
                    .collect();
                let index = tables.len();
                tables.push(K1SharedHandTable {
                    player: d.player,
                    target_board: board_cards(&d.board),
                    source_boards: boards.iter().map(|b| board_cards(b)).collect(),
                    source_hands: hands,
                });
                index
            });
        let sources = boards
            .iter()
            .map(|b| {
                nodes.get(&(d.key.clone(), b.clone(), d.player)).map(|s| {
                    assert_eq!(s.signature, d.signature);
                    s.id
                })
            })
            .collect();
        copies.push(K1SharedNodeCopy {
            target_node: d.id,
            table,
            source_nodes: sources,
        });
    }
    let plan = game.prepare_k1_shared_transfer(tables, copies).unwrap();
    println!(
        "K=1 plan: {} destination nodes, {} shared hand tables",
        plan.destination_nodes(),
        plan.table_count()
    );
    plan
}
