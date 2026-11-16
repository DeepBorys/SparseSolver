// SparseSolver — Copyright (C) 2026 Jerzy Slaby (DeepBorys).
// Uses the postflop-solver engine by Wataru Inariba; see NOTICE.
// SPDX-License-Identifier: AGPL-3.0-or-later
// Modified for SparseSolver inference and benchmarks; see NOTICE and LICENSE.

//! Selected-card order must match the trained inference weights.
use postflop_solver::*;
pub fn select(flop: [Card; 3], model: &str) -> (String, Vec<Card>, Vec<Card>) {
    let name = [
        "QsJh2h", "As9h4d", "Ks4h4d", "Js9s4h", "QsTs7h", "Js7h3d", "Td9d6h", "Js8s3s",
    ]
    .into_iter()
    .find(|name| flop_from_str(name).unwrap() == flop)
    .map(str::to_owned)
    .unwrap_or_else(|| {
        flop.iter()
            .rev()
            .map(|&c| card_to_string(c).unwrap())
            .collect::<String>()
    });
    if model == "49x48" {
        return (name, Vec::new(), Vec::new());
    }
    let (turns,rivers)=match (name.as_str(),model) {
        ("QsJh2h","1x1") => ("3d","7s"),
        ("QsJh2h","2x2") => ("3d,4h","7s,8c"),
        ("QsJh2h","4x4") => ("3d,4h,9c,Ks","7s,8c,Th,5d"),
        ("QsJh2h","6x6") => ("3d,4h,9c,Ks,6d,Tc","7s,8c,Th,5d,Qc,2d"),
        ("As9h4d","1x1") => ("3s","Qc"),
        ("As9h4d","2x2") => ("3s,6d","2s,Qc"),
        ("As9h4d","4x4") => ("3s,6d,Tc,Kh","2s,4c,8d,Qc"),
        ("As9h4d","6x6") => ("2c,3s,6d,8h,Tc,Kh","2s,4c,6s,8d,Th,Qc"),
        ("Ks4h4d","1x1") => ("3d","Qc"),
        ("Ks4h4d","2x2") => ("3d,6c","2s,Qc"),
        ("Ks4h4d","4x4") => ("3d,6c,9h,Qs","2s,4c,8d,Qc"),
        ("Ks4h4d","6x6") => ("2c,3d,6c,9h,Js,Qs","2s,4c,6s,8d,Th,Qc"),
        ("Js9s4h","1x1") => ("3d","Qc"),
        ("Js9s4h","2x2") => ("3d,6c","2s,Qc"),
        ("Js9s4h","4x4") => ("3d,6c,8h,Ks","2s,4c,8d,Qc"),
        ("Js9s4h","6x6") => ("2h,3d,6c,8h,Tc,Ks","2s,4c,6s,8d,Th,Qc"),
        ("QsTs7h","1x1") => ("3d","Qc"),
        ("QsTs7h","2x2") => ("3d,5s","2s,Qc"),
        ("QsTs7h","4x4") => ("3d,5s,9c,Kh","2s,4c,8d,Qc"),
        ("QsTs7h","6x6") => ("2c,3d,5s,8h,9c,Kh","2s,4c,6s,8d,Th,Qc"),
        ("Js7h3d","1x1") => ("5s","Qc"),
        ("Js7h3d","2x2") => ("2c,5s","2s,Qc"),
        ("Js7h3d","4x4") => ("2c,5s,9h,Kd","2s,4c,8d,Qc"),
        ("Js7h3d","6x6") => ("2c,5s,6d,9h,Tc,Kd","2s,4c,6s,8d,Th,Qc"),
        ("Td9d6h","1x1") => ("3c","Qc"),
        ("Td9d6h","2x2") => ("3c,Kd","2s,Qc"),
        ("Td9d6h","4x4") => ("3c,7h,9s,Kd","2s,4c,8d,Qc"),
        ("Td9d6h","6x6") => ("3c,5d,7h,9s,Jc,Kd","2s,4c,6s,8d,Th,Qc"),
        ("Js8s3s","1x1") => ("5s","Qc"),
        ("Js8s3s","2x2") => ("2c,5s","2s,Qc"),
        ("Js8s3s","4x4") => ("2c,5s,9h,Kd","2s,4c,8d,Qc"),
        ("Js8s3s","6x6") => ("2c,5s,6d,9h,Tc,Kd","2s,4c,6s,8d,Th,Qc"),
        _ => panic!("No trained NN model bank for flop {name}, model {model}. The release provides eight flop-specific banks; use explicit 49x48 for classical solving. No automatic fallback is performed."),
    };
    let parse = |s: &str| s.split(',').map(|c| card_from_str(c).unwrap()).collect();
    (name, parse(turns), parse(rivers))
}
