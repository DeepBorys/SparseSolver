// SparseSolver — Copyright (C) 2026 Jerzy Slaby (DeepBorys).
// Uses the postflop-solver engine by Wataru Inariba; see NOTICE.
// SPDX-License-Identifier: AGPL-3.0-or-later
// Modified for SparseSolver inference and benchmarks; see NOTICE and LICENSE.

use postflop_solver::*;
use std::path::Path;
#[cfg(test)]
use std::{fs, path::PathBuf};
/// Sparse runs require the exact flop/family bank before allocating a game.
pub fn require_flop_models(assets: &Path, flop: &str, model: &str, turns: &[Card]) {
    if model == "49x48" {
        return;
    }
    let bank = assets.join("models/banks").join(flop).join(model);
    assert!(bank.is_dir(), "No trained NN model bank for flop {flop}, model {model}: {}. This release supplies eight flop-specific banks; run bash setup.sh for QJ2 or bash download_models.sh for all eight. Other flops require their own trained models. Classical 49x48 does not need NN models; no automatic fallback is performed.", bank.display());
    let mut required = vec![
        bank.join("turn/model_interface.json"),
        bank.join("river/model_interface.json"),
        bank.join("river/pot_fraction_only.json"),
        bank.join("turn/model_oop.cpu.bin"),
        bank.join("turn/model_ip.cpu.bin"),
    ];
    for &turn in turns {
        let folder = bank.join("river").join(card_to_string(turn).unwrap());
        required.push(folder.join("model_oop.cpu.bin"));
        required.push(folder.join("model_ip.cpu.bin"));
    }
    for file in required {
        assert!(file.is_file() && file.metadata().map_or(false, |m|m.len()>0),
            "Missing or empty NN asset for flop {flop}, model {model}: {}. Run bash setup.sh for QJ2 or bash download_models.sh for all eight. Every selected turn needs both river models; no different-flop or classical fallback is performed.", file.display());
    }
}

#[cfg(test)]
mod model_availability_tests {
    use super::*;
    fn temporary() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "sparse-model-check-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }
    #[test]
    fn each_sparse_family_rejects_a_missing_flop_bank() {
        let path = temporary();
        for model in ["1x1", "2x2", "4x4", "6x6"] {
            assert!(std::panic::catch_unwind(|| require_flop_models(
                &path,
                "AsKhQd",
                model,
                &[card_from_str("3d").unwrap()]
            ))
            .is_err());
        }
        fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn incomplete_river_bank_is_rejected() {
        let path = temporary();
        let bank = path.join("models/banks/QsJh2h/1x1");
        for file in [
            "turn/model_interface.json",
            "river/model_interface.json",
            "river/pot_fraction_only.json",
            "turn/model_oop.cpu.bin",
            "turn/model_ip.cpu.bin",
            "river/3d/model_oop.cpu.bin",
        ] {
            let p = bank.join(file);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(p, b"fixture").unwrap();
        }
        assert!(std::panic::catch_unwind(|| require_flop_models(
            &path,
            "QsJh2h",
            "1x1",
            &[card_from_str("3d").unwrap()]
        ))
        .is_err());
        fs::write(bank.join("river/3d/model_ip.cpu.bin"), b"fixture").unwrap();
        require_flop_models(&path, "QsJh2h", "1x1", &[card_from_str("3d").unwrap()]);
        fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn classical_requires_no_model_bank() {
        let path = temporary();
        require_flop_models(&path, "AsKhQd", "49x48", &[]);
        fs::remove_dir_all(path).unwrap();
    }
}
