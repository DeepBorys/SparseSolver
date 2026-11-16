// SparseSolver — Copyright (C) 2026 Jerzy Slaby (DeepBorys).
// Uses the postflop-solver engine by Wataru Inariba; see NOTICE.
// SPDX-License-Identifier: AGPL-3.0-or-later
// Modified for SparseSolver inference and benchmarks; see NOTICE and LICENSE.

#[path = "support/basic_k1.rs"]
mod basic_k1;
#[path = "support/benchmark_runtime.rs"]
mod benchmark_runtime;

#[cfg(feature = "neural")]
#[path = "support/basic_refine.rs"]
mod basic_refine;

fn main() {
    benchmark_runtime::run_refine_all();
}
