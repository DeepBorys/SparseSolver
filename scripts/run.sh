#!/usr/bin/env bash
# SparseSolver — Copyright (C) 2026 Jerzy Slaby (DeepBorys).
# Uses the postflop-solver engine by Wataru Inariba; see NOTICE.
# SPDX-License-Identifier: AGPL-3.0-or-later
# Modified for SparseSolver inference and benchmarks; see NOTICE and LICENSE.

set -euo pipefail
run_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$run_root"
run_mode=${1:-quick}; shift || true
if [[ ! -x target/release/examples/qj2_3bet_sparse ]]; then
 echo 'Run bash setup.sh first.' >&2; exit 1
fi
if [[ -x .venv/bin/python ]]; then
 run_python="$run_root/.venv/bin/python"
 run_torch_lib=$("$run_python" -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("torch").origin).parent/"lib")')
elif [[ -n ${LIBTORCH:-} && -f ${LIBTORCH}/lib/libtorch_cpu.so ]]; then
 run_python=python3
 run_torch_lib="$LIBTORCH/lib"
else
 echo 'Run bash setup.sh, or provide a compatible CPU LIBTORCH directory.' >&2; exit 1
fi
export LD_LIBRARY_PATH="$run_torch_lib:${LD_LIBRARY_PATH:-}"
run_threads=$("$run_python" -c 'from scripts.cpu_threads import available_threads; print(available_threads())')
export RAYON_NUM_THREADS="$run_threads"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Keep previously documented shell commands working, without ambiguous source filenames.
case "$run_mode" in
 basic_classical) run_mode=qj2_3bet_classical ;;
 basic_sparse) run_mode=qj2_3bet_sparse ;;
 basic_staged) run_mode=qj2_3bet_warm_start ;;
 basic_refine_all) run_mode=complete_policy ;;
esac
# TSV-driven cases belong to the batch driver, not the copy-and-edit examples.
for arg in "$@"; do
 if [[ $arg == --case-config && ( $run_mode == qj2_3bet_sparse || $run_mode == qj2_3bet_classical ) ]]; then run_mode=speed_case; break; fi
done
case "$run_mode" in
 sweep) exec "$run_python" -u scripts/speed_sweep.py --threads "$run_threads" "$@" ;;
 quick) exec "$run_python" -u python_scripts/benchmark_basic_models.py --assets . --threads "$run_threads" "$@" ;;
 basic) exec target/release/examples/basic "$@" ;;
 qj2_3bet_twosize_classical|qj2_3bet_twosize_fast|qj2_3bet_twosize_deep|qj2_3bet_classical|qj2_3bet_sparse|qj2_3bet_warm_start|qj2_3bet_refine|complete_policy|speed_case)
  run_output="results/${run_mode}_$(date -u +%Y%m%d_%H%M%S)_$$"
  mkdir -p "$run_output"
  exec > >(tee "$run_output/console.log") 2>&1
  exec "target/release/examples/$run_mode" --threads "$run_threads" --assets . --output "$run_output" --allin-cache "$run_root/results/allin.bin" "$@" ;;
 *) echo "Unknown mode: $run_mode" >&2;exit 2 ;;
esac
