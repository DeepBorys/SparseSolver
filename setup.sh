#!/usr/bin/env bash
# SparseSolver — Copyright (C) 2026 Jerzy Slaby (DeepBorys).
# Uses the postflop-solver engine by Wataru Inariba; see NOTICE.
# SPDX-License-Identifier: AGPL-3.0-or-later
# Modified for SparseSolver inference and benchmarks; see NOTICE and LICENSE.

set -euo pipefail
setup_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$setup_root"
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || { echo 'This preview targets Ubuntu 24.04 x86-64, including c8a.4xlarge.' >&2; exit 1; }
if [[ ${SPARSE_SKIP_SYSTEM_INSTALL:-0} != 1 ]]; then
 setup_sudo=(); if (( EUID != 0 )); then setup_sudo=(sudo); fi
 "${setup_sudo[@]}" apt-get update
 "${setup_sudo[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends build-essential ca-certificates curl cmake pkg-config libssl-dev python3-venv gzip tmux
fi
export PATH="$HOME/.cargo/bin:$PATH"
setup_toolchain=1.88.0
if ! command -v rustup >/dev/null; then
 setup_installer=$(mktemp)
 curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs -o "$setup_installer"
 sh "$setup_installer" -y --profile minimal --default-toolchain "$setup_toolchain"
 rm -- "$setup_installer"
fi
rustup toolchain install "$setup_toolchain" --profile minimal
if [[ ! -x .venv/bin/python ]]; then python3 -m venv .venv; fi
if ! .venv/bin/python -c 'import torch; assert torch.__version__ == "2.9.0+cpu"' 2>/dev/null; then
 .venv/bin/python -m pip install --no-cache-dir 'torch==2.9.0' --index-url https://download.pytorch.org/whl/cpu
fi
.venv/bin/python scripts/prepare_models.py
export PATH="$setup_root/.venv/bin:$PATH"
export VIRTUAL_ENV="$setup_root/.venv"
export CARGO_TARGET_DIR="$setup_root/target"
export LIBTORCH_USE_PYTORCH=1
setup_torch_lib=$(.venv/bin/python -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("torch").origin).parent/"lib")')
export LD_LIBRARY_PATH="$setup_torch_lib:${LD_LIBRARY_PATH:-}"
export RUSTFLAGS='-C target-cpu=native -A dangerous_implicit_autorefs'
export CXXFLAGS='-O3 -march=native -mtune=native'
export CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1
cargo "+$setup_toolchain" build --locked --release --features neural --examples -j "${SPARSE_BUILD_JOBS:-2}"
mkdir -p build-info
rustup run "$setup_toolchain" rustc -Vv > build-info/rustc.txt
.venv/bin/python - <<'PY'
import json,os,platform,subprocess
from pathlib import Path
Path('build-info/system.json').write_text(json.dumps({'platform':platform.platform(),'torch':'2.9.0+cpu','cpu':subprocess.check_output(['lscpu','-J'],text=True),'settings':{k:v for k,v in os.environ.items() if k.startswith(('RUSTFLAGS','CXXFLAGS','CARGO_PROFILE_RELEASE','LIBTORCH'))}},indent=2)+'\n')
PY
printf '\nBuild ready. Run: bash run_quick_benchmark.sh\n'
