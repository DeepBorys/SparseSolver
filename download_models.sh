#!/usr/bin/env bash
# SparseSolver — Copyright (C) 2026 Jerzy Slaby (DeepBorys).
# Uses the postflop-solver engine by Wataru Inariba; see NOTICE.
# SPDX-License-Identifier: AGPL-3.0-or-later
# Modified for SparseSolver inference and benchmarks; see NOTICE and LICENSE.

set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python3 "$root/scripts/download_models.py" "$@"
python3 "$root/scripts/prepare_models.py" --all
