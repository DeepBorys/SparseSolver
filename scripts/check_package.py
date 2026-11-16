#!/usr/bin/env python3
# SparseSolver — Copyright (C) 2026 Jerzy Slaby (DeepBorys).
# Uses the postflop-solver engine by Wataru Inariba; see NOTICE.
# SPDX-License-Identifier: AGPL-3.0-or-later
# Modified for SparseSolver inference and benchmarks; see NOTICE and LICENSE.

"""Audit the tracked preview payload; model hydration/build output is ignored."""
import json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
files=subprocess.check_output(['git','ls-files'],cwd=root,text=True).splitlines()
allowed={'src','examples','scripts','python_scripts','models','docs','configs','tests','.cargo','.github'}
for name in files:
 p=Path(name)
 if len(p.parts)>1 and p.parts[0] not in allowed:raise ValueError('Unexpected directory: '+name)
 if any(word in p.name.lower() for word in ['datagen','training','train_','experiment']):raise ValueError('Research/training filename: '+name)
 if name.startswith('examples/') and p.suffix=='.rs' and '/support/'not in name and p.stem not in {'basic','qj2_3bet_classical','qj2_3bet_sparse','qj2_3bet_warm_start','complete_policy','speed_case','qj2_3bet_refine','qj2_3bet_twosize_classical','qj2_3bet_twosize_fast','qj2_3bet_twosize_deep'}:raise ValueError('Unexpected example: '+name)
 if name.startswith('python_scripts/') and p.name!='benchmark_basic_models.py':raise ValueError('Unexpected Python entry point: '+name)
 if name.startswith('models/') and (p.name.endswith('.cpu.bin') or '/cache/'in name):raise ValueError('Hydrated cache must not be tracked: '+name)
 if (root/name).stat().st_size>10*1024**2:raise ValueError('Oversized individual file: '+name)
for family in ['1x1','2x2','4x4','6x6']:
 for name in ['turn/model_interface.json','river/model_interface.json','river/pot_fraction_only.json']:
  assert (root/'models/banks/QsJh2h'/family/name).is_file(),(family,name)
assert len(json.loads((root/'models/MANIFEST.json').read_text())['files'])==34
assert len(json.loads((root/'configs/speed_cases.json').read_text())['cases'])==160
print(f'PACKAGE_OK tracked_files={len(files)} training_programs=0 datagen_programs=0 supported_examples=10')
