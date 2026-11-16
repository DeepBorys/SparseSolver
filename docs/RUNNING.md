# Command-line reference

[Back to the overview](../README.md) · [Website](http://localhost:8022/SparseSolver/)

All runnable examples and benchmark scripts now default to the available CPU
count. Omit `--threads` on either a four-CPU or sixteen-CPU instance; explicit
overrides remain supported. `OMP_NUM_THREADS=1` limits NN/BLAS internals and
does not reduce independent CFR worker selection. Rebuild with `bash setup.sh`
after updating the Rust examples.


A small CPU-only reproduction branch based on Wataru Inariba’s postflop solver.
It includes solving and neural **inference**, frozen models, and three runnable
benchmark demonstrations. It contains no dataset generators, neural training
programs, training datasets, research experiment runners or private viewer code.

## EC2: clone, compile, run

Use **Ubuntu 24.04 x86-64**, preferably **c8a.4xlarge** (16 workers, 32 GiB RAM).
A 30–40 GiB EBS volume leaves room for Ubuntu, CPU Torch, compilation and caches.
No GPU is needed. Initial installation requires internet access and sudo.

```bash
git clone --depth 1 --single-branch --branch main https://github.com/DeepBorys/SparseSolver.git
cd SparseSolver
bash setup.sh
bash run_quick_benchmark.sh
```

If the repository requires authentication, use your existing GitHub access.
The setup installs Rust 1.88.0 and CPU Torch 2.9.0, verifies/unpacks the models,
and compiles all six examples with native CPU instructions. It prints
`Build ready` when finished. Build and machine details go into `build-info/`.
The scripts do not provision an instance or shut it down.

The checked-in QJ2 model blobs total approximately **128 MiB compressed**, plus
approximately **5 MiB** for K=1 hand correspondence. No solved full-tree reference
is required or downloaded. Models are stored as small content-addressed files;
`models/MANIFEST.json` is a runtime integrity index, not a training manifest.
The QJ2 bank is bundled; the optional single archive supplies all eight flops.

## 1. Quick speed, RAM and flop strategies

```bash
bash run_quick_benchmark.sh
# Optional worker override:
bash run_quick_benchmark.sh --threads 1
```

Runs 1×1/60, 2×2/120, 4×4/150 and 6×6/200 on the published **Q♠J♥2♥ normal
one-size** tree: pot180, stack910; flop52%, turn55%, river70%; raises45%;
normal donk, automatic all-in and bet-merging rules. Starting ranges are inline in
`examples/qj2_3bet_sparse.rs` (also retained in `examples/data/basic_qj2/`
for the batch driver).

For each model: one warmup and three clean repetitions, median CFR solve-loop
milliseconds, maximum process peak RSS, and averaged per-hand flop strategies
at every flop decision. Repeated strategies must match. The result directory
contains `SUMMARY.csv`, `RESULTS.json`, `SYSTEM.json`, per-run console logs and
`flop_strategy.csv` files. Strategy export, setup, finalization and quality
checks are outside the solve timer. Peak RSS includes setup and export.
The exact all-in payout cache is shared across repetitions.

Historical website times are hardware-specific observations, not a promise of
identical milliseconds on every machine. This script does not run expensive
locked-flop best-response comparisons or require saved ground truth.

## Full 40-situation speed sweep

```bash
bash download_models.sh
bash run_all_40.sh
```

This runs eight flops × five situations (`normal`, `wide`, `forced_ip_checks`,
`forced_ip_bets`, `SRP_deep`) × four models: **160 configurations**. Standard
budgets are60/120/150/200 for WASM and120/200/250/300 for SRP_deep. Each has
one warmup plus three timed processes. Outputs include median CFR milliseconds,
peak process RSS, averaged per-hand flop policies, commands and progress.
The JSON/CSV tables update after each configuration. No ground truth, root-RMSE
scoring, locked-BR solve or training program is run. Forced-check/bet cases
retain the original geometry and lock IP opening choices on turn and river.

One archive contains **all eight flops and all four families**: `models-v1.tar.gz`,
approximately1.0GiB compressed. It includes272 frozen inference files plus the
runtime interface markers, not training data. Downloaded models are SHA-256
verified, unpacked and hydrated; allow several GiB for the unpacked assets.
QJ2-only tests still work without this optional download.

Browse or upload model files under the repository’s **Releases → models-v1 →
Assets**, not Deployments:
https://github.com/DeepBorys/SparseSolver/releases

The models-v1 release contains the archive. For a private release, supply `GH_TOKEN`/`GITHUB_TOKEN` in the environment,
or download through your authenticated browser and install the file locally:

```bash
bash download_models.sh --archive /path/to/models-v1.tar.gz
```

`models/ALL_ASSETS.json` pins the filename, size and SHA-256. The release archive
stays outside Git history. For a future repository, update the repository/URL
fields while retaining the same asset checksum. An alternate public download
host can be supplied with `--url` or `SPARSE_MODEL_ARCHIVE_URL`.

Select smaller runs, inspect cases, or resume a disconnected run:

```bash
bash run_all_40.sh --list
bash run_all_40.sh --flops QsJh2h --models 2x2
bash run_all_40.sh --output results/my_sweep
bash run_all_40.sh --resume --output results/my_sweep
```

Resume checks the binary, configuration hashes, worker count, budgets and
selection before reusing finished runs. Keep long jobs alive with tmux when
using SSH. Setup installs tmux. Timing excludes setup, model loading,
finalization, strategy export and checks; process RSS includes those phases.
The static fixtures below cover QJ2 normal only; other sweep policies receive
coverage, finiteness, normalization and repeated-policy checks.

## Static numerical checks

The quick benchmark automatically compares every measured flop policy against
small frozen fixtures for QJ2 single-size: all **11,024 action/hand entries across
12 flop nodes** per model. It checks the model/budget, action and hand coverage,
finite normalized probabilities, and maximum absolute probability difference
≤0.0001. Each passing run prints `STATIC_CHECK_PASS`; a mismatch exits nonzero.
The compressed fixtures for all four models total approximately0.43MiB.

Saved outputs can also be checked without running a solver:

```bash
python3 scripts/check_results.py results/YOUR_QUICK_RUN
python3 scripts/check_results.py results/YOUR_STAGED_RUN
python3 scripts/check_results.py results/YOUR_ASSEMBLY_RUN
```

For the default2×2/120 seed, staged checkpoints0/10/20 are checked against
105.67829% /4.0912504% /1.9991374% full-tree exploitability. The default all-frontier
local100 assembly must retain343 frontiers, preserve the flop policy, and match
**1.300190806%** full-tree exploitability within0.005 percentage points of pot.
Other seed/refinement budgets do not have these static fixtures.
Iteration0 is the uniform averaged policy immediately after clearing averages.

These are regression checks against recorded outputs, not exact Nash ground
truth. They do not recompute locked-flop BR loss or assert wall-clock speed.
Numerical tolerances accommodate small CPU/compiler floating-point differences;
they are not accuracy bounds on the poker strategy.

## 2. Assemble a complete policy and measure exploitability

```bash
bash run_complete_policy.sh
# Change the refinement budget:
bash run_complete_policy.sh --refine-iterations 300
```

Allocates the full tree locally, solves a 2×2/120 seed, applies signed K=1 regret
transfer, restores and freezes the seed’s averaged flop policy, independently
refines **all 343 eligible turn frontiers with all physical river runouts**, and
computes complete-policy best responses for both seats. Default refinement:
100 iterations per frontier. Flop regret and average fingerprints are checked
after each subtree. Exact all-in payout leaves are handled separately.

The output includes per-frontier timings, each seat’s BR gain, mean full-tree
exploitability as a percentage of the starting pot, and seed/final flop logs.
The seed and final flop policies must agree. Full-tree allocation consumes more
RAM than the quick sparse solves. Single-frontier latency and the total cost
of refining every frontier are different measurements.

This is **unsafe range-conditioned refinement**, not a theoretical safety
guarantee. This single 2×2/120 seed is a demonstration configuration; it differs
from the website’s staged-seed rows. Its quality is measured afresh.

## 3. Warm start and continue the full classical solve

```bash
bash run_staged_example.sh --full-iterations 100
```

2×2/120 → signed K=1 transfer → fresh average accumulators → ordinary full-tree
CFR. Prints and saves full-tree exploitability every **10 full iterations**,
plus iteration0 and the final checkpoint. The CFR clock continues from120.
Flop strategy may change during the full solve. BR evaluation times are saved
separately from CFR times. Interleaved checks make this a convergence
illustration, not an uninterrupted classical timing replay.

At iteration0, averages have just been cleared and evaluate as a uniform
policy; that initial value is not the regret-matched seed quality.

## Same public game API, one model choice

`examples/qj2_3bet_classical.rs` and `examples/qj2_3bet_sparse.rs` differ only here:

```rust
nn_model::Model::prepare("49x48", &card_config); // classical
nn_model::Model::prepare("2x2", &card_config);   // sparse
```

Either accepts `--model 49x48|6x6|4x4|2x2|1x1`:

```bash
bash run_example.sh qj2_3bet_sparse --model 4x4
bash run_example.sh qj2_3bet_classical --model 49x48 --iterations 100
bash run_example.sh basic
```

Both companion examples share game construction, `solve_step`, navigation,
strategy export and EV/equity fields. Sparse EVs include neural continuations;
they are not full-game exploitability. Full exploitability is evaluated only
for a classical/assembled full tree. `examples/basic.rs` preserves Wataru’s
original **turn** example and assertions; the companion benchmarks start on
the QJ2 **flop**, so their numerical outputs naturally differ.

`examples/README.md` documents details and direct Cargo commands.
The shared inference and solver kernels retain compatible runtime tuning
switches, but no research or training command-line programs are shipped.

## Two-size strategy demonstrations

```bash
bash run_example.sh qj2_3bet_twosize_classical
bash run_example.sh qj2_3bet_twosize_fast
bash run_example.sh qj2_3bet_twosize_deep
```

These show the three methods in the landing table with inline game inputs.
Classical prints strategies after89 and160 iterations. The sparse examples use
1×1/50→2×2/25 +100 local iterations, or2×2/100→6×6/100 +300 local iterations.
Each prints the flop root and selected check/bet25%/call/3d continuation as13×13
matrices, with per-combo CSVs. No locks or full-tree BR evaluations are run.
The original two-frontier `qj2_3bet_refine` prints both requested continuations.
See [the example guide](../examples/README.md) for schedules and timing scope.

## Scope and license

This is an AWS testing preview, not a new public repository or a website
publication. Browser solution links and the 25-second presentation remain on
the author’s personal website; the video workflow implementation is excluded.

This branch tip is trimmed. The existing repository’s older history is not
rewritten. A future clean public repository should import this snapshot without
bringing the development/training history along.

Original solver © Wataru Inariba; SparseSolver additions © Jerzy Slaby.
AGPL-3.0-or-later; retain `LICENSE` and `NOTICE`.
Original project: https://github.com/b-inary/postflop-solver

## Missing models

Sparse families1x1/2x2/4x4/6x6 use flop-specific trained weights. The supplied bank
covers the eight benchmark flops only. The example checks the exact flop/family
folder, runtime interface metadata and both seats’ turn and sampled-river model
files before allocating the tree or creating output. A missing bank/file raises
an assertion naming the flop, family and expected path. Download the supplied
bank when applicable; an unsupported flop needs additional trained weights.
There is no automatic substitution of a different flop or classical solving.
Explicit49x48 mode skips model checks and solves classically.
