# Editable examples

`basic.rs` is Wataru’s original T96/Q **turn** example, unchanged.
The four `qj2_3bet_*.rs` files start on the Q♠J♥2♥ **flop**. Each shows its
ranges, cards, betting sizes and game construction directly in `main`.
Copy a file and edit those values as with the original example.

| File | Workflow |
| --- | --- |
| `qj2_3bet_classical.rs` | Classical 49×48 solve |
| `qj2_3bet_sparse.rs` | Sparse solve; defaults to 2×2 |
| `qj2_3bet_warm_start.rs` | Sparse seed → K=1 transfer → full CFR |
| `qj2_3bet_refine.rs` | Staged seed → refine a list of selected subgames |

The classical and sparse examples differ in one model setting:

```rust
let model = nn_model::Model::prepare("2x2", &card_config);
```

Valid model choices are `49x48`, `6x6`, `4x4`, `2x2`, `1x1`.
Sparse models are flop-specific; eight flop banks are supplied. Missing models
fail before allocation. Classical 49×48 needs no NN model.

The warm-start and refinement examples have two workflow settings:

```rust
let stages = &[("1x1", 50), ("2x2", 25), ("49x48", 100)];
let frontiers = &[
    (vec![Action::Check, Action::Bet(94), Action::Call,
          Action::Chance(card_from_str("7c").unwrap())], 100),
];
```

An empty frontier list skips local refinement. Omit the 49×48 stage to refine
after the sparse seed without full-game CFR first. Histories use the existing
`Action` enum: bet amounts are chips, and chance cards use `card_from_str`.
Each history must end at a turn or river decision; multiple requested subgames
must be disjoint. Invalid histories fail before CFR starts.

K=1 maps bundled here cover the canonical QJ2 ranges for 1×1, 2×2 and 6×6 sources.
The staged examples currently require those maps; changing flop/ranges or using
4×4 as a transfer source requires corresponding maps. Sparse-only model
selection supports all four families on the eight supplied flops.

```bash
bash run_example.sh qj2_3bet_sparse --model 2x2
bash run_example.sh qj2_3bet_classical --iterations 100
bash run_example.sh qj2_3bet_warm_start --full-iterations 100
bash run_example.sh qj2_3bet_refine --refine-iterations 100
```

Run from the repository root after `bash setup.sh`. `--threads` defaults to
available CPUs. `--iterations` overrides the first stage budget;
`--full-iterations` overrides the final classical stage, and
`--refine-iterations` overrides every requested subgame budget.

Flop strategies are logged to CSV. The warm-start example reports exact full-tree
exploitability every ten full iterations. The two-frontier refinement example
prints a 13×13 matrix for each action at **both** requested frontiers, plus the
flop root. It saves full per-combo probabilities to separate CSVs. It skips
full-tree BR calculations and locked-loss comparisons.

Local refinement preserves the averaged flop policy. Other subgames remain
unrefined. This is unsafe range-conditioned solving, without a safety guarantee.
Use the complete-policy benchmark to assess refinement everywhere. Setup,
transfer, printing, finalization and exports are excluded from CFR timing.

## Two-size fast-strategy demonstrations

All three examples show the same QJ2 ranges and two-size betting configuration
inline: flop25/75%, turn33/100%, river50/150%, raises45%.

| Command after `bash setup.sh` | What it solves and prints |
| --- | --- |
| `bash run_example.sh qj2_3bet_twosize_classical` | Full classical CFR; flop and check/bet25%/call/3d strategies at 89 and 160 iterations |
| `bash run_example.sh qj2_3bet_twosize_fast` | 1×1/50 → 2×2/25; 100 unsafe iterations at check/bet25%/call/3d; flop and turn strategies |
| `bash run_example.sh qj2_3bet_twosize_deep` | 2×2/100 → 6×6/100; 300 unsafe iterations at the same frontier; flop and turn strategies |

Edit the `stages` and `frontiers` lists in either sparse example to request
other budgets or multiple disjoint continuations. `qj2_3bet_refine.rs` demonstrates
two requested frontiers on the original one-size tree.

Each prints CFR milliseconds and strategy-log paths. Sparse examples also print
transfer milliseconds and seed-plus-selected-frontier CFR milliseconds. The
website's ≈109ms and1.162s rows use seed plus **one mean continuation** on the
original benchmark hardware, excluding transfers. A chosen continuation has its
own cost. These examples reproduce the seed schedules and provide the requested
strategies; they do not assemble all539 subgames or recompute the table's
full-tree/locked losses. No saved exact reference is needed.

`complete_policy.rs` and `run_complete_policy.sh` retain the separate benchmark
that refines **all** physical turn frontiers. `speed_case.rs` reads TSV cases for
the 40-situation batch sweep; those inputs are separate from the editable examples.
