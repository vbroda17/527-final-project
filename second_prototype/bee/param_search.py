#!/usr/bin/env python3
"""
Grid-search driver for bee/bco.py
---------------------------------
Runs every combination of
    n_bees × scout_frac × employed_frac × extra_food
`N_RUNS` times each and stores results under tests/.

Folder layout
tests/
 ├─ master_results.csv
 └─ (n_bees=30, scout_frac=0.15, employed_frac=0.50, extra_food=5)/
       ├─ combo_summary.csv
       ├─ trial_01/   ← raw optimiser output
       ├─ trial_02/
       └─ …
"""

import itertools, subprocess, csv
from pathlib import Path

# ───────── fixed settings (edit if needed) ─────────
START   = "Earth"
DEST    = "Mars"
YEARS   = 2.0
DT      = 1.0
SPEED   = 100_000       # km/h
EPOCHS  = 1
N_RUNS  = 20           # trials per parameter set
TESTS   = Path("tests")
TESTS.mkdir(exist_ok=True)
MASTER  = TESTS / "master_results.csv"

# ───────── parameter grid ─────────
N_BEES_LIST        = [20, 30, 50, 75]
SCOUT_FRAC_LIST    = [0.10, 0.15, 0.25, 0.50]
EMPLOYED_FRAC_LIST = [0.10, 0.25, 0.50, 0.75]
EXTRA_FOOD_LIST    = [0, 3, 5, 8]

# write master header once
if not MASTER.exists():
    with MASTER.open("w", newline="") as f:
        csv.writer(f).writerow(
            ["n_bees", "scout_frac", "employed_frac", "extra_food",
             "trial", "best_fit", "best_step", "path_len", "solved"]
        )

def run_trial(combo_dir: Path,
              n_bees: int, scout_frac: float,
              employed_frac: float, extra_food: int,
              trial: int):
    """Run optimiser once and return metrics."""
    trial_dir = combo_dir / f"trial_{trial:02d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "bee.bco",
        "--start", START, "--dest", DEST,
        "--years", str(YEARS), "--dt", str(DT), "--speed", str(SPEED),
        "--n_bees", str(n_bees),
        "--scout_frac", str(scout_frac),
        "--employed_frac", str(employed_frac),
        "--extra_food", str(extra_food),
        "--epochs", str(EPOCHS),
        "--out_root", str(trial_dir),
        "--skip_gif",
        "--use_gravity"
    ]
    subprocess.run(cmd, check=True)

    run_id  = f"{START}_{DEST}_ABC_{n_bees}b"
    summ    = trial_dir / run_id / "summary.csv"
    with summ.open() as f:
        next(f)                                      # skip header
        (_,_,_,_, best_fit, best_step,
         path_len, solved) = next(f).strip().split(",")

    return best_fit, best_step, path_len, solved

def main():
    grid = itertools.product(
        N_BEES_LIST, SCOUT_FRAC_LIST, EMPLOYED_FRAC_LIST, EXTRA_FOOD_LIST
    )

    for n_bees, scout_frac, employed_frac, extra_food in grid:
        combo_dir = TESTS / (
            f"(n_bees={n_bees}, scout_frac={scout_frac}, "
            f"employed_frac={employed_frac}, extra_food={extra_food})"
        )
        combo_dir.mkdir(exist_ok=True)

        combo_rows = []

        for trial in range(1, N_RUNS + 1):
            print(f"▶ {combo_dir.name}, trial {trial:02d}")
            bf, bs, pl, ok = run_trial(
                combo_dir, n_bees, scout_frac, employed_frac, extra_food, trial
            )
            row = [n_bees, scout_frac, employed_frac, extra_food,
                   trial, bf, bs, pl, ok]
            combo_rows.append(row)

            # append to master
            with MASTER.open("a", newline="") as f:
                csv.writer(f).writerow(row)

        # per-combo summary
        with (combo_dir / "combo_summary.csv").open("w", newline="") as f:
            csv.writer(f).writerow(
                ["n_bees", "scout_frac", "employed_frac", "extra_food",
                 "trial", "best_fit", "best_step", "path_len", "solved"]
            )
            csv.writer(f).writerows(combo_rows)

if __name__ == "__main__":
    main()
