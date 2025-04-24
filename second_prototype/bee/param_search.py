#!/usr/bin/env python3
# bco_experiments.py

import argparse
import subprocess
import sys
import os
import csv
import shutil
from itertools import product

from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser(
        description="Run parameter sweeps of bee/bco.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--bees", type=int, nargs="+", required=True,
                   help="List of n_bees values to try")
    p.add_argument("--scout_fracs", type=float, nargs="+", required=True,
                   help="List of scout_frac values to try")
    p.add_argument("--extra_foods", type=int, nargs="+", required=True,
                   help="List of extra_food values to try")
    p.add_argument("--runs", type=int, default=20,
                   help="How many repeats per parameter combo")
    p.add_argument("--years", type=float, default=5.0,
                   help="Pass‐through to bco.py --years")
    p.add_argument("--dt", type=float, default=1.0,
                   help="Pass‐through to bco.py --dt")
    p.add_argument("--start", default="Earth",
                   help="Pass‐through to bco.py --start")
    p.add_argument("--dest", default="Mars",
                   help="Pass‐through to bco.py --dest")
    return p.parse_args()

def main():
    args = parse_args()

    EXP_DIR = "bco_experiments"
    os.makedirs(EXP_DIR, exist_ok=True)

    # prepare aggregate CSVs
    results_path  = os.path.join(EXP_DIR, "results.csv")
    failures_path = os.path.join(EXP_DIR, "failures.csv")

    # headers for our summary
    headers = [
        "bees","scout_frac","extra_food","run_idx",
        "best_fit","path_length_AU","arrival_step"
    ]
    with open(results_path,  "w", newline="") as f:
        csv.writer(f).writerow(headers)
    with open(failures_path, "w", newline="") as f:
        csv.writer(f).writerow(headers)

    combos = list(product(args.bees, args.scout_fracs, args.extra_foods))
    total_runs = len(combos) * args.runs
    pbar = tqdm(total=total_runs, desc="Experiment runs")

    # for each parameter combination
    for n_bees, scout_frac, extra_food in combos:
        combo_dir = f"bees{n_bees}_scout{scout_frac}_food{extra_food}"
        combo_path = os.path.join(EXP_DIR, combo_dir)
        os.makedirs(combo_path, exist_ok=True)

        # repeat runs
        for run_idx in range(1, args.runs+1):
            # create a fresh temp working dir
            run_dir = os.path.join(combo_path, f"run_{run_idx:02d}")
            if os.path.exists(run_dir):
                shutil.rmtree(run_dir)
            os.makedirs(run_dir)

            # build the command
            cmd = [
                sys.executable, "-m", "bee.bco",
                "--start", args.start,
                "--dest",  args.dest,
                "--years", str(args.years),
                "--dt",    str(args.dt),
                "--n_bees",      str(n_bees),
                "--scout_frac",  str(scout_frac),
                "--extra_food",  str(extra_food),
                "--epochs",      "1",
            ]
            # run it
            try:
                subprocess.run(cmd, cwd=run_dir, check=True,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                # treat as failure
                best_fit = -1.0
                path_len = 0.0
                arrival  = -1
            else:
                # read summary.csv out of run_dir/bco_output/run_id/summary.csv
                run_id = f"{args.start}_{args.dest}_ABC_{n_bees}b"
                sum_path = os.path.join(
                    run_dir, "bco_output", run_id, "summary.csv"
                )
                if not os.path.isfile(sum_path):
                    # missing summary → treat as failure
                    best_fit = -1.0
                    path_len = 0.0
                    arrival  = -1
                else:
                    with open(sum_path) as sf:
                        reader = csv.DictReader(sf)
                        row    = next(reader)
                        best_fit = float(row["best_fit"])
                        path_len = float(row["path_length_AU"])
                        arrival  = int(row["arrival_step"])

            out_row = [
                n_bees, scout_frac, extra_food, run_idx,
                best_fit, path_len, arrival
            ]

            # append to results or failures
            if best_fit < 0:
                with open(failures_path, "a", newline="") as f:
                    csv.writer(f).writerow(out_row)
            else:
                with open(results_path, "a", newline="") as f:
                    csv.writer(f).writerow(out_row)

            pbar.update()

    pbar.close()
    print("All done.")
    print("  successes →", results_path)
    print("  failures →", failures_path)

if __name__ == "__main__":
    main()
