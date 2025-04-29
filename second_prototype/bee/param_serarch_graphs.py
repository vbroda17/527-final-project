#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from pathlib import Path
import matplotlib as mpl

mpl.rcParams.update({
    # Base font for text & annotations
    'font.size':          20,       # default text
    'font.family':        'serif',  # or 'sans-serif'
    # Axes
    'axes.titlesize':     24,       # main plot title
    'axes.labelsize':     20,       # x/y axis labels
    # Tick labels
    'xtick.labelsize':    16,
    'ytick.labelsize':    16,
    # Legend
    'legend.fontsize':    16,
    # Figure title (if you use suptitle)
    'figure.titlesize':   28,
    # When you call plt.figure() without figsize, these defaults kick in:
    'figure.figsize':     (12, 9),  # inches
    'figure.dpi':         300,      # high‐res for printing
})

# ───── CONFIG ─────
MASTER_CSV = Path("tests") / "master_results.csv"
BASE_DIR   = Path("results") / "heatmaps" / "all"
REPORT_TXT = Path("results") / "summary_report.txt"

PARAMS  = ["n_bees", "scout_frac", "employed_frac", "extra_food"]
METRICS = ["best_fit", "best_step", "path_len"]
AGG     = "mean"

BASE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)


def plot_and_save(df, x, y, metric, out_dir,
                  cell_fontsize=12,  # ↑ make inner text bigger
                  title_pad=15):     # ↑ extra space above title
    table = df.groupby([y, x])[metric].mean().unstack(x)
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(table.values, aspect="auto", origin="lower")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(f"mean of {metric}", fontsize=cell_fontsize)

    # ticks
    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=45, ha="right", fontsize=cell_fontsize)
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_yticklabels(table.index, fontsize=cell_fontsize)

    # annotations
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i,
                    f"{table.iat[i, j]:.2f}",
                    ha="center", va="center",
                    fontsize=cell_fontsize,
                    color="black" if im.norm(table.iat[i, j]) > 0.5 else "white")

    # labels
    ax.set_xlabel(x, fontsize=cell_fontsize + 4)
    ax.set_ylabel(y, fontsize=cell_fontsize + 4)

    # title with padding
    ax.set_title(f"Mean {metric}\nby {y} vs {x}",
                 pad=title_pad,  # ↑ move it up
                 fontsize=cell_fontsize + 6)

    # make room for the title
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # save
    out_path = out_dir / f"{y}_vs_{x}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved heatmap: {out_path}")



def compute_extremes(df):
    """
    Compute average‐based extremes over each parameter combo.
    Returns dict: { title → pandas.Series(row) }
    """
    grouped = (
        df.groupby(PARAMS)
          .agg(
             avg_fit   = ("best_fit",  "mean"),
             avg_steps = ("best_step", "mean"),
             avg_path  = ("path_len",  "mean"),
          )
          .reset_index()
    )

    extremes = {}
    specs = [
      ("Avg best fit",   "avg_fit",  True),
      ("Avg worst fit",  "avg_fit",  False),
      ("Avg fastest",    "avg_steps", False),  # fewer steps is “fastest”
      ("Avg slowest",    "avg_steps", True),
      ("Avg shortest",   "avg_path",  False),
      ("Avg longest",    "avg_path",  True),
    ]
    for title, col, take_max in specs:
        idx = grouped[col].idxmax() if take_max else grouped[col].idxmin()
        extremes[title] = grouped.loc[idx]
    return extremes


def compute_run_extremes(df):
    """
    Compute single‐run extremes (no averaging).
    Returns dict: { title → pandas.Series(row) }
    """
    extremes = {}
    # best_fit: higher is better
    extremes["Best fit run"]  = df.loc[df.best_fit.idxmax()]
    extremes["Worst fit run"] = df.loc[df.best_fit.idxmin()]
    # best_step: fewer steps is “fastest”
    extremes["Fastest run"]   = df.loc[df.best_step.idxmin()]
    extremes["Slowest run"]   = df.loc[df.best_step.idxmax()]
    # path_len: shorter is better
    extremes["Shortest run"]  = df.loc[df.path_len.idxmin()]
    extremes["Longest run"]   = df.loc[df.path_len.idxmax()]
    return extremes


def write_report(ext_avgs: dict, ext_runs: dict, out_path: Path):
    # ensure the parent dir exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # open with explicit utf-8 encoding
    with out_path.open("w", encoding="utf-8") as f:
        # plain ASCII hyphens only
        f.write("=== AVERAGE-BASED EXTREMES (per combo) ===\n\n")
        for title, row in ext_avgs.items():
            f.write(f"--- {title} ---\n")
            params = ", ".join(f"{k}={row[k]}" for k in PARAMS)
            f.write(f"combo: {params}\n")
            f.write(f"  avg_fit:   {row.avg_fit:.3f}\n")
            f.write(f"  avg_steps: {row.avg_steps:.1f}\n")
            f.write(f"  avg_path:  {row.avg_path:.1f}\n\n")

        f.write("=== SINGLE-RUN EXTREMES ===\n\n")
        for title, row in ext_runs.items():
            f.write(f"--- {title} ---\n")
            params = ", ".join(f"{k}={row[k]}" for k in PARAMS)
            f.write(f"{params}, trial={int(row.trial)}\n")
            f.write(f"  best_fit:   {float(row.best_fit):.3f}\n")
            f.write(f"  best_step:  {float(row.best_step):.1f}\n")
            f.write(f"  path_len:   {float(row.path_len):.1f}\n\n")

    print(f"Report written to {out_path}")

# ─────────── defaults ───────────
N_BEES_LIST        = [20, 30, 50, 75]
SCOUT_FRAC_LIST    = [0.10, 0.15, 0.25, 0.50]
EMPLOYED_FRAC_LIST = [0.10, 0.25, 0.50, 0.75]
EXTRA_FOOD_LIST    = [0, 3, 5, 8]

DEFAULTS = {
    "n_bees": N_BEES_LIST[2],         # 50
    "scout_frac": SCOUT_FRAC_LIST[2], # 0.25
    "employed_frac": EMPLOYED_FRAC_LIST[2], # 0.50
    "extra_food": EXTRA_FOOD_LIST[2], # 5
}

PARAMS  = ["n_bees","scout_frac","employed_frac","extra_food"]
METRICS = {
    "best_fit":   ("mean_fit","min_fit","max_fit","Average Best Fit"),
    "best_step":  ("mean_step","min_step","max_step","Average Steps"),
    "path_len":   ("mean_path","min_path","max_path","Average Path Length"),
}

LINE_OUT = Path("results")/"lineplots"


# ─────────── Line plot generator ───────────
def plot_metric_vs(summary_df: pd.DataFrame,
                   x_param: str,
                   hue_param: str,
                   mean_col: str,
                   min_col: str,
                   max_col: str,
                   ylabel: str,
                   fixed_params: dict,
                   out_file: Path):
    df = summary_df.copy()
    x_vals = sorted(df[x_param].unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ['o','s','D','^','v','x','*']

    # Shade the overall min→max range
    global_min = df[min_col].min()
    global_max = df[max_col].max()
    ax.fill_between(x_vals, global_min, global_max,
                    color='grey', alpha=0.1)

    # Plot mean line + min/max points for each hue level
    for idx, level in enumerate(sorted(df[hue_param].unique())):
        sub = df[df[hue_param] == level]
        # means, minima, maxima per x
        means = sub.groupby(x_param)[mean_col].mean().reindex(x_vals)
        mins  = sub.groupby(x_param)[min_col].min().reindex(x_vals)
        maxs  = sub.groupby(x_param)[max_col].max().reindex(x_vals)

        line, = ax.plot(x_vals, means,
                        label=f"{hue_param}={level}",
                        marker=markers[idx % len(markers)])
        color = line.get_color()

        # overlay min / max markers
        ax.scatter(x_vals, mins, marker='v', color=color, s=30, zorder=3)
        ax.scatter(x_vals, maxs, marker='^', color=color, s=30, zorder=3)

    # axes labels
    ax.set_xticks(x_vals)
    ax.set_xticklabels([str(v) for v in x_vals])
    ax.set_xlabel(x_param)
    ax.set_ylabel(ylabel)

    # title with fixed defaults
    defaults_txt = ", ".join(f"{k}={v}" for k, v in fixed_params.items())
    ax.set_title(f"Effect of {x_param} on {ylabel}\nFixed: {defaults_txt}")

    # legend to the right, no title
    legend = ax.legend(loc='center left',
                       bbox_to_anchor=(1.02, 0.5),
                       frameon=False)
    legend.set_title(None)

    ax.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved line plot to {out_file}")

# ─────────── glue ───────────
def generate_line_plots(master_csv: Path):
    df = pd.read_csv(master_csv)
    summary = (
        df
        .groupby(PARAMS)
        .agg(
            mean_fit   = ("best_fit",  "mean"),
            min_fit    = ("best_fit",  "min"),
            max_fit    = ("best_fit",  "max"),
            mean_step  = ("best_step", "mean"),
            min_step   = ("best_step", "min"),
            max_step   = ("best_step", "max"),
            mean_path  = ("path_len",  "mean"),
            min_path   = ("path_len",  "min"),
            max_path   = ("path_len",  "max"),
        )
        .reset_index()
    )

    for metric_key, (mean_c, min_c, max_c, ylabel) in METRICS.items():
        out_dir = LINE_OUT / metric_key
        for x_param in PARAMS:
            for hue_param in PARAMS:
                if x_param == hue_param:
                    continue

                # build fixed_params = the two not in (x_param, hue_param)
                fixed = {
                    p: DEFAULTS[p]
                    for p in PARAMS
                    if p not in (x_param, hue_param)
                }

                # filter summary to only those defaults
                filt = summary
                for p, val in fixed.items():
                    filt = filt[filt[p] == val]

                out_file = out_dir / f"{hue_param}_vs_{x_param}.png"
                plot_metric_vs(filt,
                               x_param, hue_param,
                               mean_c, min_c, max_c,
                               ylabel,
                               fixed,
                               out_file)
                
def plot_non_solved_counts(df_all: pd.DataFrame,
                           param: str,
                           defaults: dict,
                           values_list: list,
                           mode: str = "default",
                           out_file: Path | None = None):
    """
    Bars = number of runs where solved==0, for each value in values_list,
    holding other params at defaults (mode='default') or aggregating all (mode='aggregate').
    """
    # 1) filter non-solved
    df_ns = df_all[df_all["solved"] == 0]
    if mode == "default":
        for p, v in defaults.items():
            if p != param:
                df_ns = df_ns[df_ns[p] == v]

    # 2) count & reindex
    counts = df_ns.groupby(param).size().reindex(values_list, fill_value=0)

    # 3) categorical positions for uniform spacing
    positions = np.arange(len(values_list))
    width = 0.6

    # 4) plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(positions, counts.values, width=width)
    ax.set_xticks(positions)
    ax.set_xticklabels(values_list, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xlabel(param, fontsize=16)
    ax.set_ylabel("Non-solved run count", fontsize=16)

    title = f"Non-solved Runs vs {param}"
    title += " (defaults)" if mode=="default" else " (aggregated)"
    ax.set_title(title, fontsize=18)

    # 5) tighten things up
    ax.margins(x=0.05)                    # small padding on left/right
    plt.tight_layout(pad=0.5)            # less white space around figure

    # 6) save or show
    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=300)
        print(f"Saved: {out_file}")
    else:
        plt.show()

    plt.close(fig)

def main():
    df = pd.read_csv(MASTER_CSV)
    df_solved = df[df["solved"] == 1].copy()
    # Heatmaps
    # for metric in METRICS:
    #     metric_dir = BASE_DIR / metric
    #     metric_dir.mkdir(parents=True, exist_ok=True)
    #     for x_param, y_param in combinations(PARAMS, 2):
    #         plot_and_save(df_solved, x_param, y_param, metric, metric_dir)

    # Line plots
    # generate_line_plots(MASTER_CSV)
    out_dir = Path("results/non")
    params_to_plot = [
        ("n_bees",        N_BEES_LIST),
        ("scout_frac",    SCOUT_FRAC_LIST),
        ("employed_frac", EMPLOYED_FRAC_LIST),
    ]
    for param, values in params_to_plot:
        # default‐mode
        # out_def = out_dir / f"{param}_non_solved_default.png"
        # plot_non_solved_counts(df, param, DEFAULTS, values,
        #                        mode="default", out_file=out_def)

        # aggregate‐mode
        out_agg = out_dir / f"{param}_non_solved_aggregate.png"
        plot_non_solved_counts(df, param, DEFAULTS, values,
                               mode="aggregate", out_file=out_agg)

    # # Reports
    # avgs = compute_extremes(df)
    # runs = compute_run_extremes(df)
    # write_report(avgs, runs, REPORT_TXT)


if __name__ == "__main__":
    main()
