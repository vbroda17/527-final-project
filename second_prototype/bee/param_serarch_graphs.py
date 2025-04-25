#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────────
# Setup result directories
# ────────────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
LOG_DIR     = os.path.join(RESULTS_DIR, "logs")
HEATMAP_DIR = os.path.join(RESULTS_DIR, "heatmaps")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)

# always overwrite a single analysis log
LOG_FILE = os.path.join(LOG_DIR, "analysis.txt")

def log(msg: str, f):
    """Print to stdout and write to log file."""
    print(msg)
    f.write(msg + "\n")

# ────────────────────────────────────────────────────────────────────────
# Data-loading and analysis functions
# ────────────────────────────────────────────────────────────────────────
def load_master(path="tests/master_results.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df['solved'] = df['solved'].astype(bool)
    return df

def compute_individual_metrics(df: pd.DataFrame) -> dict:
    solved = df[df['solved']]
    return {
        'best_fit':       solved.loc[solved.best_fit.idxmax()],
        'fewest_steps':   solved.loc[solved.best_step.idxmin()],
        'shortest_path':  solved.loc[solved.path_len.idxmin()],
        'worst_fit':      solved.loc[solved.best_fit.idxmin()],
        'most_steps':     solved.loc[solved.best_step.idxmax()],
        'longest_path':   solved.loc[solved.path_len.idxmax()]
    }

def compute_average_metrics(df: pd.DataFrame) -> dict:
    solved = df[df['solved']]
    grp = (solved
           .groupby(['n_bees','scout_frac','employed_frac','extra_food'], as_index=False)
           .mean())
    return {
        'best_avg_fit':      grp.loc[grp.best_fit.idxmax()],
        'lowest_avg_steps':  grp.loc[grp.best_step.idxmin()],
        'lowest_avg_path':   grp.loc[grp.path_len.idxmin()],
        'worst_avg_fit':     grp.loc[grp.best_fit.idxmin()],
        'highest_avg_steps': grp.loc[grp.best_step.idxmax()],
        'highest_avg_path':  grp.loc[grp.path_len.idxmax()],
        'grouped':           grp
    }

def plot_heatmap(df: pd.DataFrame,
                 row_param: str,
                 col_param: str,
                 value_param: str,
                 title: str,
                 save_path: str,
                 cmap: str = 'viridis'):
    pivot = df.pivot_table(
        index=row_param,
        columns=col_param,
        values=value_param,
        aggfunc='mean'
    )
    fig, ax = plt.subplots(figsize=(8,6))
    cax = ax.imshow(pivot.values, aspect='auto', cmap=cmap)

    # ticks & labels
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # colorbar
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label(value_param)

    # annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i,j]:.2f}",
                    ha="center", va="center", color="white", fontsize=8)

    ax.set_xlabel(col_param)
    ax.set_ylabel(row_param)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# ────────────────────────────────────────────────────────────────────────
# Main analysis
# ────────────────────────────────────────────────────────────────────────
def main():
    df  = load_master()
    ind = compute_individual_metrics(df)
    avg = compute_average_metrics(df)

    # Log individual & average metrics
    with open(LOG_FILE, "w") as f:
        log("=== Individual Runs ===", f)
        for name, series in ind.items():
            log(f"-- {name} --", f)
            log(series.to_string(), f)
            log("", f)

        log("=== Averages per Combo ===", f)
        for name, series in avg.items():
            if name != 'grouped':
                log(f"-- {name} --", f)
                log(series.to_string(), f)
                log("", f)

    # Heatmap configurations:
    row_params  = ['scout_frac', 'employed_frac']
    metrics     = ['best_fit', 'best_step', 'path_len']

    for row in row_params:
        for metric in metrics:
            save_name = f"avg_{metric}_{row}_vs_n_bees.png"
            title     = f"Avg {metric.replace('_',' ').title()} by {row} vs n_bees"
            plot_heatmap(
                avg['grouped'],
                row_param=row,
                col_param='n_bees',
                value_param=metric,
                title=title,
                save_path=os.path.join(HEATMAP_DIR, save_name)
            )

if __name__ == "__main__":
    main()
