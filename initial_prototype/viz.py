import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

plt.style.use("dark_background")

# colour palette (extend freely)
COL = {
    "Sun": "#ffcc00",
    "Start": "#20d67b",
    "Target": "#00bfff",
    "Obstacle": "#ff5033",
}

# --------------------------------------------------------------------- #

def _orbit(ax, r: float, col: str, ctr=(0, 0), k: float = 1.0):
    """Draw a circular orbit of radius *r* around *ctr*."""
    θ = np.linspace(0, 2 * np.pi, 400)
    ax.plot(ctr[0] + k * r * np.cos(θ), ctr[1] + k * r * np.sin(θ), lw=1, color=col, alpha=0.25)


def _bodies(ax, planets):
    for p in planets:
        col = COL.get(p.name, "#aaaaaa")
        ax.scatter(*p.pos, s=150 if p.name == "Sun" else 50, color=col, edgecolor="k", zorder=4, label=p.name)


# --------------------------------------------------------------------- #

def plot_best(planets, best_path, cost, save=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")

    # orbits
    for p in planets:
        if p.name != "Sun":
            _orbit(ax, np.linalg.norm(p.pos), COL.get(p.name, "#888"))

    _bodies(ax, planets)

    # best path (dotted white)
    pts = np.vstack(best_path)
    ax.plot(pts[:, 0], pts[:, 1], ls=":", lw=2.5, color="white", label="best")

    ax.legend(fontsize="small")
    ax.set_title("Optimal path")
    ax.text(0.02, 0.97, f"Cost = {cost:.2e}", transform=ax.transAxes, ha="left", va="top", fontsize="small", color="white")

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()


def plot_swarm_iter(planets, snapshots, pct=0.25, save=None):
    """Colour‑coded swarm trajectories (iteration → colour).

    pct : float 0‑1 → fraction of bees to draw per snapshot
          int  >1  → maximum number of bees to draw per snapshot
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")

    for p in planets:
        if p.name != "Sun":
            _orbit(ax, np.linalg.norm(p.pos), COL.get(p.name, "#888"))
    _bodies(ax, planets)

    n_snaps = len(snapshots)
    cmap = cm.get_cmap("viridis", n_snaps)

    import random

    for i, swarm in enumerate(snapshots):
        colour = cmap(i)
        if 0 < pct < 1:
            keep = random.sample(swarm, int(len(swarm) * pct))
        else:
            keep = random.sample(swarm, min(int(pct), len(swarm)))
        for path in keep:
            pts = np.vstack(path)
            ax.plot(pts[:, 0], pts[:, 1], lw=0.5, color=colour, alpha=0.6)

    ax.set_title("Swarm exploration (colour = iteration)")
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()


def plot_history(history, save=None):
    fig, ax = plt.subplots()
    ax.plot(history, lw=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best cost")
    ax.set_title("Training curve")
    ax.grid(alpha=0.3)
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()