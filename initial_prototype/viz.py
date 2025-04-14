import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

plt.style.use("dark_background")

# -------------------------------- colour map for bodies -----------------------------
COL = {
    "Sun":      "#ffcc00",
    "Start":    "#20d67b",
    "Target":   "#00bfff",
    "Obstacle": "#ff5033",
}

# -------------------------------- helpers -------------------------------------------

def orbit(ax, body, k: float = 1.0, t_sample: float = 0.0):
    """Draw a faint circular orbit for *body* (static‑pos bodies only)."""
    if body.name == "Sun":
        return  # Sun at origin, no orbit ring
    if body.orbit_func is None:
        r = np.linalg.norm(body.pos(t_sample))
        θ = np.linspace(0, 2*np.pi, 400)
        ax.plot(k*r*np.cos(θ), k*r*np.sin(θ), lw=1, color=COL.get(body.name, "#888"), alpha=.25)


def bodies_plot(ax, bodies, t: float = 0.0):
    for b in bodies:
        p = b.pos(t)
        col = COL.get(b.name, "#aaaaaa")
        size = 150 if b.name == "Sun" else 60
        ax.scatter(*p, s=size, color=col, edgecolor="k", zorder=4, label=b.name)


def gradient_line(ax, pts: np.ndarray, cmap: str = "turbo", add_cbar: bool = True):
    """Draw dotted path with smooth gradient & optional colour‑bar (time)."""
    segs = np.concatenate([pts[:-1, None, :], pts[1:, None, :]], axis=1)
    norm = Normalize(vmin=0, vmax=len(pts)-1)
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=2, linestyle=":")
    lc.set_array(np.arange(len(pts)))
    ax.add_collection(lc)
    if add_cbar:
        cbar = plt.colorbar(lc, ax=ax, orientation="vertical", fraction=.045, pad=.03)
        cbar.set_label("Time →", rotation=270, labelpad=15)
    return lc

# -------------------------------- top‑level plots -----------------------------------

def best_plot(run_dir, bodies, best_pts: np.ndarray):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")

    for b in bodies:
        orbit(ax, b)
    bodies_plot(ax, bodies)
    gradient_line(ax, best_pts, cmap="turbo", add_cbar=True)

    ax.set_title("Optimal path")
    # move legend outside
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="small")
    fig.tight_layout()
    fig.savefig(run_dir / "best_path.png", dpi=150)
    plt.show()


def swarm_final(run_dir, bodies, swarm_pts):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    for b in bodies:
        orbit(ax, b)
    bodies_plot(ax, bodies)
    for pts in swarm_pts:
        ax.plot(pts[:, 0], pts[:, 1], lw=0.4, color="#666", alpha=0.5)
    ax.set_title("Swarm exploration – final iter")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="small")
    fig.tight_layout()
    fig.savefig(run_dir / "swarm_final.png", dpi=150)
    plt.show()


def training_curve(run_dir, hist):
    fig, ax = plt.subplots()
    ax.plot(hist)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best cost")
    ax.set_title("Training curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "training_curve.png", dpi=150)
    plt.show()


def snapshot_plots(train_dir, bodies, snaps, pct):
    """Save colour‑coded swarm snapshots to *train_dir*."""
    import random
    cmap = cm.get_cmap("viridis", len(snaps))
    for i, swarm in enumerate(snaps):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect("equal")
        for b in bodies:
            orbit(ax, b)
        bodies_plot(ax, bodies)
        col = cmap(i)
        if 0 < pct < 1:
            keep = random.sample(swarm, int(len(swarm) * pct))
        else:
            keep = random.sample(swarm, min(int(pct), len(swarm)))
        for pts in keep:
            ax.plot(pts[:, 0], pts[:, 1], lw=0.5, color=col, alpha=0.6)
        ax.set_title(f"Swarm iteration {i}")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="x-small")
        fig.tight_layout()
        fig.savefig(train_dir / f"iter_{i}.png", dpi=120)
        plt.close(fig)