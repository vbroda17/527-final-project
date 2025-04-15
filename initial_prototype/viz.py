import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

plt.style.use("dark_background")

COLOURS = {
    "Sun": "#ffcc00",
    "Start": "#20d67b",
    "Target": "#00bfff",
    "Obstacle": "#ff5033",
}


# -----------------------------------------------------------------------------
# Helper drawing functions
# -----------------------------------------------------------------------------

def draw_orbit(ax, body, t0: float = 0.0, k: float = 1.0) -> None:
    if body.name == "Sun":
        return
    if body.orbit_func is not None:
        return  # skipping moving orbits for now
    radius = np.linalg.norm(body.pos(t0))
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    x = k * radius * np.cos(theta)
    y = k * radius * np.sin(theta)
    ax.plot(x, y, lw=1.0, color=COLOURS.get(body.name, "#888888"), alpha=0.25)


def draw_bodies(ax, bodies, t: float = 0.0) -> None:
    for body in bodies:
        p = body.pos(t)
        colour = COLOURS.get(body.name, "#aaaaaa")
        size = 150 if body.name == "Sun" else 60
        ax.scatter(*p, s=size, color=colour, edgecolor="k", zorder=4, label=body.name)


def gradient_path(ax, points: np.ndarray, cmap: str = "turbo") -> None:
    segments = np.concatenate([points[:-1, None, :], points[1:, None, :]], axis=1)
    norm = Normalize(vmin=0, vmax=len(points) - 1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=2, linestyle=":")
    lc.set_array(np.arange(len(points)))
    ax.add_collection(lc)
    cbar = plt.colorbar(lc, ax=ax, orientation="vertical", fraction=0.045, pad=0.03)
    cbar.set_label("Time →", rotation=270, labelpad=15)


# -----------------------------------------------------------------------------
# Public plot helpers
# -----------------------------------------------------------------------------

def save_best_plot(run_dir, bodies, best_points: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")

    for body in bodies:
        draw_orbit(ax, body)
    draw_bodies(ax, bodies)
    gradient_path(ax, best_points)

    ax.set_title("Optimal path")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="small")
    fig.tight_layout()
    fig.savefig(run_dir / "best_path.png", dpi=150)
    plt.show()


def save_training_curve(run_dir, history: list[float]) -> None:
    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best cost")
    ax.set_title("Training curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "training_curve.png", dpi=150)
    plt.show()


def save_swarm_snapshots(train_dir, bodies, snapshots, pct: float) -> None:
    cmap = cm.get_cmap("viridis", len(snapshots))
    for i, swarm in enumerate(snapshots):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect("equal")
        for body in bodies:
            draw_orbit(ax, body)
        draw_bodies(ax, bodies)
        colour = cmap(i)
        if 0.0 < pct < 1.0:
            import random
            keep = random.sample(swarm, int(len(swarm) * pct))
        else:
            keep = swarm
        for path_points in keep:
            ax.plot(path_points[:, 0], path_points[:, 1], lw=0.5, color=colour, alpha=0.6)
        ax.set_title(f"Swarm iteration {i}")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="x-small")
        fig.tight_layout()
        fig.savefig(train_dir / f"iter_{i}.png", dpi=120)
        plt.close(fig)