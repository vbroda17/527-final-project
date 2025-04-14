import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

plt.style.use("dark_background")

COL = {
    "Sun":      "#ffcc00",
    "Start":    "#20d67b",
    "Target":   "#00bfff",
    "Obstacle": "#ff5033",
}

def _orbit(ax, r, col, ctr=(0,0), k=1.0):
    theta = np.linspace(0, 2*np.pi, 400)
    x = ctr[0] + k*r*np.cos(theta)
    y = ctr[1] + k*r*np.sin(theta)
    ax.plot(x, y, lw=1, color=col, alpha=.25)

def _bodies(ax, planets):
    for p in planets:
        col = COL.get(p.name,"#aaaaaa")
        size = 150 if p.name=="Sun" else 50
        ax.scatter(*p.pos, s=size, color=col, edgecolor="k",
                   zorder=4, label=p.name)

def plot_best(planets, best_path, best_cost, save=None):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect("equal")
    for p in planets:
        if p.name != "Sun":
            _orbit(ax, np.linalg.norm(p.pos), COL.get(p.name,"#888"), k=1.0)
    _bodies(ax, planets)

    # best path dotted line
    pts = np.vstack([seg.point if hasattr(seg, 'point') else seg
                     for seg in best_path])
    ax.plot(pts[:,0], pts[:,1], ls=":", lw=2.5, color="white", label="best")

    ax.legend(fontsize="small")
    ax.set_title("Optimal path")
    ax.text(0.02, 0.97, f"Cost = {best_cost:.2e}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize="small", color="white")
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()

def plot_history(history, save=None):
    fig, ax = plt.subplots()
    ax.plot(history, lw=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best cost")
    ax.set_title("Training curve")
    ax.grid(alpha=.3)
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()

def plot_swarm_final(planets, swarm_paths, save=None):
    """
    Swarm at final iteration (each path in grey).
    """
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect("equal")

    for p in planets:
        if p.name != "Sun":
            _orbit(ax, np.linalg.norm(p.pos), COL.get(p.name,"#888"), k=1.0)
    _bodies(ax, planets)

    for path in swarm_paths:
        pts = np.vstack([seg.point if hasattr(seg, 'point') else seg
                         for seg in path])
        ax.plot(pts[:,0], pts[:,1], lw=0.4, color="#666666", alpha=0.5)

    ax.set_title("Swarm exploration (final iteration)")
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()

def plot_swarm_snapshots(planets, snapshots, pct=0.25, outdir=None):
    """
    Color-coded by iteration. Optionally save each snapshot image to 'outdir'.
    - snapshots is a list of [paths], each a list of segments.
    - We sub-sample each snapshot's swarm with 'pct'.
    """
    # if outdir is specified, we skip calling plt.show() for each iteration
    # and only show the last plot to avoid spamming windows.
    n_snaps = len(snapshots)
    cmap = cm.get_cmap("viridis", n_snaps)

    for i, swarm in enumerate(snapshots):
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_aspect("equal")
        for p in planets:
            if p.name != "Sun":
                _orbit(ax, np.linalg.norm(p.pos), COL.get(p.name,"#888"), k=1.0)
        _bodies(ax, planets)

        color = cmap(i)
        # pick a fraction or max number from the swarm
        if 0 < pct < 1:
            import random
            keep = random.sample(swarm, int(len(swarm)*pct))
        else:
            # interpret as absolute max
            keep = swarm
            if len(keep) > pct:
                import random
                keep = random.sample(keep, int(pct))

        for path in keep:
            pts = np.vstack([seg.point if hasattr(seg, 'point') else seg
                             for seg in path])
            ax.plot(pts[:,0], pts[:,1], lw=0.5, color=color, alpha=0.6)

        ax.set_title(f"Swarm at iteration {i}")

        if outdir:
            os.makedirs(outdir, exist_ok=True)
            fpath = os.path.join(outdir, f"iter_{i}.png")
            fig.savefig(fpath, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    # If we saved all figures, optionally show the last snapshot
    if outdir and snapshots:
        i = len(snapshots)-1
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_aspect("equal")
        for p in planets:
            if p.name != "Sun":
                _orbit(ax, np.linalg.norm(p.pos), COL.get(p.name,"#888"), k=1.0)
        _bodies(ax, planets)
        color = cmap(i)
        keep = snapshots[-1]
        if 0 < pct < 1:
            import random
            keep = random.sample(keep, int(len(keep)*pct))
        else:
            if len(keep) > pct:
                import random
                keep = random.sample(keep, int(pct))

        for path in keep:
            pts = np.vstack([seg.point if hasattr(seg, 'point') else seg
                             for seg in path])
            ax.plot(pts[:,0], pts[:,1], lw=0.5, color=color, alpha=0.6)

        ax.set_title(f"Swarm at iteration {i}")
        plt.show()
