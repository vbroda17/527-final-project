import matplotlib.pyplot as plt
import numpy as np

plt.style.use("dark_background")

# colour palette you can tweak freely
PLANET_COLOURS = {
    "Sun":      "#ffcc00",
    "Mercury":  "#e6d300",
    "Venus":    "#00bfff",
    "Earth":    "#20d67b",
    "Mars":     "#ff5033",
    "Obstacle": "#ff5033",   # fallback for demo
    "Target":   "#00bfff",   #  “
    "Start":    "#20d67b",   #  “
}

def draw_orbit(ax, centre, radius, colour, ls="-", alpha=0.3):
    θ = np.linspace(0, 2*np.pi, 400)
    x = centre[0] + radius*np.cos(θ)
    y = centre[1] + radius*np.sin(θ)
    ax.plot(x, y, ls=ls, color=colour, alpha=alpha, lw=1.0)

def plot_scene(planets, best_path, all_paths=None,
               show_orbits=True, orbit_scale=1.1, title=""):
    """
    planets    – iterable of Planet objects (must have .name, .pos, .radius)
    best_path  – list of np.array way‑points
    all_paths  – optional list[list[points]] to show exploration
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    # 1. draw optional orbits (circular approximation)
    if show_orbits:
        for p in planets:
            if p.name == "Sun":           # Sun sits at origin
                continue
            r = np.linalg.norm(p.pos)
            colour = PLANET_COLOURS.get(p.name, "#888888")
            draw_orbit(ax, (0,0), r*orbit_scale, colour)

    # 2. draw planets
    for p in planets:
        colour = PLANET_COLOURS.get(p.name, "#ffffff")
        ax.scatter(*p.pos, s=50 if p.name!="Sun" else 150,
                   color=colour, label=p.name, zorder=4, edgecolor="k")

    # 3. (optional) swarm exploration
    if all_paths is not None:
        for path in all_paths:
            pts = np.vstack(path)
            ax.plot(pts[:,0], pts[:,1], lw=0.4, color="#666666", alpha=0.4)

    # 4. best path – dotted white
    pts = np.vstack(best_path)
    ax.plot(pts[:,0], pts[:,1], ls=":", lw=2.5, color="white",
            label="optimal path", zorder=5)

    ax.legend(loc="upper right", fontsize="small")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
