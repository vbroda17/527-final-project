import os, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.lines import Line2D

# -------------------------------------------------------------
# bee/viz.py
# Visualization for BeeColony optimization
# -------------------------------------------------------------

# -------------------------------------------------------------
# Save fitness history
# -------------------------------------------------------------
def save_fitness(colony, filename="fitness.csv"):
    """
    Writes the colony's best-fitness history to a CSV file:
    step,fitness
    """
    import csv
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'fitness'])
        for i, fit in enumerate(colony.history):
            writer.writerow([i, fit])

# -------------------------------------------------------------
# Plot static best path
# -------------------------------------------------------------
def plot_best(colony, fname="best_path.png", out_dir=None):
    """
    Saves a static plot of the best-performing path overlaid on orbits.
    """
    path = os.path.join(out_dir, fname) if out_dir else fname
    os.makedirs(os.path.dirname(path), exist_ok=True)

    sim  = colony.sim
    best = colony.best_path

    fig, ax = plt.subplots(figsize=(6,6))
    fig.patch.set_facecolor('k'); ax.set_facecolor('k')

    # static orbits of all bodies
    cols_all = cm.plasma(np.linspace(0.2, 0.9, len(sim.grav.bodies)))
    for traj, col in zip(sim.grav.traj[:,:,:2], cols_all):
        ax.plot(traj[:,0], traj[:,1], '--', color=col, alpha=0.3)
        ax.scatter(traj[0,0], traj[0,1], color=col, s=20)

    # best path dashed
    ax.plot(best[:,0], best[:,1], '--', color='magenta', lw=2, label='best path')
    ax.scatter(best[0,0], best[0,1], color='lime', marker='s', label='start')
    ax.scatter(best[-1,0], best[-1,1], color='red', marker='*', label='end')

    ax.set_aspect('equal'); ax.set_xlabel('AU'); ax.set_ylabel('AU')
    ax.legend(loc='upper right', fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)

# -------------------------------------------------------------
# Animate full colony + dynamic orbit coloring
# -------------------------------------------------------------
def animate_colony(colony, fname="colony.gif", out_dir=None, show_orbits=False):
    """
    Animates the full bee colony run with:
      • static orbits of all bodies
      • destination orbit colored by fitness (linear default: uniform)
      • bonus hotspots
      • bee trails & markers colored by role
      • legend of roles & bonus

    show_orbits=True will move planet markers along their orbits.
    """
    out_path = os.path.join(out_dir, fname) if out_dir else fname
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    sim   = colony.sim
    orbit = colony.orbit       # Nx2
    steps = len(colony.history)

    # prepare destination orbit segments (no wrap)
    segs = np.stack([orbit[:-1], orbit[1:]], axis=1)
    lc   = LineCollection(segs, cmap=cm.viridis, norm=plt.Normalize(0,1), linewidth=2)
    lc.set_array(np.full(len(segs), 0.5))  # uniform color for linear

    # figure
    fig, ax = plt.subplots(figsize=(6,6))
    fig.patch.set_facecolor('k'); ax.set_facecolor('k')
    ax.add_collection(lc)

    # static orbits + optional moving planet markers
    cols_all   = cm.plasma(np.linspace(0.2,0.9,len(sim.grav.bodies)))
    planet_pts = []
    for traj, col in zip(sim.grav.traj[:,:,:2], cols_all):
        ax.plot(traj[:,0], traj[:,1], '--', color=col, alpha=0.3)
        pt, = ax.plot([], [], 'o', color=col, ms=4)
        planet_pts.append((traj, pt))

    # bonus hotspots
    if colony.bonus_pts.size:
        ax.scatter(colony.bonus_pts[:,0], colony.bonus_pts[:,1],
                   s=50, color='yellow', marker='*', label='bonus')

    # bees
    roles = [b.role for b in colony.bees]
    unique = list(dict.fromkeys(roles))
    rcols  = cm.tab10(np.linspace(0,1,len(unique)))
    cmap   = dict(zip(unique, rcols))

    trails, markers = [], []
    for b in colony.bees:
        c = cmap[b.role]
        ln, = ax.plot([], [], '-', lw=1, color=c, alpha=0.7)
        pt, = ax.plot([], [], 'o', ms=6, color=c)
        trails.append(ln); markers.append(pt)

    # legend
    handles = [Line2D([],[],marker='o',color=c,linestyle='None',label=role)
               for role,c in cmap.items()]
    if colony.bonus_pts.size:
        handles.append(Line2D([],[],marker='*',color='yellow',linestyle='None',label='bonus'))
    ax.legend(handles=handles, loc='upper right', fontsize=8)

    # axis limits
    lim = np.max(np.abs(sim.grav.traj[:,:,:2]))*1.2
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
    ax.set_aspect('equal'); ax.set_xlabel('AU'); ax.set_ylabel('AU')

    # fitness helper
    def _fit(dist):
        x = dist / colony.orbit_len
        kind,k = colony.args.fitness, colony.args.k
        if kind=='sigmoid': return 1.0/(1.0+math.exp(k*(x-0.5)))
        if kind=='log':     return 1.0-math.log1p(k*x)/math.log1p(k)
        if kind=='exp':     return math.exp(-k*x)
        return None  # linear

    def init():
        lc.set_array(np.full(len(segs), 0.5))
        for traj,pt in planet_pts: pt.set_data([],[])
        for ln,pt in zip(trails, markers): ln.set_data([],[]); pt.set_data([],[])
        return [lc] + [pt for _,pt in planet_pts] + trails + markers

    def update(i):
        # planet markers
        for traj,pt in planet_pts:
            if show_orbits:
                idx = min(i, traj.shape[0]-1); xy = traj[idx]
            else:
                xy = traj[0]
            pt.set_data([xy[0]],[xy[1]])

        # recolor if non-linear
        if colony.args.fitness != 'linear':
            best = colony.best_path[i]
            dists= np.linalg.norm(orbit - best, axis=1)
            fits = np.array([_fit(d) for d in dists[:-1]])
            lc.set_array(fits)

        # bees
        for b, ln, pt in zip(colony.bees, trails, markers):
            path = np.vstack(b.path)
            if i < path.shape[0]:
                ln.set_data(path[:i+1,0], path[:i+1,1])
                pt.set_data([path[i,0]],[path[i,1]])

        return [lc] + [pt for _,pt in planet_pts] + trails + markers

    ani = FuncAnimation(fig, update, frames=steps,
                        init_func=init, interval=50, blit=True)
    ani.save(out_path, writer='pillow', fps=20)
    plt.close(fig)
