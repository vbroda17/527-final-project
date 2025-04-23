#!/usr/bin/env python3
# bee/viz.py

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.lines import Line2D

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
def plot_best(colony, fname="best_path.png", out_dir=None):
    """
    Static plot: best path overlaid on orbits, with start & dest markers.
    """
    path = os.path.join(out_dir, fname) if out_dir else fname
    os.makedirs(os.path.dirname(path), exist_ok=True)

    sim       = colony.sim
    best      = colony.best_path
    args      = colony.args
    start_idx = sim.grav.get_body(args.start)[0] if isinstance(args.start, str) else None
    dest_idx  = sim.grav.get_body(args.dest)[0]

    fig, ax = plt.subplots(figsize=(6,6))
    fig.patch.set_facecolor('k')
    ax.set_facecolor('k')

    # static orbits
    cols = cm.plasma(np.linspace(0.2, 0.9, len(sim.grav.bodies)))
    for traj, col in zip(sim.grav.traj[:,:,:2], cols):
        ax.plot(traj[:,0], traj[:,1], '--', color=col, alpha=0.3)
        ax.scatter(traj[0,0], traj[0,1], color=col, s=20)

    # best path
    ax.plot(best[:,0], best[:,1], '--', color='magenta', lw=2, label='best path')
    ax.scatter(best[0,0], best[0,1], color='lime', marker='s', label='start')
    ax.scatter(best[-1,0], best[-1,1], color='red', marker='*', label='end')

    # planet markers
    if start_idx is not None:
        spt = sim.grav.traj[start_idx,0,:2]
        ax.scatter(spt[0], spt[1], color='lime', marker='s', s=60,
                   label='start planet')
    dpt = sim.grav.traj[dest_idx,0,:2]
    ax.scatter(dpt[0], dpt[1], color='cyan', marker='D', s=60,
               label='dest planet')

    ax.set_aspect('equal')
    ax.set_xlabel('AU')
    ax.set_ylabel('AU')
    ax.legend(loc='upper right', fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)

# -------------------------------------------------------------
def animate_colony(colony, fname="colony.gif", out_dir=None, show_orbits=False):
    """
    Animated GIF: 
      • static orbit traces for all bodies
      • destination orbit colored by angular proximity
      • bonus hotspots
      • bee trails & markers by role
      • start/dest markers
      • horizontal colorbar legend
    """
    out_path = os.path.join(out_dir, fname) if out_dir else fname
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    sim   = colony.sim
    orbit = colony.orbit           # (N,2)
    N     = orbit.shape[0]
    steps = len(colony.history)

    # precompute angular weights (1 at dest, 0 opposite)
    dest_idx, _ = sim.grav.get_body(colony.args.dest)
    angles      = np.linspace(0, 2*np.pi, N, endpoint=False)
    ang_dest    = angles[dest_idx]
    delta       = np.abs((angles - ang_dest + np.pi) % (2*np.pi) - np.pi)
    weights     = 1.0 - delta/np.pi

    # build line collection for orbit
    segs = np.stack([orbit[:-1], orbit[1:]], axis=1)
    lc   = LineCollection(segs,
                          cmap=cm.viridis,
                          norm=plt.Normalize(0,1),
                          linewidth=2)
    lc.set_array(weights[:-1])

    # figure setup
    fig, ax = plt.subplots(figsize=(6,6))
    fig.patch.set_facecolor('k')
    ax.set_facecolor('k')
    ax.add_collection(lc)

    # static orbit traces + planet markers
    cols = cm.plasma(np.linspace(0.2,0.9,len(sim.grav.bodies)))
    planet_pts = []
    for traj, col in zip(sim.grav.traj[:,:,:2], cols):
        ax.plot(traj[:,0], traj[:,1], '--', color=col, alpha=0.3)
        pt, = ax.plot([], [], 'o', color=col, ms=4)
        planet_pts.append((traj, pt))

    # start & dest markers
    args = colony.args
    if isinstance(args.start, str):
        si, _ = sim.grav.get_body(args.start)
        spt   = sim.grav.traj[si,0,:2]
        ax.scatter(spt[0], spt[1], color='lime', marker='s', s=60,
                   label='start planet')
    di, _ = sim.grav.get_body(args.dest)
    dpt   = sim.grav.traj[di,0,:2]
    ax.scatter(dpt[0], dpt[1], color='cyan', marker='D', s=60,
               label='dest planet')

    # bonus hotspots
    if colony.bonus_pts.size:
        ax.scatter(colony.bonus_pts[:,0], colony.bonus_pts[:,1],
                   s=50, color='yellow', marker='*', label='bonus')

    # bees: trails & markers
    roles = [b.role for b in colony.bees]
    unique = list(dict.fromkeys(roles))
    rcmap  = cm.tab10(np.linspace(0,1,len(unique)))
    role_col = dict(zip(unique, rcmap))

    trails, markers = [], []
    for b in colony.bees:
        c = role_col[b.role]
        ln, = ax.plot([], [], '-', lw=1, color=c, alpha=0.7)
        pt, = ax.plot([], [], 'o', ms=6, color=c)
        trails.append(ln)
        markers.append(pt)

    # legend (roles + bonus + start/end)
    handles = [Line2D([],[],marker='o',color=role_col[r],linestyle='None',label=r)
               for r in unique]
    if colony.bonus_pts.size:
        handles.append(Line2D([],[],marker='*',color='yellow',linestyle='None',label='bonus'))
    handles.extend([
        Line2D([],[],marker='s',color='lime',linestyle='None',label='start'),
        Line2D([],[],marker='D',color='cyan',linestyle='None',label='dest')
    ])
    ax.legend(handles=handles, loc='upper right', fontsize=8)

    # colorbar
    cbar = fig.colorbar(cm.ScalarMappable(norm=plt.Normalize(0,1),
                                          cmap=cm.viridis),
                        ax=ax, orientation='horizontal', pad=0.05)
    cbar.set_label('Arc distance (1=dest, 0=opposite)')

    # axes
    lim = np.max(np.abs(sim.grav.traj[:,:,:2])) * 1.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('AU')
    ax.set_ylabel('AU')

    # animation callbacks
    def init():
        for traj, pt in planet_pts:
            pt.set_data([], [])
        for ln, pt in zip(trails, markers):
            ln.set_data([], [])
            pt.set_data([], [])
        return [lc] + [pt for _,pt in planet_pts] + trails + markers

    def update(i):
        # move planet markers if requested
        for traj, pt in planet_pts:
            idx = min(i, traj.shape[0]-1) if show_orbits else 0
            xy  = traj[idx]
            pt.set_data([xy[0]], [xy[1]])

        # bees
        for b, ln, pt in zip(colony.bees, trails, markers):
            path = np.vstack(b.path)
            if i < path.shape[0]:
                ln.set_data(path[:i+1,0], path[:i+1,1])
                pt.set_data([path[i,0]], [path[i,1]])

        return [lc] + [pt for _,pt in planet_pts] + trails + markers

    ani = FuncAnimation(fig, update,
                        frames=steps,
                        init_func=init,
                        interval=50,
                        blit=True)
    ani.save(out_path, writer='pillow', fps=20)
    plt.close(fig)
