import numpy as np, matplotlib.pyplot as plt, os
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib import cm
plt.style.use('dark_background')

OUT_DIR = "rocket_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- helper ----------------------------------------------
def _v_colors(path, speeds):
    norm = plt.Normalize(speeds.min(), speeds.max())
    cmap = cm.magma
    segs = np.concatenate([path[:-1,None,:], path[1:,None,:]], axis=1)
    lc   = LineCollection(segs, cmap=cmap, norm=norm, linewidth=1.5)
    lc.set_array(speeds)
    return lc, cmap, norm

# ---------------- static ----------------------------------------------
def static(sim, rocket, fname="start.png"):
    fname = os.path.join(OUT_DIR, fname)
    pos0 = sim.grav.traj[:,0]
    fig,ax = plt.subplots(figsize=(5,5))
    for j in range(len(sim.grav.bodies)):
        ax.plot(sim.grav.traj[j,:,0], sim.grav.traj[j,:,1], '--', alpha=0.25)
        ax.scatter(*pos0[j], s=15)
    ax.scatter(*rocket.path[0], color='magenta', marker='X', s=50)
    ax.scatter(0,0,color='y',s=70)
    ax.set_aspect('equal'); ax.set_xlabel("AU"); ax.set_ylabel("AU")
    fig.tight_layout(); fig.savefig(fname,dpi=160); plt.close(fig)

# ---------------- snapshot --------------------------------------------
def snapshot(sim, rocket, fname="snapshot.png"):
    fname = os.path.join(OUT_DIR, fname)
    path   = np.vstack(rocket.path)
    speeds = np.linalg.norm(np.diff(path,axis=0), axis=1) / sim.dt
    lc,cmap,norm = _v_colors(path, speeds)
    fig,ax = plt.subplots(figsize=(5,5))
    ax.add_collection(lc)
    ax.plot(*path[-1], 'X', color='magenta', ms=6)
    for j in range(len(sim.grav.bodies)):
        ax.plot(sim.grav.traj[j,:,0], sim.grav.traj[j,:,1], '--', alpha=0.15)
    ax.scatter(0,0,color='gold',s=90,zorder=5)
    lim=np.max(np.abs(sim.grav.traj[:,:,:2]))*1.2
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_aspect('equal')
    ax.set_xlabel("AU"); ax.set_ylabel("AU")
    fig.colorbar(cm.ScalarMappable(norm,cmap), ax=ax,label="speed (AU/day)")
    fig.tight_layout(); fig.savefig(fname,dpi=160); plt.close(fig)

# ---------------- animate ---------------------------------------------
def animate(sim, rocket, fname="flight.gif"):
    fname = os.path.join(OUT_DIR, fname)
    path   = np.vstack(rocket.path)
    speeds = np.linalg.norm(np.diff(path, axis=0), axis=1) / sim.dt
    norm   = plt.Normalize(speeds.min(), speeds.max())
    lc, cmap, _ = _v_colors(path, speeds)

    Np   = len(sim.grav.bodies)
    cols = plt.cm.plasma(np.linspace(0.2,0.9,Np))

    fig, ax = plt.subplots(figsize=(6,6))
    fig.patch.set_facecolor('k'); ax.set_facecolor('k')
    ax.add_collection(lc)
    for j,col in enumerate(cols):
        ax.plot(sim.grav.traj[j,:,0], sim.grav.traj[j,:,1],'--',
                color=col, alpha=0.25, lw=0.8)
    ax.scatter(0,0,color='gold',s=90,zorder=6)

    planet_pts = [ax.plot([],[],'o',color=col)[0] for col in cols]
    rocket_pt, = ax.plot([],[],'X',color='magenta',ms=6)

    lim=np.max(np.abs(sim.grav.traj[:,:,:2]))*1.3
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
    ax.set_aspect('equal'); ax.set_xlabel("AU"); ax.set_ylabel("AU")

    def init():
        lc.set_segments([]); rocket_pt.set_data([],[])
        for p in planet_pts: p.set_data([],[])
        return [lc, rocket_pt, *planet_pts]

    def update(i):
        for j,p in enumerate(planet_pts):
            p.set_data([sim.grav.traj[j,i,0]], [sim.grav.traj[j,i,1]])
        rocket_pt.set_data([path[i,0]],[path[i,1]])
        if i>0:
            lc.set_segments(np.concatenate([path[:i,None,:], path[1:i+1,None,:]], axis=1))
            lc.set_array(speeds[:i])
        return [lc, rocket_pt, *planet_pts]

    ani = FuncAnimation(fig, update, frames=len(path),
                        init_func=init, interval=25, blit=True)
    ani.save(fname, writer='pillow', fps=30)
    plt.close(fig)

# ---------- colour helper for single‑rocket plots ----------------------
def _v_colors(path, speeds):
    norm = plt.Normalize(speeds.min(), speeds.max())
    cmap = cm.magma
    segs = np.concatenate([path[:-1,None,:], path[1:,None,:]], axis=1)
    lc   = LineCollection(segs, cmap=cmap, norm=norm, linewidth=1.5)
    lc.set_array(speeds)
    return lc, cmap, norm

# ---------- static & snapshot unchanged (omitted for brevity) ----------

# ======================================================================
#  MULTI‑ROCKET ANIMATION (one GIF with many rockets + moving planets)
# ======================================================================
def animate_multi(sim, rockets, fname="multi.gif"):
    # honour caller’s path; if it has no dir, save in OUT_DIR
    if not os.path.dirname(fname):
        fname = os.path.join(OUT_DIR, fname)

    # ---------- data ---------------------------------------------------
    paths  = [np.vstack(rk.path) for rk in rockets]
    speeds = [np.linalg.norm(np.diff(p,axis=0),axis=1)/sim.dt for p in paths]
    vmax   = max(sp.max() for sp in speeds)
    norm   = plt.Normalize(0, vmax); cmap = cm.magma

    # ---------- figure -------------------------------------------------
    fig, ax = plt.subplots(figsize=(6,6))
    fig.patch.set_facecolor('k'); ax.set_facecolor('k')

    # planet orbits (dashed) + moving planet markers
    p_cols = cm.plasma(np.linspace(0.2, 0.9, len(sim.grav.bodies)))
    for j,c in enumerate(p_cols):
        ax.plot(sim.grav.traj[j,:,0], sim.grav.traj[j,:,1],
                '--', color=c, alpha=0.3, lw=0.8)
    ax.scatter(0,0,color='gold',s=90,zorder=6)
    planet_pts = [ax.plot([],[],'o',color=c)[0] for c in p_cols]   # NEW

    # rocket trails + markers
    rk_cols = cm.tab10(np.linspace(0,1,len(rockets)))
    trails, markers = [], []
    for col in rk_cols:
        ln, = ax.plot([],[],'-',lw=1,color=col)
        pt, = ax.plot([],[],'X',color=col,ms=6)
        trails.append(ln); markers.append(pt)

    lim = np.max(np.abs(sim.grav.traj[:,:,:2]))*1.3
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
    ax.set_aspect('equal'); ax.set_xlabel("AU"); ax.set_ylabel("AU")

    frames = max(len(p) for p in paths)

    # ---------- init ---------------------------------------------------
    def init():
        for ln,pt in zip(trails,markers):
            ln.set_data([],[]); pt.set_data([],[])
        for p in planet_pts:
            p.set_data([],[])
        return trails + markers + planet_pts

    # ---------- update -------------------------------------------------
    def update(i):
        # planets
        if i < sim.grav.N_steps:
            for j,p in enumerate(planet_pts):
                px,py = sim.grav.traj[j,i,:2]
                p.set_data([px],[py])

        # rockets
        for p_arr, sp, ln, pt in zip(paths, speeds, trails, markers):
            if i < len(p_arr):
                ln.set_data(p_arr[:i+1,0], p_arr[:i+1,1])
                ln.set_color(cmap(norm(sp[min(i,len(sp)-1)])))
                pt.set_data([p_arr[i,0]], [p_arr[i,1]])     # x‑seq, y‑seq
        return trails + markers + planet_pts

    ani = FuncAnimation(fig, update, frames=frames,
                        init_func=init, interval=25, blit=True)
    ani.save(fname, writer='pillow', fps=30)
    plt.close(fig)