import numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib import cm
plt.style.use('dark_background')

def _v_colors(path, speeds):
    norm = plt.Normalize(speeds.min(), speeds.max())
    cmap = cm.magma
    segs = np.concatenate([path[:-1,None,:], path[1:,None,:]], axis=1)
    lc   = LineCollection(segs, cmap=cmap, norm=norm, linewidth=1.5)
    lc.set_array(speeds)
    return lc, cmap, norm

def static(sim, rocket, fname="rocket/start.png"):
    pos0 = sim.grav.traj[:,0]
    fig,ax = plt.subplots(figsize=(5,5))
    for j in range(len(sim.grav.bodies)):
        ax.plot(sim.grav.traj[j,:,0], sim.grav.traj[j,:,1], '--', alpha=0.25)
        ax.scatter(pos0[j,0], pos0[j,1], s=15)
    ax.scatter(*rocket.path[0], color='magenta', marker='X', s=50)
    ax.scatter(0,0,color='y',s=70)
    ax.set_aspect('equal'); ax.set_xlabel("AU"); ax.set_ylabel("AU")
    fig.tight_layout(); fig.savefig(fname,dpi=160); plt.close(fig)

def snapshot(sim, rocket, fname="rocket/snapshot.png"):
    path = np.vstack(rocket.path)
    speeds = np.linalg.norm(np.diff(path,axis=0), axis=1) / sim.dt   # AU/day
    lc,cmap,norm = _v_colors(path, speeds)
    fig,ax = plt.subplots(figsize=(5,5))
    ax.add_collection(lc)
    ax.plot(path[-1,0], path[-1,1], 'X', color='magenta', ms=6)
    for j in range(len(sim.grav.bodies)):
        ax.plot(sim.grav.traj[j,:,0], sim.grav.traj[j,:,1], '--', alpha=0.15)
    ax.scatter(0,0,color='gold',s=90,zorder=5)
    lim=np.max(np.abs(sim.grav.traj[:,:,:2]))*1.2
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_aspect('equal')
    ax.set_xlabel("AU"); ax.set_ylabel("AU")
    fig.colorbar(cm.ScalarMappable(norm,cmap), ax=ax,label="speed (AU/day)")
    fig.tight_layout(); fig.savefig(fname,dpi=160); plt.close(fig)

def animate(sim, rocket, fname="rocket/flight.gif"):
    import numpy as np, matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import LineCollection
    from matplotlib import cm

    # --- data ------------------------------------------------------------
    path   = np.vstack(rocket.path)
    speeds = np.linalg.norm(np.diff(path, axis=0), axis=1) / sim.dt
    norm   = plt.Normalize(speeds.min(), speeds.max())
    cmap   = cm.magma

    segs   = np.concatenate([path[:-1, None, :], path[1:, None, :]], axis=1)
    lc     = LineCollection(segs, cmap=cmap, norm=norm, linewidth=1.5)
    lc.set_array(speeds)

    Np     = len(sim.grav.bodies)
    colours= plt.cm.plasma(np.linspace(0.2, 0.9, Np))

    # --- figure ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('k');  ax.set_facecolor('k')
    ax.add_collection(lc)

    # planet orbits (dashed) and Sun
    for j, col in enumerate(colours):
        ax.plot(sim.grav.traj[j, :, 0], sim.grav.traj[j, :, 1],
                '--', color=col, alpha=0.25, linewidth=0.8)
    ax.scatter(0, 0, color='gold', s=90, zorder=6)

    # artists for moving markers
    planet_pts = [ax.plot([], [], 'o', color=col)[0] for col in colours]
    rocket_pt, = ax.plot([], [], 'X', color='magenta', ms=6)

    # axis limits
    lim = np.max(np.abs(sim.grav.traj[:, :, :2])) * 1.3
    ax.set_xlim(-lim, lim);  ax.set_ylim(-lim, lim)
    ax.set_aspect('equal');  ax.set_xlabel("AU");  ax.set_ylabel("AU")

    # --- init & update ---------------------------------------------------
    def init():
        lc.set_segments([])
        rocket_pt.set_data([], [])
        for p in planet_pts:  p.set_data([], [])
        return [lc, rocket_pt, *planet_pts]

    def update(i):
        # planets
        for j, p in enumerate(planet_pts):
            p.set_data([sim.grav.traj[j, i, 0]], [sim.grav.traj[j, i, 1]])
        # rocket
        rocket_pt.set_data([path[i, 0]], [path[i, 1]])
        # trail up to i
        if i > 0:
            lc.set_segments(np.concatenate([path[:i, None, :],
                                            path[1:i+1, None, :]], axis=1))
            lc.set_array(speeds[:i])
        return [lc, rocket_pt, *planet_pts]

    ani = FuncAnimation(fig, update, frames=len(path),
                        init_func=init, interval=25, blit=True)
    ani.save(fname, writer='pillow', fps=30)
    plt.close(fig)