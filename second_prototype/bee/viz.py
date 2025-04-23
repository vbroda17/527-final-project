#!/usr/bin/env python3
# bee/viz.py

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.lines import Line2D

# ------------------------------------------------------------------------
def save_fitness(colony, filename="fitness.csv"):
    import csv
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename,'w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['step','fitness'])
        for i,val in enumerate(colony.history):
            w.writerow([i,val])

# ------------------------------------------------------------------------
def plot_best(colony, fname="best_path.png", out_dir=None):
    path = os.path.join(out_dir,fname) if out_dir else fname
    os.makedirs(os.path.dirname(path), exist_ok=True)

    sim = colony.sim
    best= colony.best_path
    args= colony.args

    fig,ax = plt.subplots(figsize=(6,6))
    fig.patch.set_facecolor('k');  ax.set_facecolor('k')

    # static orbits
    cols = cm.plasma(np.linspace(0.2,0.9,len(sim.grav.bodies)))
    for traj,col in zip(sim.grav.traj[:,:,:2],cols):
        ax.plot(traj[:,0],traj[:,1],'--',color=col,alpha=0.3)

    # best path
    ax.plot(best[:,0],best[:,1],'-',color='magenta',lw=2,label='best path')
    ax.scatter(best[0,0],best[0,1],c='lime',marker='s',label='start')
    ax.scatter(best[-1,0],best[-1,1],c='red',marker='*',label='end')

    # planet markers
    if isinstance(args.start,str):
        si,_=sim.grav.get_body(args.start)
        spt=sim.grav.traj[si,0,:2]
        ax.scatter(spt[0],spt[1],c='lime',marker='s',s=60,label='start planet')
    di,_=sim.grav.get_body(args.dest)
    dpt=sim.grav.traj[di,0,:2]
    ax.scatter(dpt[0],dpt[1],c='cyan',marker='D',s=60,label='dest planet')

    ax.set_aspect('equal')
    ax.set_xlabel('AU'); ax.set_ylabel('AU')
    ax.legend(loc='upper right',fontsize=8)
    fig.savefig(path,dpi=150)
    plt.close(fig)

# ------------------------------------------------------------------------
def animate_colony(colony, fname="colony.gif",
                   out_dir=None, show_orbits=False):
    out_path = os.path.join(out_dir,fname) if out_dir else fname
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    sim   = colony.sim
    orbit = colony.orbit      # (M,2)
    M,steps = orbit.shape[0], len(colony.history)

    # orbit gradient
    ws    = colony.orbit_weights
    segs  = np.stack([orbit[:-1],orbit[1:]],axis=1)
    lc    = LineCollection(segs,
                           cmap=cm.viridis,
                           norm=plt.Normalize(0,1),
                           linewidth=2)
    lc.set_array(ws[:-1])

    fig,ax = plt.subplots(figsize=(6,6))
    fig.patch.set_facecolor('k'); ax.set_facecolor('k')
    ax.add_collection(lc)

    # static orbit traces + moving planet markers
    cols = cm.plasma(np.linspace(0.2,0.9,len(sim.grav.bodies)))
    planet_pts=[]
    for traj,col in zip(sim.grav.traj[:,:,:2],cols):
        ax.plot(traj[:,0],traj[:,1],'--',color=col,alpha=0.3)
        pt,=ax.plot([],[], 'o', color=col, ms=4)
        planet_pts.append((traj,pt))

    # start/dest markers
    args = colony.args
    if isinstance(args.start,str):
        si,_=sim.grav.get_body(args.start)
        spt=sim.grav.traj[si,0,:2]
        ax.scatter(spt[0],spt[1],c='lime',marker='s',s=60,label='start planet')
    di,_=sim.grav.get_body(args.dest)
    dpt=sim.grav.traj[di,0,:2]
    ax.scatter(dpt[0],dpt[1],c='cyan',marker='D',s=60,label='dest planet')

    # food patches (size+color by value)
    if colony.bonus_pts.size:
        vals = colony.bonus_vals
        norm = plt.Normalize(0,1)
        cmap = cm.plasma
        ax.scatter(colony.bonus_pts[:,0],
                   colony.bonus_pts[:,1],
                   s=100*vals + 20,    # base size + scale
                   c=vals, cmap=cmap, norm=norm,
                   edgecolors='white',
                   label='food patch')
        # add colorbar
        sm = cm.ScalarMappable(norm=norm,cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, orientation='vertical',
                     pad=0.02, label='food value')

    # bees: trails & markers by role
    roles   = [b.role for b in colony.bees]
    unique  = list(dict.fromkeys(roles))
    rcols   = cm.tab10(np.linspace(0,1,len(unique)))
    cmap_r  = dict(zip(unique, rcols))
    trails,markers = [],[]
    for b in colony.bees:
        c = cmap_r[b.role]
        ln,=ax.plot([],[], '-', color=c, alpha=0.7)
        pt,=ax.plot([],[], 'o', color=c, ms=6)
        trails.append(ln); markers.append(pt)

    # legend
    handles = [Line2D([],[],marker='o',color=c,linestyle='None',label=r)
               for r,c in cmap_r.items()]
    if colony.bonus_pts.size:
        handles.append(Line2D([],[],marker='o',color='white',
                              linestyle='None',label='food patch'))
    handles += [
        Line2D([],[],marker='s',color='lime',linestyle='None',label='start'),
        Line2D([],[],marker='D',color='cyan',linestyle='None',label='dest')
    ]
    ax.legend(handles=handles, loc='upper right',fontsize=8)

    # axes
    lim = np.max(np.abs(sim.grav.traj[:,:,:2]))*1.2
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
    ax.set_aspect('equal')
    ax.set_xlabel('AU'); ax.set_ylabel('AU')

    # animation
    def init():
        for traj,pt in planet_pts:
            pt.set_data([],[])
        for ln,pt in zip(trails,markers):
            ln.set_data([],[]); pt.set_data([],[])
        return [lc] + [pt for _,pt in planet_pts] + trails + markers

    def update(i):
        # planet markers
        for traj,pt in planet_pts:
            idx = min(i, traj.shape[0]-1) if show_orbits else 0
            xy  = traj[idx]
            pt.set_data([xy[0]],[xy[1]])

        # bee trails
        for b,ln,pt in zip(colony.bees, trails,markers):
            path = np.vstack(b.path)
            if i < path.shape[0]:
                ln.set_data(path[:i+1,0],path[:i+1,1])
                pt.set_data([path[i,0]],[path[i,1]])

        return [lc] + [pt for _,pt in planet_pts] + trails + markers

    ani = FuncAnimation(fig, update, frames=steps,
                        init_func=init, interval=50, blit=True)
    ani.save(out_path, writer='pillow', fps=20)
    plt.close(fig)
