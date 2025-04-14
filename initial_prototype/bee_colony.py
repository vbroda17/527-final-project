import argparse, random, datetime, os, pathlib
import numpy as np
from dataclasses import dataclass
from physics import Planet, circle_orbit, gravity
import viz

# ---------------- config ----------------
@dataclass
class Config:
    bees:int=60; iters:int=400; limit:int=30; way:int=6
    dt:float=2000.0            # integration step (s)
    seg_T:float=2e5            # time per segment (s)
    thrust_max:float=5e4       # m/s^2 magnitude
    wp_sigma:float=1e9; ang_sigma:float=0.3
    alpha:float=1.0; beta:float=0.01   # cost weights
    snap:int=10; pct:float=0.25

# -------------- bodies --------------
SUN  = Planet("Sun", 1.989e30, radius_draw=7e8, static_pos=[0,0])
START= Planet("Start",6e24,  static_pos=[ 2.5e11, 0])
OBST = Planet("Obstacle",6e24, static_pos=[ 3.0e11/2, 1.0e11])
TARG = Planet("Target",6e24, static_pos=[ 3.0e11, 2.0e11])
BODIES=[SUN,START,OBST,TARG]

# -------------- helper classes --------------
class Way:
    def __init__(self, point:np.ndarray, theta:float):
        self.p=point; self.th=theta  # radians

def rand_way(cfg):
    x=random.uniform(min(START.static_pos[0],TARG.static_pos[0]),TARG.static_pos[0])
    y=random.uniform(min(START.static_pos[1],TARG.static_pos[1]),TARG.static_pos[1])
    return Way(np.array([x,y]), random.uniform(0,2*np.pi))

def new_path(cfg):
    segs=[rand_way(cfg) for _ in range(cfg.way-1)]
    segs.append(Way(np.array(TARG.static_pos),0))
    return segs

# -------------- physics integration --------------

def simulate(path,cfg):
    pos=np.array(START.static_pos,dtype=float); vel=np.zeros(2)
    fuel=0.0; t_total=0.0
    for seg in path:
        thrust_dir=np.array([np.cos(seg.th),np.sin(seg.th)])
        thrust=thrust_dir*cfg.thrust_max
        steps=int(cfg.seg_T/cfg.dt)
        for _ in range(steps):
            a=thrust+gravity(pos,BODIES,t_total)
            vel+=a*cfg.dt; pos+=vel*cfg.dt; fuel+=np.linalg.norm(thrust)*cfg.dt
            t_total+=cfg.dt
        # corrective burn to hit waypoint exactly (simple snap)
        pos=seg.p.copy()
    time=t_total; return pos,vel,time,fuel

# -------------- cost --------------

def cost(path,cfg):
    _,_,time,fuel=simulate(path,cfg)
    return cfg.alpha*time + cfg.beta*fuel

# -------------- mutation --------------

def mutate(path,cfg):
    new=[]
    for seg in path[:-1]:
        p=seg.p+np.random.normal(0,cfg.wp_sigma,2)
        th=(seg.th+np.random.normal(0,cfg.ang_sigma))%(2*np.pi)
        new.append(Way(p,th))
    new.append(path[-1])
    return new

# -------------- ABC core --------------

def abc(cfg):
    paths=[new_path(cfg) for _ in range(cfg.bees)]
    costs=[cost(p,cfg) for p in paths]; trials=[0]*cfg.bees
    best=paths[int(np.argmin(costs))]; best_c=min(costs)
    hist=[]; snaps=[]
    for it in range(cfg.iters):
        # employed
        for i in range(cfg.bees):
            cand=mutate(paths[i],cfg); c=cost(cand,cfg)
            if c<costs[i]: paths[i],costs[i],trials[i]=cand,c,0
            else: trials[i]+=1
        # on‑lookers
        probs=(1/np.array(costs)); probs/=probs.sum()
        for _ in range(cfg.bees):
            i=np.random.choice(cfg.bees,p=probs); cand=mutate(paths[i],cfg); c=cost(cand,cfg)
            if c<costs[i]: paths[i],costs[i],trials[i]=cand,c,0
            else: trials[i]+=1
        # scouts
        for i in range(cfg.bees):
            if trials[i]>=cfg.limit:
                paths[i]=new_path(cfg); costs[i]=cost(paths[i],cfg); trials[i]=0
        # best
        i=int(np.argmin(costs))
        if costs[i]<best_c: best,best_c=paths[i],costs[i]
        hist.append(best_c)
        if it%cfg.snap==0:
            snaps.append([[w.p for w in p] for p in paths])
    return best,paths,hist,snaps

# -------------- main --------------

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--bees',type=int,default=60); ap.add_argument('--iters',type=int,default=400); ap.add_argument('--show',action='store_true'); ap.add_argument('--pct',type=float,default=0.25)
    args=ap.parse_args(); cfg=Config(bees=args.bees,iters=args.iters,pct=args.pct)

    best,swarm,hist,snaps=abc(cfg)
    print(f"Best cost: {hist[-1]:.3e}")

    # ---- save run ----
    run_dir=pathlib.Path("runs")/f"run_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"; run_dir.mkdir(parents=True,exist_ok=True); train_dir=run_dir/"training"; train_dir.mkdir()

    # convert path lists to np arrays for plotting
    best_pts=np.vstack([w.p for w in best])
    swarm_pts=[np.vstack([w.p for w in p]) for p in swarm]

    viz.best_plot(run_dir,BODIES,best_pts)
    viz.swarm_final(run_dir,BODIES,swarm_pts)
    viz.training_curve(run_dir,hist)
    viz.snapshot_plots(train_dir,BODIES,[[np.vstack(p) for p in snap] for snap in snaps],cfg.pct)

if __name__=='__main__':
    main()