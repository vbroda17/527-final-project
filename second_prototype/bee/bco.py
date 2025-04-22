#!/usr/bin/env python3
# bee/bco.py
"""
Artificial‑Bee‑Colony optimisation for inter‑planet transfers.

• 1 script, fully configurable from the command line (argparse).
• Uses your existing  solar_system.build_sim  (+ optional planet motion)
  and rocket.RocketSim (+ optional gravity acceleration).
• ‘Food’ = every point on the destination orbit, plus optional random
  bonuses.  Fitness is a tunable distance‑to‑target function.
• Roles: scouts, onlookers, employed; simple abandonment & re‑scouting
  rules; trail intensity decays.

GIF + CSV logs go to  bco_output/
"""
import os, argparse, random, math, numpy as np
from rocket.sim import RocketSim
from rocket import viz
from rocket.demo import start_pos           # reuse helper
from solar_system import build_sim

OUT_DIR = "bco_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# command‑line
# ----------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--start", default="Earth", help="start body or x,y")
    p.add_argument("--dest",  default="Mars",  help="destination body")
    p.add_argument("--years", type=float, default=1.0, help="simulation span (yrs)")
    p.add_argument("--dt",    type=float, default=1.0, help="day per step")
    p.add_argument("--move_planets", action="store_true", help="planets advance")
    p.add_argument("--use_gravity",  action="store_true", help="bees feel gravity")
    p.add_argument("--n_bees", type=int, default=50)
    p.add_argument("--scout_frac", type=float, default=0.25, help="fraction scouts")
    p.add_argument("--speed", type=float, default=30000, help="max bee speed km/h")
    p.add_argument("--fitness", choices=["sigmoid","log","exp","linear"],
                   default="sigmoid")
    p.add_argument("--k", type=float, default=5.0, help="steepness for sigmoid/exp")
    p.add_argument("--extra_food", type=int, default=0,
                   help="# random bonus sites on destination orbit")
    return p.parse_args()

# ----------------------------------------------------------------------
# distance‑to‑orbit utility
# ----------------------------------------------------------------------
def closest_orbit_dist(r, orbit_pts):
    d = np.linalg.norm(orbit_pts - r, axis=1)
    return d.min()

def fitness_fn(dist, orbit_len, kind="sigmoid", k=5.0):
    x = dist / orbit_len           # [0…1]   (0 = on planet, 1 = far side)
    if   kind=="sigmoid": return 1/(1+math.exp(k*(x-0.5)))
    elif kind=="log":     return 1-math.log1p(k*x)/math.log1p(k)
    elif kind=="exp":     return math.exp(-k*x)
    else:                 return 1-x             # linear

# ----------------------------------------------------------------------
class Bee:
    def __init__(self, pos, angle, speed_AUd):
        self.r  = pos.copy()
        self.v  = speed_AUd * np.array([math.cos(angle), math.sin(angle)])
        self.fit= 0
        self.trial=0   # abandonment counter

class BeeColony:
    def __init__(self, args, sim_base):
        self.args = args
        self.N    = args.n_bees
        self.scouts = int(args.scout_frac*self.N)
        self.speed_AUd = args.speed * 1000 * 24 / 1.495978707e11
        # clone sim to avoid contaminating base ephemeris
        self.sim  = RocketSim(years=args.years, dt=args.dt, elliptical=True)
        if not args.use_gravity:
            self.sim.grav.gravity_accel = lambda r: np.zeros(2)
        self.planet_idx,_ = self.sim.grav.get_body(args.dest)
        self.orbit = self.sim.grav.traj[self.planet_idx,:,:2]    # (T,2)
        self.orbit_len = np.max(np.linalg.norm(self.orbit,axis=1))
        self.bees = []
        start_r,_ = start_pos(self.sim, args.start)
        for _ in range(self.N):
            ang = random.uniform(0,2*math.pi)
            self.bees.append(Bee(start_r, ang, self.speed_AUd))
        # extra bonus food points
        if args.extra_food:
            idx  = np.random.choice(len(self.orbit), args.extra_food, replace=False)
            self.bonus_pts = self.orbit[idx]
        else:
            self.bonus_pts = np.empty((0,2))

    # ------------------------------------------------------------------
    def step_bee(self, bee):
        bee.r += bee.v * self.sim.dt      # straight line (own engine)
        if self.args.use_gravity:
            bee.v += self.sim.grav.gravity_accel(bee.r) * self.sim.dt

        dist = closest_orbit_dist(bee.r, self.orbit)
        bee.fit = fitness_fn(dist, self.orbit_len,
                             self.args.fitness, self.args.k)
        # bonus sites
        if self.bonus_pts.size:
            bonus_d = np.linalg.norm(self.bonus_pts - bee.r,axis=1).min()
            if bonus_d < 0.02: bee.fit += 0.5

    # ------------------------------------------------------------------
    def run(self):
        steps = int(self.args.years*365/self.args.dt)
        best_path = []
        for step in range(steps):
            # employed/onlooker update
            for bee in self.bees:
                self.step_bee(bee)
                if random.random() < bee.fit:   # share food
                    bee.trial = 0
                else:
                    bee.trial += 1
                    if bee.trial > 15:          # abandonment
                        ang = random.uniform(0,2*math.pi)
                        bee.v = self.speed_AUd*np.array([math.cos(ang),math.sin(ang)])
                        bee.trial = 0
            # scouts randomise
            for _ in range(self.scouts):
                bee=random.choice(self.bees)
                ang = random.uniform(0,2*math.pi)
                bee.v = self.speed_AUd*np.array([math.cos(ang),math.sin(ang)])
            # record best of this step
            best = max(self.bees, key=lambda b:b.fit)
            best_path.append(best.r.copy())
            self.sim.grav.step() if self.args.move_planets else None
        return np.vstack(best_path)

# ----------------------------------------------------------------------
def main():
    args = get_args()
    base_sim = build_sim(years=args.years, dt=args.dt, elliptical=True)

    colony = BeeColony(args, base_sim)
    best_traj = colony.run()

    # ---- visualise best‑bee path ------------------------------------
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(6,6))
    ax.set_facecolor('k'); fig.patch.set_facecolor('k')
    ax.plot(best_traj[:,0], best_traj[:,1],'w-')
    ax.scatter(best_traj[0,0], best_traj[0,1],color='lime',marker='s',label='start')
    ax.scatter(colony.orbit[:,0], colony.orbit[:,1], s=1,color='gray',alpha=0.3)
    ax.scatter(colony.orbit[0,0], colony.orbit[0,1],color='orange',marker='o',label='dest orbit')
    ax.set_aspect('equal'); ax.set_xlabel("AU"); ax.set_ylabel("AU")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(OUT_DIR,"best_path.png"),dpi=150)

    # animated GIF
    rk_col = cm.get_cmap('turbo')(0.8)
    sim_viz = RocketSim(years=args.years, dt=args.dt)
    dummy_rk = sim_viz.add_rocket(best_traj[0], np.zeros(2),
                                  mass=1, max_thrust=0, max_v_kmh=0)
    dummy_rk.path = best_traj.tolist()
    viz.animate(sim_viz, dummy_rk, fname=os.path.join(OUT_DIR,"best_path.gif"))
    print("Results saved to", OUT_DIR)

if __name__ == "__main__":
    main()
