#!/usr/bin/env python3
# bee/bco.py

"""
Artificial-Bee-Colony optimisation for inter-planet transfers.

• Fully configurable via argparse.
• Uses solar_system.build_sim + rocket.RocketSim.
• Food = full destination orbit + random bonus spots.
• Fitness = distance-to-orbit (linear by default).
• Roles: scouts vs workers; abandonment & re-scouting.
• Outputs: bco_output/<run_id>/fitness.csv + colony.gif + best_path.png
"""

import os, argparse, random, math
import numpy as np
from rocket.sim import RocketSim
from bee.viz import save_fitness, animate_colony, plot_best

# ----------------------------------------------------------------------------
# Base output folder
# ----------------------------------------------------------------------------
OUT_DIR = "bco_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# Command-line arguments
# ----------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--start",       default="Earth", help="start body or 'x,y'")
    p.add_argument("--dest",        default="Mars",  help="destination body")
    p.add_argument("--years",  type=float, default=5.0, help="span in years")
    p.add_argument("--dt",      type=float, default=1.0, help="day per step")
    p.add_argument("--move_planets", action="store_true",
                   help="advance planets each tick")
    p.add_argument("--show_orbits", action="store_true",
                   help="animate planet markers along orbits")
    p.add_argument("--use_gravity",  action="store_true",
                   help="apply gravity to bees")
    p.add_argument("--n_bees",      type=int,   default=50,
                   help="total number of bees")
    p.add_argument("--scout_frac",  type=float, default=0.25,
                   help="fraction scouts vs workers")
    p.add_argument("--speed",       type=float, default=100000,
                   help="max bee speed (km/h)")
    p.add_argument("--fitness", choices=["sigmoid","log","exp","linear"],
                   default="linear",  # linear by default
                   help="fitness curve type")
    p.add_argument("--k",       type=float, default=5.0,
                   help="steepness for sigmoid/exp")
    p.add_argument("--extra_food", type=int, default=5,
                   help="number of random bonus sites")
    return p.parse_args()

# ----------------------------------------------------------------------------
# Helpers: start_pos + fitness
# ----------------------------------------------------------------------------
def start_pos(sim, start):
    """Get (r0, v0) in AU, AU/day from body name or raw coords."""
    if isinstance(start, str):
        idx, _ = sim.grav.get_body(start)
        r0 = sim.grav.traj[idx, 0].copy()
        v0 = (sim.grav.traj[idx, 1] - sim.grav.traj[idx, 0]) / sim.dt
    else:
        coords = np.fromstring(start, sep=',')
        r0 = coords.copy()
        v0 = np.zeros(2)
    return r0, v0

def closest_orbit_dist(r, orbit_pts):
    return np.linalg.norm(orbit_pts - r, axis=1).min()

def fitness_fn(dist, orbit_len, kind="linear", k=5.0):
    x = dist / orbit_len
    if kind == "sigmoid":
        return 1.0/(1.0 + math.exp(k*(x-0.5)))
    if kind == "log":
        return 1.0 - math.log1p(k*x)/math.log1p(k)
    if kind == "exp":
        return math.exp(-k*x)
    # linear
    return 1.0 - x

# ----------------------------------------------------------------------------
class Bee:
    def __init__(self, pos, angle, speed_AUd):
        self.r     = pos.copy()
        self.v     = speed_AUd * np.array([math.cos(angle), math.sin(angle)])
        self.fit   = 0.0
        self.trial = 0
        self.path  = [self.r.copy()]
        self.role  = None

class BeeColony:
    def __init__(self, args):
        self.args      = args
        self.N         = args.n_bees
        self.scouts    = int(args.scout_frac * self.N)
        self.speed_AUd = args.speed * 1000*24.0 / 1.495978707e11

        # planet sim
        self.sim = RocketSim(years=args.years, dt=args.dt, elliptical=True)
        if not args.use_gravity:
            self.sim.grav.gravity_accel = lambda r: np.zeros(2)

        # destination orbit
        idx, _        = self.sim.grav.get_body(args.dest)
        self.orbit    = self.sim.grav.traj[idx,:,:2]
        self.orbit_len= np.max(np.linalg.norm(self.orbit, axis=1))

        # create bees at start
        start_r, _ = start_pos(self.sim, args.start)
        self.bees   = []
        for i in range(self.N):
            ang = random.uniform(0,2*math.pi)
            b   = Bee(start_r, ang, self.speed_AUd)
            b.role = "scout" if i < self.scouts else "worker"
            self.bees.append(b)

        self.history   = []   # best fitness each step
        self.best_path = None

        # random bonus sites
        if args.extra_food > 0:
            idx = np.random.choice(len(self.orbit), args.extra_food, replace=False)
            self.bonus_pts = self.orbit[idx]
        else:
            self.bonus_pts = np.empty((0,2))

    def step_bee(self, bee):
        bee.r += bee.v * self.sim.dt
        if self.args.use_gravity:
            bee.v += self.sim.grav.gravity_accel(bee.r) * self.sim.dt

        d = closest_orbit_dist(bee.r, self.orbit)
        bee.fit = fitness_fn(d, self.orbit_len, self.args.fitness, self.args.k)
        if self.bonus_pts.size:
            d2 = np.linalg.norm(self.bonus_pts - bee.r, axis=1).min()
            if d2 < 0.02:
                bee.fit += 0.5
        bee.path.append(bee.r.copy())

    def run(self):
        steps     = int(self.args.years * 365.0 / self.args.dt)
        best_path = []
        for _ in range(steps):
            for bee in self.bees:
                self.step_bee(bee)
                if random.random() < bee.fit:
                    bee.trial = 0
                else:
                    bee.trial += 1
                    if bee.trial > 15:
                        ang = random.uniform(0,2*math.pi)
                        bee.v = self.speed_AUd*np.array([math.cos(ang),math.sin(ang)])
                        bee.trial = 0

            # scouts re-randomize
            for _ in range(self.scouts):
                b   = random.choice(self.bees)
                ang = random.uniform(0,2*math.pi)
                b.v = self.speed_AUd * np.array([math.cos(ang), math.sin(ang)])

            best = max(self.bees, key=lambda b: b.fit)
            best_path.append(best.r.copy())
            self.history.append(best.fit)

            if self.args.move_planets:
                self.sim.grav.step()

        self.best_path = np.vstack(best_path)
        return self.best_path

# ----------------------------------------------------------------------------
def main():
    args      = get_args()
    colony    = BeeColony(args)
    best_traj = colony.run()

    run_id = f"{args.start}_{args.dest}_{args.fitness}_{args.n_bees}bees"
    out_sub= os.path.join(OUT_DIR, run_id)
    os.makedirs(out_sub, exist_ok=True)

    save_fitness(colony, os.path.join(out_sub, "fitness.csv"))
    animate_colony(colony,
                   fname="colony.gif",
                   out_dir=out_sub,
                   show_orbits=args.show_orbits)
    plot_best(colony,
              fname="best_path.png",
              out_dir=out_sub)

    print("BCO outputs in", out_sub)

if __name__ == "__main__":
    main()
