#!/usr/bin/env python3
# bee/bco.py

"""
Artificial-Bee-Colony optimisation for inter-planet transfers.

• Fully configurable via argparse.
• Uses solar_system.build_sim + rocket.RocketSim.
• Food = full destination orbit + random bonus spots.
• Fitness = (angular proximity to dest orbit) × (1/(1+path_length)) + optional bonus.
• Roles: employed → onlooker → scout flow each iteration.
• Outputs: bco_output/<run_id>/{fitness.csv,colony.gif,best_path.png}
"""

import os
import argparse
import random
import math
import numpy as np
from rocket.sim import RocketSim
from bee.viz   import save_fitness, animate_colony, plot_best

# ----------------------------------------------------------------------------
OUT_DIR = "bco_output"
os.makedirs(OUT_DIR, exist_ok=True)

# what fraction of the *remaining* bees (after scouts) become employed
EMPLOYED_FRAC = 0.5

# ----------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--start",       default="Earth",
                   help="start body or 'x,y'")
    p.add_argument("--dest",        default="Mars",
                   help="destination body")
    p.add_argument("--years",  type=float, default=5.0,
                   help="span of simulation in years")
    p.add_argument("--dt",      type=float, default=1.0,
                   help="days per step")
    p.add_argument("--move_planets", action="store_true",
                   help="advance planets each tick")
    p.add_argument("--show_orbits", action="store_true",
                   help="animate planet markers along orbits")
    p.add_argument("--use_gravity",  action="store_true",
                   help="apply gravity to bees")
    p.add_argument("--n_bees",      type=int,   default=50,
                   help="total number of bees")
    p.add_argument("--scout_frac",  type=float, default=0.25,
                   help="fraction of bees that are scouts")
    p.add_argument("--speed",       type=float, default=30000,
                   help="max bee speed (km/h)")
    p.add_argument("--fitness", choices=["sigmoid","log","exp","linear"],
                   default="linear",
                   help="unused in composite, kept for legacy")
    p.add_argument("--k",       type=float, default=5.0,
                   help="unused steepness param")
    p.add_argument("--extra_food", type=int, default=5,
                   help="number of random bonus sites on dest orbit")
    return p.parse_args()

# ----------------------------------------------------------------------------
def start_pos(sim, start):
    """Return (r0, v0) in AU, AU/day from body name or literal coords."""
    if isinstance(start, str):
        idx, _ = sim.grav.get_body(start)
        r0 = sim.grav.traj[idx,0].copy()
        v0 = (sim.grav.traj[idx,1] - sim.grav.traj[idx,0]) / sim.dt
    else:
        coords = np.fromstring(start, sep=",")
        r0 = coords.copy()
        v0 = np.zeros(2)
    return r0, v0

# ----------------------------------------------------------------------------
class Bee:
    def __init__(self, pos, angle, speed_AUd):
        self.r           = pos.copy()
        # initial velocity vector
        self.v           = speed_AUd * np.array([math.cos(angle),
                                                  math.sin(angle)])
        self.fit         = 0.0
        self.trial       = 0       # for abandonment
        self.path        = [self.r.copy()]
        self.path_length = 0.0     # AU traveled
        self.role        = None

# ----------------------------------------------------------------------------
class BeeColony:
    def __init__(self, args):
        self.args = args

        # partition bees into scouts / rest
        self.N         = args.n_bees
        self.ns        = int(args.scout_frac * self.N)
        self.nr        = self.N - self.ns
        self.ne        = int(EMPLOYED_FRAC * self.nr)
        self.no        = self.nr - self.ne

        # convert km/h → AU/day
        self.speed_AUd = args.speed * 1e3 * 24.0 / 1.495978707e11

        # build solar system sim
        self.sim = RocketSim(years=args.years, dt=args.dt, elliptical=True)
        if not args.use_gravity:
            self.sim.grav.gravity_accel = lambda r: np.zeros(2)

        # destination orbit
        di, _          = self.sim.grav.get_body(args.dest)
        self.orbit     = self.sim.grav.traj[di,:,:2]       # (M,2)
        M               = self.orbit.shape[0]
        # precompute angular‐distance weights
        angles         = np.linspace(0,2*math.pi,M,endpoint=False)
        theta_dest     = angles[di]
        delta          = np.abs((angles - theta_dest + math.pi)
                                % (2*math.pi) - math.pi)
        self.orbit_weights = 1.0 - delta/math.pi           # in [0,1]

        # bonus sites
        if args.extra_food>0:
            idxs = np.random.choice(M, args.extra_food, replace=False)
            self.bonus_pts = self.orbit[idxs]
        else:
            self.bonus_pts = np.empty((0,2))

        # initialize bees
        r0, _ = start_pos(self.sim, args.start)
        self.scouts   = []
        self.employed = []
        self.onlookers= []
        for i in range(self.N):
            ang = random.uniform(0,2*math.pi)
            b   = Bee(r0, ang, self.speed_AUd)
            if   i < self.ne:      b.role="employed"; self.employed.append(b)
            elif i < self.ne+self.no: b.role="onlooker";self.onlookers.append(b)
            else:                   b.role="scout";   self.scouts.append(b)

        self.bees       = self.employed + self.onlookers + self.scouts
        self.history    = []    # best fitness each step
        self.best_path  = None

    # ------------------------------------------------------------------------
    def step_bee(self, bee):
        prev = bee.r.copy()
        bee.r += bee.v * self.sim.dt
        if self.args.use_gravity:
            bee.v += self.sim.grav.gravity_accel(bee.r) * self.sim.dt

        bee.path_length += np.linalg.norm(bee.r - prev)

        # nearest orbit sample
        dists = np.linalg.norm(self.orbit - bee.r, axis=1)
        i_near = np.argmin(dists)
        w_orbit = self.orbit_weights[i_near]
        w_path  = 1.0/(1.0 + bee.path_length)

        bee.fit = w_orbit * w_path

        # bonus
        if self.bonus_pts.size:
            d2 = np.linalg.norm(self.bonus_pts - bee.r, axis=1).min()
            if d2 < 0.02:
                bee.fit += 0.5

        bee.path.append(bee.r.copy())

    # ------------------------------------------------------------------------
    def run(self):
        steps     = int(self.args.years * 365.0 / self.args.dt)
        best_path = []

        for _ in range(steps):
            # 1) Employed phase: local neighborhood search
            for b in self.employed:
                # small random perturbation of direction
                ang = math.atan2(b.v[1], b.v[0]) \
                    + random.uniform(-0.1,0.1)
                b.v = self.speed_AUd * np.array([math.cos(ang),
                                                 math.sin(ang)])
                self.step_bee(b)

                # abandonment tracking
                if random.random() < b.fit:
                    b.trial = 0
                else:
                    b.trial += 1

            # 2) Onlooker phase: choose employed bee to follow
            fits = np.array([b.fit for b in self.employed])
            total = fits.sum() or 1.0
            probs = fits/total

            for b in self.onlookers:
                # pick one employed solution
                j = random.choices(self.employed, weights=probs)[0]
                # copy its velocity exactly
                b.v = j.v.copy()
                self.step_bee(b)

                # abandonment
                if random.random() < b.fit:
                    b.trial = 0
                else:
                    b.trial += 1

            # 3) Scout phase: re-initialize bad solutions
            for b in self.scouts:
                if b.trial > 15:
                    ang = random.uniform(0,2*math.pi)
                    b.v   = self.speed_AUd * np.array([math.cos(ang),
                                                       math.sin(ang)])
                    b.trial = 0
                self.step_bee(b)

            # record best
            best = max(self.bees, key=lambda b: b.fit)
            best_path.append(best.r.copy())
            self.history.append(best.fit)

            # optional planet step
            if self.args.move_planets:
                self.sim.grav.step()

        self.best_path = np.vstack(best_path)
        return self.best_path

# ----------------------------------------------------------------------------
def main():
    args      = get_args()
    colony    = BeeColony(args)
    best_traj = colony.run()

    run_id = f"{args.start}_{args.dest}_ABC_{args.n_bees}b"
    out_sub= os.path.join(OUT_DIR, run_id)
    os.makedirs(out_sub, exist_ok=True)

    save_fitness(colony, os.path.join(out_sub,"fitness.csv"))
    animate_colony(colony,
                   fname="colony.gif",
                   out_dir=out_sub,
                   show_orbits=args.show_orbits)
    plot_best(colony,
              fname="best_path.png",
              out_dir=out_sub)

    print("BCO (ABC) outputs in", out_sub)

if __name__ == "__main__":
    main()
