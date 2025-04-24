#!/usr/bin/env python3
# bee/bco.py
"""
Artificial‐Bee‐Colony optimisation for inter‐planet transfers, **with global-best
history tracking**.

New in this version
-------------------
* While the simulation is running we continuously watch every bee and remember
  the *single best fitness value ever seen*, **which step it occurred at**, and
  the **path (and its length) from launch up to that step**.
* These values are written to *summary.csv* and are also used for the final
  best-path plot, so the graphic now shows the *true* best sub-path rather than
  the winner’s final trajectory end-to-end.
* `epoch_fitness.csv` gains two extra columns – the step index of the epoch’s
  max fitness and the path length at that step.

The command-line interface and all previous outputs remain unchanged; new
columns are simply appended to the existing CSVs so no downstream tooling
breaks.
"""

import os
import argparse
import random
import math
import csv
from copy import deepcopy

import numpy as np

from rocket.sim import RocketSim
from bee.viz   import save_fitness, animate_colony, plot_best

OUT_DIR = "bco_output"
os.makedirs(OUT_DIR, exist_ok=True)

EMPLOYED_FRAC = 0.5


def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--start",      default="Earth", help="start body or 'x,y'")
    p.add_argument("--dest",       default="Mars",  help="destination body")
    p.add_argument("--years",  type=float, default=5.0, help="transfer span (yrs)")
    p.add_argument("--dt",    type=float, default=1.0, help="days per step")
    p.add_argument("--move_planets", action="store_true",
                   help="advance planet markers each tick")
    p.add_argument("--show_orbits", action="store_true",
                   help="animate planet markers moving")
    p.add_argument("--use_gravity",  action="store_true",
                   help="apply gravity to bees")
    p.add_argument("--n_bees",     type=int,   default=50,
                   help="total bees in colony")
    p.add_argument("--scout_frac", type=float, default=0.25,
                   help="fraction scouts vs employed/onlooker")
    p.add_argument("--speed",      type=float, default=30000,
                   help="max bee speed (km/h)")
    p.add_argument("--extra_food", type=int,   default=5,
                   help="extra random patches on dest orbit")
    p.add_argument("--epochs",     type=int,   default=1,
                   help="number of independent ABC runs")
    return p.parse_args()


def start_pos(sim, start):
    if isinstance(start, str):
        idx,_ = sim.grav.get_body(start)
        r0    = sim.grav.traj[idx,0].copy()
        v0    = (sim.grav.traj[idx,1] - sim.grav.traj[idx,0]) / sim.dt
    else:
        coords = np.fromstring(start, sep=",")
        r0     = coords.copy()
        v0     = np.zeros(2)
    return r0, v0


class Bee:
    def __init__(self, pos, angle, speed_AUd):
        self.r           = pos.copy()
        self.v           = speed_AUd * np.array([math.cos(angle),
                                                  math.sin(angle)])
        self.fit         = 0.0
        self.trial       = 0
        self.path        = [self.r.copy()]
        self.path_length = 0.0
        self.role        = None
        # once True, this bee stops moving/appending to its path
        self.finished    = False


class BeeColony:
    def __init__(self, args):
        self.args = args

        # numbers of scout/employed/onlooker
        self.N   = args.n_bees
        self.ns  = int(args.scout_frac * self.N)
        self.nr  = self.N - self.ns
        self.ne  = int(EMPLOYED_FRAC * self.nr)
        self.no  = self.nr - self.ne

        # convert km/h → AU/day
        self.speed_AUd = args.speed * 1e3 * 24.0 / 1.495978707e11

        # build a fresh planetary sim
        self.sim       = RocketSim(years=args.years,
                                   dt=args.dt,
                                   elliptical=True)
        if not args.use_gravity:
            self.sim.grav.gravity_accel = lambda r: np.zeros(2)

        # starting & destination positions
        r0,_           = start_pos(self.sim, args.start)
        di,_           = self.sim.grav.get_body(args.dest)
        dest0          = self.sim.grav.traj[di,0,:2]
        self.dest_pos  = dest0
        # when within this AU, we consider "arrived"
        self.stop_dist = 0.02

        # straight‐line distance & a loose threshold (for fitness bonus)
        self.D         = np.linalg.norm(dest0 - r0)
        self.thresh    = 1.05 * self.D

        # sample the full destination orbit & compute angular weights
        self.orbit       = self.sim.grav.traj[di,:,:2]
        M               = self.orbit.shape[0]
        angles          = np.linspace(0,2*math.pi,M,endpoint=False)
        θ_dest          = angles[di]
        Δ               = np.abs((angles - θ_dest + math.pi)
                                 % (2*math.pi) - math.pi)
        self.orbit_weights = 1.0 - Δ/math.pi

        # pick the fixed‐food‐patch indices: always include true planet idx
        idxs = {di}
        if args.extra_food > 0:
            idxs |= set(random.sample(range(M), args.extra_food))
        self.bonus_idxs = np.array(sorted(idxs), dtype=int)
        self.bonus_pts  = self.orbit[self.bonus_idxs]
        self.bonus_vals = self.orbit_weights[self.bonus_idxs]

        # initialize bees in the three roles
        self.scouts    = []
        self.employed  = []
        self.onlookers = []
        for i in range(self.N):
            ang = random.uniform(0,2*math.pi)
            b   = Bee(r0, ang, self.speed_AUd)
            if   i < self.ne:
                b.role="employed";  self.employed.append(b)
            elif i < self.ne+self.no:
                b.role="onlooker"; self.onlookers.append(b)
            else:
                b.role="scout";     self.scouts.append(b)
        self.bees    = self.employed + self.onlookers + self.scouts

        # history & global-best trackers
        self.history          = []  # time-series of colony best fitness
        self.global_best_fit  = -1e9
        self.global_best_step = -1
        self.global_best_path = None  # numpy array (N×2)
        self.global_best_len  = 0.0

    # ---------------------------------------------------------------------
    # single-bee integrator and fitness update
    # ---------------------------------------------------------------------
    def step_bee(self, b):
        if b.finished:
            return

        prev = b.r.copy()
        b.r  += b.v * self.sim.dt
        if self.args.use_gravity:
            b.v += self.sim.grav.gravity_accel(b.r) * self.sim.dt

        # arrival check
        if np.linalg.norm(b.r - self.dest_pos) < self.stop_dist:
            b.finished = True
            b.path.append(b.r.copy())
            return

        # update path length
        b.path_length += np.linalg.norm(b.r - prev)

        # fitness components ------------------------------------------------
        dists   = np.linalg.norm(self.orbit - b.r, axis=1)
        i_near  = np.argmin(dists)
        w_orbit = self.orbit_weights[i_near]

        L          = b.path_length
        diff       = self.thresh - L
        frac       = diff / self.thresh
        path_bonus = (1.0 + frac) if diff >= 0 else frac
        b.fit      = w_orbit * path_bonus

        # patch bonus
        d2  = np.linalg.norm(self.bonus_pts - b.r, axis=1)
        hit = np.where(d2 < 0.02)[0]
        if hit.size:
            b.fit += float(self.bonus_vals[hit[0]])

        # record new position
        b.path.append(b.r.copy())

    # ------------------------------------------------------------------
    # full colony evolution for the configured years
    # ------------------------------------------------------------------
    def run(self):
        steps = int(self.args.years * 365.0 / self.args.dt)
        for step in range(steps):
            # employed bees ------------------------------------------------
            for b in self.employed:
                θ0      = math.atan2(b.v[1], b.v[0])
                dest_xy = self.bonus_pts[0]   # true planet
                θ1      = math.atan2(dest_xy[1]-b.r[1],
                                     dest_xy[0]-b.r[0])
                θ       = 0.8*θ0 + 0.2*θ1 + random.uniform(-0.02,0.02)
                b.v     = self.speed_AUd * np.array([math.cos(θ),
                                                     math.sin(θ)])
                self.step_bee(b)
                b.trial = 0 if random.random() < b.fit else b.trial + 1

            # onlookers ----------------------------------------------------
            fits = np.array([b.fit for b in self.employed])
            pr   = fits / (fits.sum() or 1.0)
            for b in self.onlookers:
                j   = random.choices(self.employed, weights=pr, k=1)[0]
                b.v = j.v.copy()
                self.step_bee(b)
                b.trial = 0 if random.random() < b.fit else b.trial + 1

            # scouts -------------------------------------------------------
            for b in self.scouts:
                if b.trial > 15:
                    θ   = random.uniform(0, 2*math.pi)
                    b.v = self.speed_AUd * np.array([math.cos(θ),
                                                     math.sin(θ)])
                    b.trial = 0
                self.step_bee(b)

            # book-keeping --------------------------------------------------
            colony_best = max(self.bees, key=lambda b: b.fit)
            self.history.append(colony_best.fit)

            # global-best update
            if colony_best.fit > self.global_best_fit:
                self.global_best_fit  = colony_best.fit
                self.global_best_step = step
                self.global_best_len  = colony_best.path_length
                # deep copy to freeze the path as it exists *now*
                self.global_best_path = np.vstack(colony_best.path).copy()

            if self.args.move_planets:
                self.sim.grav.step()

        # nothing to return – caller just reads the attributes
        return


# =============================================================================
# top-level driver – handles multiple epochs and all I/O
# =============================================================================

def main():
    args = get_args()

    epoch_rows = []
    overall_best_fit  = -1e9
    overall_best_col  = None

    for ep in range(1, args.epochs + 1):
        colony = BeeColony(args)
        colony.run()

        # epoch-level summary row
        epoch_rows.append((ep,
                           colony.global_best_fit,
                           colony.global_best_step,
                           colony.global_best_len))

        # cross-epoch comparison
        if colony.global_best_fit > overall_best_fit:
            overall_best_fit = colony.global_best_fit
            overall_best_col = colony

    # ---------------------------------------------------------------------
    # output directory bookkeeping
    # ---------------------------------------------------------------------
    run_id  = f"{args.start}_{args.dest}_ABC_{args.n_bees}b"
    out_dir = os.path.join(OUT_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)

    # epoch_fitness.csv ----------------------------------------------------
    with open(os.path.join(out_dir, "epoch_fitness.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "best_fit", "best_step", "path_len"])
        w.writerows(epoch_rows)

    # summary.csv ----------------------------------------------------------
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start", "dest", "n_bees", "epochs",
                    "best_fit", "best_step", "path_len"])
        w.writerow([args.start, args.dest, args.n_bees,
                    args.epochs,
                    overall_best_fit,
                    overall_best_col.global_best_step,
                    overall_best_col.global_best_len])

    # time-series of colony best fitness (from the winning epoch)
    save_fitness(overall_best_col, os.path.join(out_dir, "fitness.csv"))

    # animation ------------------------------------------------------------
    animate_colony(overall_best_col,
                   fname="colony_best.gif",
                   out_dir=out_dir,
                   show_orbits=args.show_orbits)

    # static best-path plot -----------------------------------------------
    overall_best_col.best_path = overall_best_col.global_best_path
    plot_best(overall_best_col,
              fname="best_path.png",
              out_dir=out_dir)

    print("DONE →", out_dir)


if __name__ == "__main__":
    main()
