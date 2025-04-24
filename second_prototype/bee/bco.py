#!/usr/bin/env python3
# bee/bco.py

"""
Artificial‐Bee‐Colony optimisation for inter‐planet transfers, with multi‐epoch support.

• Fully configurable via argparse.
• Runs `--epochs` independent ABC searches over the same transfer span.
• Records each epoch’s best fitness in epoch_fitness.csv (only if epochs>1).
• Selects the single best‐epoch’s winning bee and uses *its* path up to planet arrival.
• Stops each bee immediately upon arrival (no wandering past the destination).
• Caps total steps via --max_steps (default 999).
• Outputs to bco_output/<run_id>/:
      • epoch_fitness.csv   (if --epochs > 1)
      • summary.csv
      • fitness.csv         (time‐series of best fitness)
      • colony_best.gif     (animation of winning colony)
      • best_path.png       (static plot of the winning bee’s path)
"""

import os
import argparse
import random
import math
import csv

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
    p.add_argument("--max_steps", type=int, default=999,
                   help="maximum number of ABC iterations")
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
        self.finished    = False   # once True, this bee stops moving


class BeeColony:
    def __init__(self, args):
        self.args = args
        # number of scouts / remainder, then employed / onlooker split
        self.N   = args.n_bees
        self.ns  = int(args.scout_frac * self.N)
        self.nr  = self.N - self.ns
        self.ne  = int(EMPLOYED_FRAC * self.nr)
        self.no  = self.nr - self.ne

        # convert km/h → AU/day
        self.speed_AUd = args.speed * 1e3 * 24.0 / 1.495978707e11

        # planetary simulator
        self.sim = RocketSim(years=args.years, dt=args.dt, elliptical=True)
        if not args.use_gravity:
            self.sim.grav.gravity_accel = lambda r: np.zeros(2)

        # start & destination
        r0,_           = start_pos(self.sim, args.start)
        di,_           = self.sim.grav.get_body(args.dest)
        dest0          = self.sim.grav.traj[di,0,:2]
        self.dest_pos  = dest0
        self.stop_dist = 0.02    # arrival threshold (AU)

        # straight‐line distance & fitness threshold
        self.D      = np.linalg.norm(dest0 - r0)
        self.thresh = 1.05 * self.D

        # full dest orbit + angular‐distance weights
        self.orbit   = self.sim.grav.traj[di,:,:2]
        M           = len(self.orbit)
        angles      = np.linspace(0,2*math.pi,M,endpoint=False)
        θ_dest      = angles[di]
        Δ           = np.abs((angles - θ_dest + math.pi)
                             % (2*math.pi) - math.pi)
        self.orbit_weights = 1.0 - Δ/math.pi

        # fixed “food patches” on the orbit
        idxs = {di}
        if args.extra_food>0:
            idxs |= set(random.sample(range(M), args.extra_food))
        self.bonus_idxs = np.array(sorted(idxs), dtype=int)
        self.bonus_pts  = self.orbit[self.bonus_idxs]
        self.bonus_vals = self.orbit_weights[self.bonus_idxs]

        # create the bees in each role
        r0,_         = start_pos(self.sim, args.start)
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

        self.history   = []   # best fitness per step
        self.best_path = None


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

        # angular‐proximity fitness
        dists   = np.linalg.norm(self.orbit - b.r, axis=1)
        i_near  = np.argmin(dists)
        w_orbit = self.orbit_weights[i_near]

        # path‐length penalty/bonus
        L        = b.path_length
        diff     = self.thresh - L
        frac     = diff/self.thresh
        path_b   = (1.0 + frac) if diff>=0 else frac

        b.fit = w_orbit * path_b

        # patch bonus
        d2  = np.linalg.norm(self.bonus_pts - b.r, axis=1)
        hit = np.where(d2< self.stop_dist)[0]
        if hit.size:
            b.fit += float(self.bonus_vals[hit[0]])

        b.path.append(b.r.copy())


    def run(self):
        # cap total iterations
        max_it = self.args.max_steps
        nominal_steps = int(self.args.years*365.0/self.args.dt)
        steps = min(nominal_steps, max_it)

        for _ in range(steps):
            # employed: small steer toward primary patch
            for b in self.employed:
                θ0      = math.atan2(b.v[1], b.v[0])
                dest_xy = self.bonus_pts[0]
                θ1      = math.atan2(dest_xy[1]-b.r[1],
                                     dest_xy[0]-b.r[0])
                θ       = 0.8*θ0 + 0.2*θ1 + random.uniform(-0.02,0.02)
                b.v = self.speed_AUd * np.array([math.cos(θ),
                                                 math.sin(θ)])
                self.step_bee(b)
                b.trial = 0 if random.random()<b.fit else b.trial+1

            # onlookers: probabilistically copy employed velocities
            fits = np.array([b.fit for b in self.employed])
            pr   = fits/(fits.sum() or 1.0)
            for b in self.onlookers:
                j = random.choices(self.employed, weights=pr, k=1)[0]
                b.v = j.v.copy()
                self.step_bee(b)
                b.trial = 0 if random.random()<b.fit else b.trial+1

            # scouts: random reseed if abandoned
            for b in self.scouts:
                if b.trial>15:
                    θ = random.uniform(0,2*math.pi)
                    b.v = self.speed_AUd*np.array([math.cos(θ),
                                                   math.sin(θ)])
                    b.trial = 0
                self.step_bee(b)

            # record this step’s best fitness
            best = max(self.bees, key=lambda b: b.fit)
            self.history.append(best.fit)

            if self.args.move_planets:
                self.sim.grav.step()

        # done—caller will pick the overall winner
        return



def main():
    args = get_args()

    # run epochs
    epoch_summary    = []
    overall_best_f   = -1e9
    overall_best_bee = None
    for ep in range(1, args.epochs+1):
        col = BeeColony(args)
        col.run()
        winner = max(col.bees, key=lambda b: b.fit)
        epoch_summary.append((ep, winner.fit))
        if winner.fit > overall_best_f:
            overall_best_f   = winner.fit
            overall_best_bee = (col, winner)

    # prepare output folder
    run_id = f"{args.start}_{args.dest}_ABC_{args.n_bees}b"
    out_sub= os.path.join(OUT_DIR, run_id)
    os.makedirs(out_sub, exist_ok=True)

    # per-epoch CSV (only if >1 epoch)
    if args.epochs > 1:
        with open(os.path.join(out_sub,"epoch_fitness.csv"),"w",newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch","best_fit"])
            w.writerows(epoch_summary)

    # extract the overall winner
    best_col, best_bee = overall_best_bee

    # truncate path upon arrival
    full_path = np.vstack(best_bee.path)
    d2        = np.linalg.norm(full_path - best_col.dest_pos, axis=1)
    hit       = np.where(d2 < best_col.stop_dist)[0]
    if hit.size:
        arrival_step = int(hit[0])
        path        = full_path[:arrival_step+1]
    else:
        arrival_step = len(full_path)-1
        path         = full_path

    # compute total path length
    path_len = float(
        np.sum(np.linalg.norm(np.diff(path,axis=0),axis=1))
    )

    # write the per‐step best‐fitness time series
    save_fitness(best_col, os.path.join(out_sub,"fitness.csv"))

    # animate the winning colony
    animate_colony(best_col,
                   fname="colony_best.gif",
                   out_dir=out_sub,
                   show_orbits=args.show_orbits)

    # write static best‐path figure
    best_col.best_path = path
    plot_best(best_col,
              fname="best_path.png",
              out_dir=out_sub)

    # summary.csv with extra statistics
    with open(os.path.join(out_sub,"summary.csv"),"w",newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "start","dest","n_bees","epochs",
            "best_fit","path_length_AU","arrival_step"
        ])
        w.writerow([
            args.start, args.dest, args.n_bees, args.epochs,
            overall_best_f, path_len, arrival_step
        ])

    print("DONE →", out_sub)


if __name__ == "__main__":
    main()
