#!/usr/bin/env python3
# bee/bco.py
"""
Artificial-Bee-Colony optimisation for inter-planet transfers
–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
• Tracks the global-best fitness (value, step, path-length, path).
• Gives a strong bonus for flying very close to the destination.
• Allows a looser path-length budget.

Adjustable constants (feel free to tune):
    PATH_THRESH_FACTOR    – path-length tolerance multiplier
    CLOSE_BONUS_RADIUS_AU – distance (AU) considered “very close”
    CLOSE_BONUS_SCALE     – additive reward when inside that radius
"""

# ─────────────────────────────────────────────────────────────
# Tunables for the new behaviour  ←—  tweak these three numbers
# ─────────────────────────────────────────────────────────────
PATH_THRESH_FACTOR    = 1.20   # 1) path-length tolerance (× straight-line)
CLOSE_BONUS_RADIUS_AU = 0.05   # 2) “very close” radius around destination
CLOSE_BONUS_SCALE     = 5.0    # 3) bonus added when inside that radius
# ─────────────────────────────────────────────────────────────

import os
import argparse
import random
import math
import csv

import numpy as np

from rocket.sim import RocketSim
from bee.viz   import save_fitness, animate_colony, plot_best

EMPLOYED_FRAC = 0.5
OUT_DIR = "bco_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# CLI setup
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def start_pos(sim, start):
    """Return initial position/velocity given a body name or x,y string."""
    if isinstance(start, str) and ',' not in start:
        idx, _ = sim.grav.get_body(start)
        r0     = sim.grav.traj[idx, 0].copy()
        v0     = (sim.grav.traj[idx, 1] - sim.grav.traj[idx, 0]) / sim.dt
    else:
        coords = np.fromstring(start, sep=',')
        r0     = coords.copy()
        v0     = np.zeros(2)
    return r0, v0

# ---------------------------------------------------------------------
# Bee container
# ---------------------------------------------------------------------
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
        self.finished    = False   # stop updating once arrived

# ---------------------------------------------------------------------
# Bee Colony (single epoch)
# ---------------------------------------------------------------------
class BeeColony:
    def __init__(self, args):
        self.args = args

        # Split colony into roles
        self.N   = args.n_bees
        self.ns  = int(args.scout_frac * self.N)
        self.nr  = self.N - self.ns
        self.ne  = int(EMPLOYED_FRAC * self.nr)
        self.no  = self.nr - self.ne

        # Velocity scaling (km/h → AU/day)
        self.speed_AUd = args.speed * 1e3 * 24.0 / 1.495978707e11

        # Planetary simulation
        self.sim = RocketSim(years=args.years, dt=args.dt, elliptical=True)
        if not args.use_gravity:
            self.sim.grav.gravity_accel = lambda r: np.zeros(2)

        # Start & destination
        r0, _          = start_pos(self.sim, args.start)
        di, _          = self.sim.grav.get_body(args.dest)
        self.dest_pos  = self.sim.grav.traj[di, 0, :2]
        self.stop_dist = 0.02

        # Path-length tolerance based on straight-line distance
        self.D      = np.linalg.norm(self.dest_pos - r0)
        self.thresh = PATH_THRESH_FACTOR * self.D

        # Destination orbit (for angular weighting)
        self.orbit          = self.sim.grav.traj[di, :, :2]
        M                   = self.orbit.shape[0]
        angles              = np.linspace(0, 2*math.pi, M, endpoint=False)
        θ_dest              = angles[di]
        Δ                   = np.abs((angles - θ_dest + math.pi) % (2*math.pi) - math.pi)
        self.orbit_weights  = 1.0 - Δ / math.pi

        # Food patches (always include real planet)
        idxs = {di}
        if args.extra_food > 0:
            idxs |= set(random.sample(range(M), args.extra_food))
        self.bonus_idxs = np.array(sorted(idxs), dtype=int)
        self.bonus_pts  = self.orbit[self.bonus_idxs]
        self.bonus_vals = self.orbit_weights[self.bonus_idxs]

        # Create bees
        self.scouts, self.employed, self.onlookers = [], [], []
        for i in range(self.N):
            ang = random.uniform(0, 2*math.pi)
            bee = Bee(r0, ang, self.speed_AUd)
            if   i < self.ne:            bee.role = "employed";  self.employed.append(bee)
            elif i < self.ne + self.no:  bee.role = "onlooker"; self.onlookers.append(bee)
            else:                        bee.role = "scout";     self.scouts.append(bee)
        self.bees = self.employed + self.onlookers + self.scouts

        # Global-best trackers
        self.history          = []
        self.global_best_fit  = -1e9
        self.global_best_step = -1
        self.global_best_len  = 0.0
        self.global_best_path = None

    # ----------------------------------------------------------
    # Physics + fitness for a single bee
    # ----------------------------------------------------------
    def step_bee(self, b):
        if b.finished:
            return

        prev = b.r.copy()
        b.r += b.v * self.sim.dt
        if self.args.use_gravity:
            b.v += self.sim.grav.gravity_accel(b.r) * self.sim.dt

        # Check arrival
        if np.linalg.norm(b.r - self.dest_pos) < self.stop_dist:
            b.finished = True
            b.path.append(b.r.copy())
            return

        # Grow path length
        b.path_length += np.linalg.norm(b.r - prev)

        # ---------- FITNESS ----------
        # (1) angular/orbit proximity
        dists   = np.linalg.norm(self.orbit - b.r, axis=1)
        i_near  = int(np.argmin(dists))
        w_orbit = (self.orbit_weights[i_near]) ** 2  # square to emphasise

        # (2) path-length bonus/penalty
        diff       = self.thresh - b.path_length
        frac       = diff / self.thresh
        path_bonus = (1.0 + frac) if diff >= 0 else frac

        b.fit = 2.0 * w_orbit * path_bonus  # ×2 for a bit more influence

        # (3) Food-patch bonus
        d2  = np.linalg.norm(self.bonus_pts - b.r, axis=1)
        hit = np.where(d2 < 0.02)[0]
        if hit.size:
            b.fit += float(self.bonus_vals[hit[0]])

        # (4) CLOSE-PROXIMITY bonus  ←— the new part you’ll likely tweak most
        d_dest = np.linalg.norm(b.r - self.dest_pos)
        if d_dest < CLOSE_BONUS_RADIUS_AU:
            # linear ramp: max bonus at 0, zero at radius limit
            b.fit += CLOSE_BONUS_SCALE * (1.0 - d_dest / CLOSE_BONUS_RADIUS_AU)

        b.path.append(b.r.copy())

    # ----------------------------------------------------------
    # Main colony loop
    # ----------------------------------------------------------
    def run(self):
        steps = int(self.args.years * 365.0 / self.args.dt)
        for step in range(steps):
            # Employed bees
            for b in self.employed:
                θ0      = math.atan2(b.v[1], b.v[0])
                dest_xy = self.bonus_pts[0]
                θ1      = math.atan2(dest_xy[1] - b.r[1],
                                     dest_xy[0] - b.r[0])
                θ       = 0.8*θ0 + 0.2*θ1 + random.uniform(-0.02, 0.02)
                b.v     = self.speed_AUd * np.array([math.cos(θ), math.sin(θ)])
                self.step_bee(b)
                b.trial = 0 if random.random() < b.fit else b.trial + 1

            # Onlookers
            fits = np.fromiter((e.fit for e in self.employed), float)
            pr   = fits / (fits.sum() or 1.0)
            for b in self.onlookers:
                leader = random.choices(self.employed, weights=pr, k=1)[0]
                b.v    = leader.v.copy()
                self.step_bee(b)
                b.trial = 0 if random.random() < b.fit else b.trial + 1

            # Scouts
            for b in self.scouts:
                if b.trial > 15:
                    θ   = random.uniform(0, 2*math.pi)
                    b.v = self.speed_AUd * np.array([math.cos(θ), math.sin(θ)])
                    b.trial = 0
                self.step_bee(b)

            # Book-keeping
            colony_best = max(self.bees, key=lambda x: x.fit)
            self.history.append(colony_best.fit)

            if colony_best.fit > self.global_best_fit:
                self.global_best_fit  = colony_best.fit
                self.global_best_step = step
                self.global_best_len  = colony_best.path_length
                self.global_best_path = np.vstack(colony_best.path).copy()

            if self.args.move_planets:
                self.sim.grav.step()

# ---------------------------------------------------------------------
# Driver (multi-epoch)
# ---------------------------------------------------------------------
def main():
    args = get_args()

    epoch_rows = []
    best_colony = None
    overall_best = -1e9

    for ep in range(1, args.epochs + 1):
        col = BeeColony(args)
        col.run()
        epoch_rows.append((ep,
                           col.global_best_fit,
                           col.global_best_step,
                           col.global_best_len))
        if col.global_best_fit > overall_best:
            overall_best = col.global_best_fit
            best_colony  = col

    # Output dir
    run_id  = f"{args.start}_{args.dest}_ABC_{args.n_bees}b"
    out_dir = os.path.join(OUT_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)

    # epoch_fitness.csv
    with open(os.path.join(out_dir, "epoch_fitness.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "best_fit", "best_step", "path_len"])
        w.writerows(epoch_rows)

    # summary.csv
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start", "dest", "n_bees", "epochs",
                    "best_fit", "best_step", "path_len"])
        w.writerow([args.start, args.dest, args.n_bees, args.epochs,
                    best_colony.global_best_fit,
                    best_colony.global_best_step,
                    best_colony.global_best_len])

    # Time-series
    save_fitness(best_colony, os.path.join(out_dir, "fitness.csv"))

    # Visualisations
    animate_colony(best_colony,
                   fname="colony_best.gif",
                   out_dir=out_dir,
                   show_orbits=args.show_orbits)

    best_colony.best_path = best_colony.global_best_path
    plot_best(best_colony,
              fname="best_path.png",
              out_dir=out_dir)

    print("DONE →", out_dir)

if __name__ == "__main__":
    main()
