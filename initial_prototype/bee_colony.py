#!/usr/bin/env python3
"""
Toy Artificial Bee Colony for 2‑D inter‑planetary hop
"""
import argparse, random
import numpy as np
from dataclasses import dataclass
from physics import Planet, gravity_acc
from viz import plot_scene

# ---------- config ----------
@dataclass
class Config:
    bees: int = 40
    limit: int = 30          # scout trigger
    iters: int = 200
    waypoints: int = 8
    max_speed: float = 4e4   # m/s
    thrust_sigma: float = 5e-3
    wp_sigma: float = 1e8    # metres
    alpha: float = 1.0       # weight: time
    beta: float = 0.01       # weight: fuel
    gamma: float = 5.0       # weight: grav penalty
# ----------------------------

# planets (positions in metres, masses in kg – completely arbitrary for demo)
SUN       = Planet('Sun',      0,         0,     1.989e30, radius=7e8)
START     = Planet('Start',  2.5e11,     0,     6e24,      radius=6.4e6)
OBSTACLE  = Planet('Obstacle',1.6e11, 1.0e11,   6e24,      radius=6.4e6)
TARGET    = Planet('Target',  3.0e11, 2.0e11,   6e24,      radius=6.4e6)
PLANETS = [SUN, START, OBSTACLE, TARGET]

def random_path(cfg:Config):
    """Return list[waypoints] including start & target positions."""
    pts = [START.pos]
    for _ in range(cfg.waypoints-2):
        # uniform box around barycentre of start/target
        x = random.uniform(min(START.pos[0], TARGET.pos[0]),
                           max(START.pos[0], TARGET.pos[0]))
        y = random.uniform(min(START.pos[1], TARGET.pos[1]),
                           max(START.pos[1], TARGET.pos[1]))
        pts.append(np.array([x,y], dtype=float))
    pts.append(TARGET.pos)
    return pts

def cost(path, cfg:Config):
    """Compute J for a full path."""
    time = 0.0
    fuel = 0.0
    grav_pen = 0.0
    v_max = cfg.max_speed

    for i in range(len(path)-1):
        p0, p1 = path[i], path[i+1]
        seg = p1 - p0
        dist = np.linalg.norm(seg)
        if dist == 0: continue
        seg_dir = seg / dist

        # constant‑speed leg
        dt = dist / v_max
        time += dt

        # thrust to align with seg_dir (toy: assume Δv = v_max in that direction)
        fuel += v_max * cfg.beta

        # sample mid‑point for gravity penalty
        mid = (p0 + p1) / 2
        r = np.linalg.norm(mid - OBSTACLE.pos)
        if r < 1.5e10:                       # “gravity shell” radius (tunable)
            g_mag = np.linalg.norm(gravity_acc(mid, OBSTACLE))
            grav_pen += g_mag * dt

    J = cfg.alpha * time + cfg.beta * fuel + cfg.gamma * grav_pen
    return J

def tweak(path, cfg:Config):
    """Return a slightly mutated copy of path."""
    new = [START.pos]
    for p in path[1:-1]:
        if random.random() < 0.8:  # 80 % chance mutate
            p = p + np.random.normal(0, cfg.wp_sigma, size=2)
        new.append(p)
    new.append(TARGET.pos)
    return new

def abc(cfg:Config):
    # init
    paths = [random_path(cfg) for _ in range(cfg.bees)]
    costs = [cost(p,cfg) for p in paths]
    trials = [0]*cfg.bees
    best_idx = int(np.argmin(costs))
    best_path = paths[best_idx].copy()
    best_cost = costs[best_idx]
    history = []

    for it in range(cfg.iters):
        # ----- employed phase -----
        for i in range(cfg.bees):
            candidate = tweak(paths[i], cfg)
            c_cost = cost(candidate, cfg)
            if c_cost < costs[i]:
                paths[i], costs[i] = candidate, c_cost
                trials[i] = 0
            else:
                trials[i] += 1

        # ----- on‑looker phase -----
        inv = 1/np.array(costs)
        probs = inv / inv.sum()
        for _ in range(cfg.bees):
            i = np.random.choice(cfg.bees, p=probs)
            candidate = tweak(paths[i], cfg)
            c_cost = cost(candidate, cfg)
            if c_cost < costs[i]:
                paths[i], costs[i] = candidate, c_cost
                trials[i] = 0
            else:
                trials[i] += 1

        # ----- scout phase -----
        for i in range(cfg.bees):
            if trials[i] >= cfg.limit:
                paths[i] = random_path(cfg)
                costs[i] = cost(paths[i], cfg)
                trials[i] = 0

        # track global best
        idx = int(np.argmin(costs))
        if costs[idx] < best_cost:
            best_cost = costs[idx]
            best_path = paths[idx].copy()
        history.append(best_cost)

    return best_path, paths, history

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bees', type=int, default=40)
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--show', action='store_true')
    args = ap.parse_args()

    cfg = Config(bees=args.bees, iters=args.iters)
    best, swarm, _ = abc(cfg)

    print(f"Best cost: {cost(best,cfg):.3e}")
    if args.show:
        plot_scene(PLANETS, best, all_paths=swarm,
                title=f'ABC – best J={cost(best,cfg):.2e}')


if __name__ == '__main__':
    main()
