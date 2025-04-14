import argparse, random
from dataclasses import dataclass
import numpy as np

from physics import Planet, gravity_acc
from viz import plot_best, plot_swarm_iter, plot_history

# --------------------------- configuration --------------------------- #
@dataclass
class Config:
    bees: int = 60
    limit: int = 30           # scout trigger
    iters: int = 400
    waypoints: int = 8

    # mutation scales
    wp_sigma: float = 1e8             # metres
    thrust_sigma_angle: float = 0.2   # radians

    # physics
    max_speed: float = 4e4            # m/s (constant cruise)
    thrust_per_dt: float = 10.0       # fuel units per second of thrust
    misalign_penalty: float = 1.0     # extra fuel per rad of mis‑alignment

    # objective weights
    alpha: float = 1.0   # flight time
    beta: float = 0.01   # fuel
    gamma: float = 5.0   # gravity penalty

    # bookkeeping
    snap_stride: int = 10  # save swarm every k iterations

# --------------------------- planets -------------------------------- #
SUN      = Planet("Sun",      0,        0,     1.989e30, radius=7e8)
START    = Planet("Start",  2.5e11,    0,     6e24)
OBSTACLE = Planet("Obstacle",1.6e11, 1.0e11,  6e24)
TARGET   = Planet("Target",  3.0e11, 2.0e11,  6e24)
PLANETS  = [SUN, START, OBSTACLE, TARGET]

# --------------------------- data classes --------------------------- #
class Segment:
    """Waypoint + thrust direction (theta)."""
    def __init__(self, point: np.ndarray, theta: float):
        self.point = point
        self.theta = theta  # radians [0, 2π)

# --------------------------- helpers -------------------------------- #

def random_path(cfg: Config):
    segs = []
    # internal waypoints (excluding start/target)
    for _ in range(cfg.waypoints - 2):
        x = random.uniform(min(START.pos[0], TARGET.pos[0]), max(START.pos[0], TARGET.pos[0]))
        y = random.uniform(min(START.pos[1], TARGET.pos[1]), max(START.pos[1], TARGET.pos[1]))
        theta = random.uniform(0, 2 * np.pi)
        segs.append(Segment(np.array([x, y]), theta))
    # final segment leads to target; theta unused
    segs.append(Segment(TARGET.pos, 0.0))
    return segs


def path_points(path):
    """Return full list of np.array points including START and TARGET."""
    pts = [START.pos]
    pts.extend(seg.point for seg in path)
    return pts


def cost(path, cfg: Config):
    time = 0.0
    fuel = 0.0
    grav_pen = 0.0

    pts = path_points(path)
    for idx in range(len(pts) - 1):
        p0, p1 = pts[idx], pts[idx + 1]
        seg_vec = p1 - p0
        dist = np.linalg.norm(seg_vec)
        if dist == 0:
            continue
        dir_vec = seg_vec / dist
        dt = dist / cfg.max_speed
        time += dt

        # thrust fuel (constant magnitude per segment)
        fuel += cfg.thrust_per_dt * dt

        # mis‑alignment penalty
        theta = path[idx].theta if idx < len(path) - 1 else 0.0
        thrust_dir = np.array([np.cos(theta), np.sin(theta)])
        mis = np.arccos(np.clip(np.dot(dir_vec, thrust_dir), -1, 1))
        fuel += cfg.misalign_penalty * abs(mis)

        # gravity penalty (sample mid‑point vs obstacle)
        mid = (p0 + p1) / 2
        r = np.linalg.norm(mid - OBSTACLE.pos)
        if r < 1.5e10:  # arbitrary shell radius
            g_mag = np.linalg.norm(gravity_acc(mid, OBSTACLE))
            grav_pen += g_mag * dt

    return cfg.alpha * time + cfg.beta * fuel + cfg.gamma * grav_pen


def tweak(path, cfg: Config):
    new = []
    for seg in path[:-1]:  # leave target untouched
        if random.random() < 0.8:
            point = seg.point + np.random.normal(0, cfg.wp_sigma, 2)
            theta = (seg.theta + np.random.normal(0, cfg.thrust_sigma_angle)) % (2 * np.pi)
        else:
            point, theta = seg.point.copy(), seg.theta
        new.append(Segment(point, theta))
    new.append(path[-1])  # keep final segment
    return new

# --------------------------- ABC core ------------------------------- #

def abc(cfg: Config):
    paths = [random_path(cfg) for _ in range(cfg.bees)]
    costs = [cost(p, cfg) for p in paths]
    trials = [0] * cfg.bees
    best_idx = int(np.argmin(costs))
    best_path = paths[best_idx]
    best_cost = costs[best_idx]

    history = []
    snapshots = []

    for it in range(cfg.iters):
        # employed phase
        for i in range(cfg.bees):
            cand = tweak(paths[i], cfg)
            c_cost = cost(cand, cfg)
            if c_cost < costs[i]:
                paths[i], costs[i] = cand, c_cost
                trials[i] = 0
            else:
                trials[i] += 1

        # on‑looker phase
        inv = 1 / np.array(costs)
        probs = inv / inv.sum()
        for _ in range(cfg.bees):
            i = np.random.choice(cfg.bees, p=probs)
            cand = tweak(paths[i], cfg)
            c_cost = cost(cand, cfg)
            if c_cost < costs[i]:
                paths[i], costs[i] = cand, c_cost
                trials[i] = 0
            else:
                trials[i] += 1

        # scout phase
        for i in range(cfg.bees):
            if trials[i] >= cfg.limit:
                paths[i] = random_path(cfg)
                costs[i] = cost(paths[i], cfg)
                trials[i] = 0

        # update global best
        idx = int(np.argmin(costs))
        if costs[idx] < best_cost:
            best_cost = costs[idx]
            best_path = paths[idx]

        history.append(best_cost)
        if it % cfg.snap_stride == 0:
            snapshots.append([path_points(p) for p in paths])

    return best_path, paths, history, snapshots

# --------------------------- CLI ------------------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bees", type=int, default=60)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--pct", type=float, default=0.25, help="fraction (0‑1) or max int of bees to plot per snapshot")
    args = ap.parse_args()

    cfg = Config(bees=args.bees, iters=args.iters)
    best, swarm, history, snaps = abc(cfg)

    best_cost = cost(best, cfg)
    print(f"Best cost: {best_cost:.3e}")

    if args.show:
        plot_best(PLANETS, path_points(best), best_cost, save="best_path.png")
        plot_swarm_iter(PLANETS, snaps, pct=args.pct, save="swarm_iters.png")
        plot_history(history, save="training_curve.png")


if __name__ == "__main__":
    main()