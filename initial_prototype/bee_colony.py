import argparse, random, os, datetime
import numpy as np
from dataclasses import dataclass

from physics import Planet, gravity_acc
from viz import (plot_best, plot_history, plot_swarm_final,
                 plot_swarm_snapshots)

# ---------- config ----------
@dataclass
class Config:
    bees: int = 40
    limit: int = 30          # scout trigger
    iters: int = 200
    waypoints: int = 8
    max_speed: float = 4e4   # m/s
    wp_sigma: float = 1e8    # for random pos shift
    thrust_sigma_angle: float = 0.2  # for random heading shift

    alpha: float = 1.0       # weight: time
    beta: float = 0.01       # weight: fuel
    gamma: float = 5.0       # weight: gravity penalty

    thrust_per_dt: float = 10.0      # baseline fuel usage
    misalign_penalty: float = 1.0    # penalty for angle mismatch

    snap_stride: int = 10    # how often to save a snapshot
    snap_pct: float = 0.25   # fraction or max # of bees to plot per snapshot
# ----------------------------

# planets (positions in metres, masses in kg – example)
SUN       = Planet('Sun',      0,         0,     1.989e30, radius=7e8)
START     = Planet('Start',  2.5e11,     0,     6e24,      radius=6.4e6)
OBSTACLE  = Planet('Obstacle',2.8e11, 1.0e11,   6e24,      radius=6.4e6)
TARGET    = Planet('Target',  3.0e11, 2.0e11,   6e24,      radius=6.4e6)
PLANETS = [SUN, START, OBSTACLE, TARGET]

# --- define a data structure for path segments ---
class Segment:
    def __init__(self, point: np.ndarray, theta: float):
        self.point = point   # 2D waypoint
        self.theta = theta   # thrust direction (radians)

def random_path(cfg: Config):
    segs = []
    # we skip adding a segment for START because that is "previous" pos
    for _ in range(cfg.waypoints - 1):
        x = random.uniform(min(START.pos[0], TARGET.pos[0]),
                           max(START.pos[0], TARGET.pos[0]))
        y = random.uniform(min(START.pos[1], TARGET.pos[1]),
                           max(TARGET.pos[1], START.pos[1]))
        theta = random.uniform(0, 2*np.pi)
        segs.append(Segment(np.array([x, y]), theta))
    # last segment forced to target pos, angle doesn't matter
    segs[-1].point = TARGET.pos.copy()
    segs[-1].theta = 0.0
    return segs

def cost(path, cfg: Config):
    """
    Return J = alpha * time + beta * fuel + gamma * grav_pen
    time ~ sum of distances / max_speed
    fuel ~ sum of thrust usage
    grav_pen ~ penalty for traveling near obstacle
    plus a small penalty for misalignment: angle between segment direction & thrust direction
    """
    total_time = 0.0
    total_fuel = 0.0
    grav_pen = 0.0

    prev_pos = START.pos
    for seg in path:
        p1 = seg.point
        dist = np.linalg.norm(p1 - prev_pos)
        if dist > 0:
            dt = dist / cfg.max_speed
            total_time += dt

            # "fuel" cost is baseline thrust usage * dt
            total_fuel += cfg.thrust_per_dt * dt

            # angle mismatch penalty
            seg_dir = (p1 - prev_pos)/dist
            thr_dir = np.array([np.cos(seg.theta), np.sin(seg.theta)])
            dot = np.clip(np.dot(seg_dir, thr_dir), -1, 1)
            angle = np.arccos(dot)
            total_fuel += cfg.misalign_penalty * abs(angle)

            # sample midpoint for gravity penalty
            mid = 0.5*(prev_pos + p1)
            r_obs = np.linalg.norm(mid - OBSTACLE.pos)
            if r_obs < 1.5e10: # "gravity shell" radius
                # approximate gravitational strength
                g_mag = np.linalg.norm(gravity_acc(mid, OBSTACLE))
                grav_pen += g_mag * dt

        prev_pos = p1

    J = cfg.alpha * total_time + cfg.beta * total_fuel + cfg.gamma * grav_pen
    return J

def tweak(path, cfg: Config):
    """
    Return a mutated copy of path:
    - shift waypoint pos
    - shift angle
    - keep last segment to target
    """
    new = []
    for seg in path[:-1]:
        # mutate position
        new_x = seg.point[0] + random.gauss(0, cfg.wp_sigma)
        new_y = seg.point[1] + random.gauss(0, cfg.wp_sigma)
        # mutate angle
        new_theta = (seg.theta + random.gauss(0, cfg.thrust_sigma_angle)) % (2*np.pi)
        new.append(Segment(np.array([new_x, new_y]), new_theta))
    # copy final (force target)
    last = Segment(TARGET.pos.copy(), 0.0)
    new.append(last)
    return new

def abc(cfg: Config):
    """
    Standard ABC with snapshots.
    We'll store snapshots at each 'snap_stride' iteration
    to visualize color-coded swarm evolution.
    """
    # init
    paths = [random_path(cfg) for _ in range(cfg.bees)]
    costs_ = [cost(p, cfg) for p in paths]
    trials = [0]*cfg.bees

    best_idx = int(np.argmin(costs_))
    best_path = [Segment(seg.point.copy(), seg.theta) for seg in paths[best_idx]]
    best_cost = costs_[best_idx]
    history = []

    snapshots = []  # list of swarm states (list of paths) at certain iters

    for it in range(cfg.iters):
        # ----- employed phase -----
        for i in range(cfg.bees):
            candidate = tweak(paths[i], cfg)
            c_cost = cost(candidate, cfg)
            if c_cost < costs_[i]:
                paths[i], costs_[i] = candidate, c_cost
                trials[i] = 0
            else:
                trials[i] += 1

        # ----- on-looker phase -----
        inv = 1/np.array(costs_)
        probs = inv / inv.sum()
        for _ in range(cfg.bees):
            i = np.random.choice(cfg.bees, p=probs)
            candidate = tweak(paths[i], cfg)
            c_cost = cost(candidate, cfg)
            if c_cost < costs_[i]:
                paths[i], costs_[i] = candidate, c_cost
                trials[i] = 0
            else:
                trials[i] += 1

        # ----- scout phase -----
        for i in range(cfg.bees):
            if trials[i] >= cfg.limit:
                paths[i] = random_path(cfg)
                costs_[i] = cost(paths[i], cfg)
                trials[i] = 0

        # track global best
        idx = int(np.argmin(costs_))
        if costs_[idx] < best_cost:
            best_cost = costs_[idx]
            best_path = [Segment(seg.point.copy(), seg.theta) for seg in paths[idx]]

        history.append(best_cost)

        # snapshot
        if it % cfg.snap_stride == 0:
            # copy swarm
            snap = []
            for p in paths:
                snap.append([Segment(seg.point.copy(), seg.theta) for seg in p])
            snapshots.append(snap)

    return best_path, paths, history, snapshots

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bees', type=int, default=40)
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--show', action='store_true')
    args = ap.parse_args()

    cfg = Config(bees=args.bees, iters=args.iters)

    best_path, swarm_paths, history, snapshots = abc(cfg)
    final_cost = cost(best_path, cfg)
    print(f"Best cost: {final_cost:.3e}")

    # create a new subfolder for each run
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_folder = os.path.join("runs", f"run_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)
    train_folder = os.path.join(run_folder, "training")
    os.makedirs(train_folder, exist_ok=True)

    # 1) Save best path
    best_path_png = os.path.join(run_folder, "best_path.png")
    # 2) Save final swarm
    swarm_png = os.path.join(run_folder, "swarm_iters.png")
    # 3) Save training curve
    curve_png = os.path.join(run_folder, "training_curve.png")

    # 4) We'll also save the snapshots as separate images in "training/" folder
    if args.show:
        plot_best(PLANETS, best_path, final_cost, save=best_path_png)
        plot_swarm_final(PLANETS, swarm_paths, save=swarm_png)
        plot_history(history, save=curve_png)
        # Now color-coded snapshots
        plot_swarm_snapshots(PLANETS, snapshots, pct=cfg.snap_pct, outdir=train_folder)
    else:
        # If not showing, just save them:
        plot_best(PLANETS, best_path, final_cost, save=best_path_png)
        plot_swarm_final(PLANETS, swarm_paths, save=swarm_png)
        plot_history(history, save=curve_png)
        plot_swarm_snapshots(PLANETS, snapshots, pct=cfg.snap_pct, outdir=train_folder)

if __name__ == '__main__':
    main()