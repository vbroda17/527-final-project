from __future__ import annotations

import argparse
import datetime
import pathlib
import random
from dataclasses import dataclass

import numpy as np

import viz
from physics import Planet, gravity

# Bodies
SUN = Planet("Sun", 1.989e30, static_pos=[0.0, 0.0])
START = Planet("Start", 6e24, static_pos=[2.5e11, 0.0])
MARS = Planet("Mars", 6e24, static_pos=[1.5e11, 1.0e11])
TARGET = Planet("Target", 6e24, static_pos=[3.0e11, 2.0e11])

BODIES = [SUN, START, MARS, TARGET]
FOOD_BODIES = [b for b in BODIES if b.name not in {"Sun", "Start"}]

# Configuration
@dataclass
class Config:
    bees: int = 60
    scouts: int = 10
    iterations: int = 400

    dt: float = 2_000.0
    segment_time: float = 2.0e5
    thrust_max: float = 5.0e4

    waypoint_sigma: float = 1.0e9
    angle_sigma: float = 0.3

    alpha: float = 1.0
    beta: float = 0.01
    delta: float = 0.001  # distance weight

    snapshot_stride: int = 10
    snapshot_pct: float = 0.25
    capture_radius: float = 2.0e9


# Data classes
class Waypoint:
    def __init__(self, position: np.ndarray, theta: float) -> None:
        self.position = position
        self.theta = theta


class Bee:
    def __init__(self, path: list[Waypoint]) -> None:
        self.path = path
        self.site: Planet | None = None
        self.cost: float = float("inf")


# Path utilities
def random_waypoint(cfg: Config) -> Waypoint:
    x = random.uniform(START.static_pos[0], TARGET.static_pos[0])
    y = random.uniform(START.static_pos[1], TARGET.static_pos[1])
    theta = random.uniform(0.0, 2.0 * np.pi)
    return Waypoint(np.array([x, y]), theta)


def new_random_path(cfg: Config) -> list[Waypoint]:
    return [random_waypoint(cfg) for _ in range(5)]


def mutate_path(path: list[Waypoint], cfg: Config) -> list[Waypoint]:
    mutated: list[Waypoint] = []
    for wp in path:
        new_pos = wp.position + np.random.normal(0.0, cfg.waypoint_sigma, 2)
        new_theta = (wp.theta + np.random.normal(0.0, cfg.angle_sigma)) % (
            2.0 * np.pi
        )
        mutated.append(Waypoint(new_pos, new_theta))
    return mutated


# Physics simulation and cost
def simulate(path: list[Waypoint], cfg: Config, food: Planet):
    position = np.array(START.static_pos, dtype=float)
    velocity = np.zeros(2)
    fuel = 0.0
    time_elapsed = 0.0

    for wp in path:
        thrust_dir = np.array([np.cos(wp.theta), np.sin(wp.theta)])
        thrust = thrust_dir * cfg.thrust_max
        steps = int(cfg.segment_time / cfg.dt)
        for _ in range(steps):
            acc = thrust + gravity(position, BODIES, time_elapsed)
            velocity += acc * cfg.dt
            position += velocity * cfg.dt
            fuel += np.linalg.norm(thrust) * cfg.dt
            time_elapsed += cfg.dt
            if np.linalg.norm(position - food.pos(time_elapsed)) < cfg.capture_radius:
                distance = np.linalg.norm(position - START.static_pos)
                return time_elapsed, fuel, distance
        position = wp.position.copy()

    distance = np.linalg.norm(position - food.pos(time_elapsed))
    return (
        time_elapsed + 1.0e9,
        fuel + 1.0e9,
        distance + 1.0e12,
    )


def evaluate(path: list[Waypoint], cfg: Config):
    best_cost = float("inf")
    best_food: Planet | None = None
    for food in FOOD_BODIES:
        time_val, fuel_val, dist_val = simulate(path, cfg, food)
        score = cfg.alpha * time_val + cfg.beta * fuel_val + cfg.delta * dist_val
        if score < best_cost:
            best_cost = score
            best_food = food
    return best_cost, best_food


# Artificial Bee Colony core
def artificial_bee_colony(cfg: Config):
    scouts = [Bee(new_random_path(cfg)) for _ in range(cfg.scouts)]
    employed: list[Bee] = []
    onlookers = [Bee(new_random_path(cfg)) for _ in range(cfg.bees - cfg.scouts)]

    best_cost = float("inf")
    best_path: list[Waypoint] | None = None
    history: list[float] = []
    snapshots: list[list[np.ndarray]] = []

    for it in range(cfg.iterations):
        # scouts 
        for bee in scouts[:]:
            candidate = mutate_path(bee.path, cfg)
            cost_val, food = evaluate(candidate, cfg)
            if cost_val < bee.cost:
                bee.path = candidate
                bee.cost = cost_val
            if food is not None:
                bee.site = food
                employed.append(bee)
                scouts.remove(bee)
                onlookers.append(Bee(new_random_path(cfg)))

        # employed 
        for bee in employed:
            candidate = mutate_path(bee.path, cfg)
            cost_val, _ = evaluate(candidate, cfg)
            if cost_val < bee.cost:
                bee.path = candidate
                bee.cost = cost_val

        # on‑lookers
        if employed:
            nectar = np.array([1.0 / b.cost for b in employed])
            prob = nectar / nectar.sum()
            for bee in onlookers:
                idx = np.random.choice(len(employed), p=prob)
                source = employed[idx]
                candidate = mutate_path(source.path, cfg)
                cost_val, _ = evaluate(candidate, cfg)
                if cost_val < bee.cost:
                    bee.path = candidate
                    bee.cost = cost_val

        # global best 
        all_bees = scouts + employed + onlookers
        current_best = min(all_bees, key=lambda b: b.cost)
        if current_best.cost < best_cost:
            best_cost = current_best.cost
            best_path = current_best.path.copy()

        history.append(best_cost)
        if it % cfg.snapshot_stride == 0:
            snap = [np.vstack([w.position for w in bee.path]) for bee in all_bees]
            snapshots.append(snap)

    return best_path, history, snapshots


# Main entry
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bees", type=int, default=60)
    parser.add_argument("--scouts", type=int, default=10)
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--delta", type=float, default=0.001)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    cfg = Config(
        bees=args.bees,
        scouts=args.scouts,
        iterations=args.iters,
        delta=args.delta,
    )

    best_path, history, snapshots = artificial_bee_colony(cfg)

    run_dir = pathlib.Path("runs") / f"run_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    train_dir = run_dir / "training"
    train_dir.mkdir(parents=True, exist_ok=True)

    best_points = np.vstack([w.position for w in best_path])
    viz.save_best_plot(run_dir, BODIES, best_points)
    viz.save_training_curve(run_dir, history)
    viz.save_swarm_snapshots(train_dir, BODIES, snapshots, cfg.snapshot_pct)


if __name__ == "__main__":
    main()