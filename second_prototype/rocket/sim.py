import numpy as np
from solar_system import build_sim
from rocket import Rocket
from tqdm import tqdm

class RocketSim:
    def __init__(self, years=3, dt=1.0, elliptical=True):
        self.grav = build_sim(years, dt, elliptical=elliptical)
        self.dt   = dt
        self.rockets = []

    # ── add rockets ─────────────────────────────────────────────
    def add_rocket(self, r0, v0, mass, max_thrust, max_v_kmh):
        rk = Rocket(r0, v0, mass, max_thrust, max_v_kmh, self)
        self.rockets.append(rk)
        return rk

    # ── loop ────────────────────────────────────────────────────
    def run(self, n_steps, controller, progress=True):
        rng = tqdm(range(n_steps)) if progress else range(n_steps)
        for step in rng:
            for rk in self.rockets:
                rk.throttle, rk.angle = controller(step, rk, self)
                rk.step(self.dt)
            self.grav.step()
