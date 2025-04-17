import numpy as np
from solar_system import build_sim
from rocket import Rocket

class RocketSim:
    def __init__(self, years=3, dt=1.0, elliptical=True):
        self.grav = build_sim(years, dt, elliptical=elliptical)
        self.dt   = dt
        self.rockets = []

    # ── add rockets ─────────────────────────────────────────────
    def add_rocket(self, r0, v0, thrust=np.zeros(2)):
        rk      = Rocket(r0.astype(float), v0.astype(float), thrust.astype(float))
        rk.sim  = self             # <‑‑ back‑reference used inside Rocket.step
        rk.record()
        self.rockets.append(rk)
        return rk

    # ── loop ────────────────────────────────────────────────────
    def run(self, n_steps, progress=True):
        from tqdm import tqdm
        rng = tqdm(range(n_steps), desc="propagating rockets") if progress else range(n_steps)
        for _ in rng:
            # advance each rocket
            for rk in self.rockets:
                rk.step(self.dt)
            # advance planet index
            self.grav.step()
