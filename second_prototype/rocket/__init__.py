from dataclasses import dataclass, field
import numpy as np
#  add a forward reference type for the IDE; not required at runtime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rocket.sim import RocketSim            # avoid circular import

@dataclass
class Rocket:
    r: np.ndarray          # position  (AU)
    v: np.ndarray          # velocity  (AU/day)
    thrust: np.ndarray     # constant accel (AU/day²)
    path: list = field(default_factory=list)
    sim: "RocketSim" = field(default=None, repr=False)     # ← NEW

    def record(self):
        self.path.append(self.r.copy())

    # one RK2 step in the inertial frame
    def step(self, dt):
        # ------------------------------------------------- 1. check for capture
        bodies_xy = self.sim.grav.get_positions()            # (N,2)
        sun_xy    = np.zeros(2)
        all_xy    = np.vstack([sun_xy, bodies_xy])
        # capture radii (AU):  Sun = 0.03,  planets = body_radius + 5e‑4
        radii = np.array(
            [0.03] + [b["radius"] + 5e-4 for b in self.sim.grav.bodies]
        )

        rel = all_xy - self.r              # (N,2)
        d   = np.linalg.norm(rel, axis=1)
        hit = np.where(d <= radii)[0]      # indices of bodies we touch

        if hit.size:                       # we hit the first in the list
            k = hit[0]
            target_xy = all_xy[k]
            # ------------------------------------------------- lock on
            self.r = target_xy.copy()
            # adopt body's velocity (Sun's is 0)
            if k == 0:
                self.v[:] = 0.0
            else:
                # planet velocity from trajectory difference
                j = k - 1                  # planet index in traj
                i = self.sim.grav.i
                # finite‑difference orbit velocity (1‑frame look‑ahead)
                nxt = (i + 1) % self.sim.grav.N_steps
                self.v = (self.sim.grav.traj[j, nxt] -
                        self.sim.grav.traj[j, i]) / self.sim.dt
            self.thrust[:] = 0.0           # thrust off while attached
            self.record()
            return                         # skip free‑flight physics

        # ------------------------------------------- 2. normal RK2 free flight
        a_g = self.sim.grav.gravity_accel(self.r)
        a   = a_g + self.thrust
        v_half = self.v + 0.5 * a * dt
        r_half = self.r + 0.5 * self.v * dt
        a_half = self.sim.grav.gravity_accel(r_half) + self.thrust
        self.v += a_half * dt
        self.r += v_half * dt
        self.record()
