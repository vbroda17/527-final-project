from dataclasses import dataclass, field
import numpy as np
#  add a forward reference type for the IDE; not required at runtime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rocket.sim import RocketSim            # avoid circular import

@dataclass
class Rocket:
    mass: float                 # kg (constant for now)
    r: np.ndarray               # AU
    v: np.ndarray               # AU/day
    max_thrust: float           # Newton
    sim: "RocketSim" = field(repr=False, default=None)

    path:  list = field(default_factory=list)
    speed: list = field(default_factory=list)

    # -------- control updated externally each step -------
    throttle: float = 0.0       # 0‑1
    angle:    float = 0.0       # rad (0 = +x)

    def record(self):
        self.path.append(self.r.copy())
        self.speed.append(np.linalg.norm(self.v))

    def accel_thrust(self):
        # convert N/kg to AU/day²
        a_si = (self.max_thrust * self.throttle) / self.mass     # m/s²
        return a_si * 86400**2 / (1.495978707e11) * np.array([np.cos(self.angle),
                                                              np.sin(self.angle)])

    def step(self, dt):
        # ----- capture logic identical to previous version -----
        bodies_xy = self.sim.grav.get_positions()
        sun_xy    = np.zeros(2)
        all_xy    = np.vstack([sun_xy, bodies_xy])
        radii = np.array(
            [0.03] + [b["radius"] + 5e-4 for b in self.sim.grav.bodies]
        )
        rel = all_xy - self.r
        d   = np.linalg.norm(rel, axis=1)
        hit = np.where(d <= radii)[0]
        if hit.size:
            k = hit[0]; self.r = all_xy[k].copy(); self.v[:] = 0; self.throttle = 0
            self.record(); return

        # ----- RK2 with thrust --------------------------------
        def grav(p): return self.sim.grav.gravity_accel(p)
        a0 = grav(self.r) + self.accel_thrust()
        v_half = self.v + 0.5 * a0 * dt
        r_half = self.r + 0.5 * self.v * dt
        a_half = grav(r_half) + self.accel_thrust()
        self.v += a_half * dt
        self.r += v_half * dt
        self.record()
