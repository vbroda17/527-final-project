# rocket/__init__.py
import numpy as np
AU_M   = 1.495978707e11            # metres in 1 AU
DAY_S  = 86400.0
KMH_TO_AUDAY = (1000.0 * 24.0) / AU_M   # km h⁻¹ → AU day⁻¹

class Rocket:
    """
    Attributes you set once when you create the rocket
      r0            [AU]         initial position
      v0            [AU/day]     initial velocity (use planet's to 'ride along')
      mass          [EarthMass]  only needed for accel calc
      max_thrust    [EarthMass·AU/day²]  (same units you already use)
      max_v_kmh     [km/h]       *new*:  engine cannot accelerate past this speed

    Controller must set *every tick*
      throttle  (0‑1)
      angle     (rad)

    Units everywhere else stay:  distance = AU,  time = day.
    """
    def __init__(self, r0, v0, mass, max_thrust, max_v_kmh, sim):
        self.r = np.asarray(r0, float)
        self.v = np.asarray(v0, float)
        self.mass = mass
        self.max_thrust = max_thrust
        self.vmax = max_v_kmh * KMH_TO_AUDAY   # AU/day
        self.throttle = 0.0
        self.angle    = 0.0
        self.sim = sim          # back‑reference to RocketSim
        self.path = [self.r.copy()]

    # ---------------- internal helpers ----------------
    def _a_thrust(self):
        """Engine acceleration vector [AU/day²] for current throttle & angle."""
        a_mag = (self.max_thrust * self.throttle) / self.mass     # AU/day²
        return a_mag * np.array([np.cos(self.angle), np.sin(self.angle)])

    # ---------------- main integrator -----------------
    def step(self, dt):
        # engine contribution
        self.v += self._a_thrust() * dt

        # cap speed *up to* vmax (gravity can exceed it)
        speed = np.linalg.norm(self.v)
        if speed < self.vmax:                         # only engine may boost
            if speed > self.vmax:                     # numerical edge
                self.v *= self.vmax / speed

        # gravity contribution
        self.v += self.sim.grav.gravity_accel(self.r) * dt
        self.r += self.v * dt
        self.path.append(self.r.copy())
