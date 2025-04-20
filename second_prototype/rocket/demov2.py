# rocket/demo.py
"""
Run with
    python -m rocket.demo

All PNG/GIF outputs land in rocket_output/ (handled by viz.py).
"""

import os, multiprocessing as mp, numpy as np
from rocket.sim import RocketSim
from rocket import viz

# =========================================================================
#  GLOBAL SIMULATION SETTINGS  -------------------------------------------                         ← CHANGED
# =========================================================================
RUN_YEARS = 2          # how long to simulate (floats OK, e.g. 1.75)    ← CHANGED
DT        = 1.0          # integration step in *days*                     ← CHANGED
# =========================================================================


# =========================================================================
#  HELPERS ----------------------------------------------------------------
# =========================================================================
def start_pos(sim, start):
    """Return (r0, v0) in AU / AU day⁻¹ from body name or raw coords."""
    if isinstance(start, str):
        idx, _ = sim.grav.get_body(start)
        r0 = sim.grav.traj[idx, 0].copy()
        v0 = (sim.grav.traj[idx, 1] - sim.grav.traj[idx, 0]) / sim.dt
    else:                           # tuple or ndarray
        r0 = np.asarray(start, float)
        v0 = np.zeros(2)
    return r0, v0


def add_ship(sim, start, vmax_kmh=30_000, mass=2.0, max_thrust=1e-3):
    """Create a rocket with chosen MAX speed (engine throttle 0‑1 sets fraction)."""
    r0, v0 = start_pos(sim, start)
    return sim.add_rocket(r0, v0,
                          mass=mass, max_thrust=max_thrust, max_v_kmh=vmax_kmh)


# =========================================================================
#  CONTROLLERS  (return throttle, angle) ----------------------------------
# =========================================================================
def retrograde(throttle=1.0):
    """Always point opposite to current velocity."""
    def ctrl(step, rk, sim):
        vx, vy = rk.v
        if vx == vy == 0:
            return 0.0, rk.angle
        angle = np.arctan2(-vy, -vx)
        return throttle, angle
    return ctrl


def point_at(body, throttle=1.0):
    """Always point directly at `body`."""
    def ctrl(step, rk, sim):
        idx, _ = sim.grav.get_body(body)
        tgt = sim.grav.traj[idx, sim.grav.i]
        dx, dy = tgt - rk.r
        angle = np.arctan2(dy, dx)
        return throttle, angle
    return ctrl


def lead_body(body, lead_factor=0.05, throttle=1.0):
    """
    Aim ahead of the body along its velocity vector.
    lead_factor * distance adds to position to approximate intercept.
    """
    def ctrl(step, rk, sim):
        idx, _ = sim.grav.get_body(body)
        pos = sim.grav.traj[idx, sim.grav.i]
        vel = (sim.grav.traj[idx, (sim.grav.i+1)%sim.grav.N_steps] - pos) / sim.dt
        dist = np.linalg.norm(pos - rk.r)
        tgt = pos + vel * dist * lead_factor
        angle = np.arctan2(*(tgt - rk.r)[::-1])
        return throttle, angle
    return ctrl


def fixed_direction(angle_rad, throttle=1.0, burn_days=None):
    """
    Point in a constant inertial direction.
    If burn_days is given, throttle=0 after that many steps.
    """
    def ctrl(step, rk, sim):
        thr = throttle if (burn_days is None or step < burn_days) else 0.0
        return thr, angle_rad
    return ctrl


def dir_keyword(name, throttle=1.0, burn_days=None):
    mapping = {"right": 0, "up": np.pi/2, "left": np.pi, "down": -np.pi/2}
    if name not in mapping:
        raise ValueError("dir_keyword must be right, up, left or down")
    return fixed_direction(mapping[name], throttle, burn_days)


# =========================================================================
#  DEMO SCENARIOS ---------------------------------------------------------
# =========================================================================
def demo_single_ship(sim):
    """
    One rocket on Earth: burn 20 days toward Mars, then coast.
    """
    rk = add_ship(sim, start="Earth")
    controller = lead_body("Mars", lead_factor=0.08, throttle=1.0)
    sim.run(total_steps, controller)
    viz.static(sim, rk); viz.snapshot(sim, rk); viz.animate(sim, rk)


def demo_retrograde_brake(sim):
    """
    Fire retrograde for 50 days to drop into lower solar orbit.
    """
    rk = add_ship(sim, start="Earth", vmax_kmh=20_000)
    controller = retrograde(throttle=0.8)
    sim.run(total_steps, controller)
    viz.animate(sim, rk, fname="retrograde.gif")


def demo_vertical_rail(sim):
    """
    5 rockets along x = -1 .. 1 AU at y = +1.8 AU, all thrust straight down.
    """
    xs = np.linspace(-1.0, 1.0, 5)
    rockets = [add_ship(sim, start=(x, 1.8)) for x in xs]
    controller = dir_keyword("down", throttle=1.0)
    sim.run(total_steps, controller)
    for i, rk in enumerate(rockets):
        viz.animate(sim, rk, fname=f"rail_{i}.gif")


# =========================================================================
#  MAIN  – choose which scenario to run -----------------------------------
# =========================================================================
def main():
    os.makedirs("rocket_output", exist_ok=True)

    # build one shared sim so all helpers see the same ephemeris
    global total_steps                                      # ← CHANGED
    sim = RocketSim(years=RUN_YEARS, dt=DT, elliptical=True) # ← CHANGED
    total_steps = int(RUN_YEARS * 365 / DT)                  # ← CHANGED

    # -----------------------------------
    # uncomment ONE of the following lines
    # -----------------------------------
    demo_single_ship(sim)
    # demo_retrograde_brake(sim)
    # demo_vertical_rail(sim)

if __name__ == "__main__":
    mp.freeze_support()
    main()
