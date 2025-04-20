# rocket/demo.py
"""
Run with
    python -m rocket.demo
Everything is saved to rocket_output/ (viz.py already does that).
"""

import os, multiprocessing as mp, numpy as np
from rocket.sim import RocketSim
from rocket import viz


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
    return sim.add_rocket(r0, v0, mass=mass, max_thrust=max_thrust, max_v_kmh=vmax_kmh)


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
        vel = (sim.grav.traj[idx, (sim.grav.i+1)%sim.grav.N_steps] - pos)/sim.dt
        dx, dy = pos - rk.r
        dist = np.hypot(dx, dy)
        tgt = pos + vel * dist * lead_factor
        tx, ty = tgt - rk.r
        angle = np.arctan2(ty, tx)
        return throttle, angle
    return ctrl


def fixed_direction(angle_rad, throttle=1.0, burn_days=None):
    """
    Point in a constant inertial direction.
    If burn_days is given, throttle=0 after that many steps.
    """
    def ctrl(step, rk, sim):
        thr = throttle
        if burn_days is not None and step >= burn_days:
            thr = 0.0
        return thr, angle_rad
    return ctrl


def dir_keyword(name, throttle=1.0, burn_days=None):
    mapping = {"right": 0, "up": np.pi/2, "left": np.pi, "down": -np.pi/2}
    if name not in mapping:
        raise ValueError("dir_keyword must be one of right, up, left, down")
    return fixed_direction(mapping[name], throttle, burn_days)


# =========================================================================
#  DEMO SCENARIOS ---------------------------------------------------------
# =========================================================================
def demo_single_ship():
    """
    One rocket on Earth: burn 20 days toward Mars, then coast.
    """
    sim = RocketSim(years=2, dt=1.0)
    rk  = add_ship(sim, start="Earth")
    controller = lead_body("Mars", lead_factor=0.08, throttle=1.0)
    sim.run(365, controller)
    viz.static(sim, rk); viz.snapshot(sim, rk); viz.animate(sim, rk)


def demo_retrograde_brake():
    """
    Fire retrograde for 50 days to drop into lower solar orbit.
    """
    sim = RocketSim(years=1, dt=1.0)
    rk  = add_ship(sim, start="Earth", vmax_kmh=20_000)
    controller = retrograde(throttle=0.8)
    sim.run(200, controller)
    viz.animate(sim, rk, fname="retrograde.gif")


def demo_vertical_rail():
    """
    5 rockets along x = -1 .. 1 AU at y = +1.8 AU, all thrust straight down.
    """
    sim = RocketSim(years=0.5, dt=1.0)
    xs = np.linspace(-1.0, 1.0, 5)
    rockets = [add_ship(sim, start=(x, 1.8)) for x in xs]
    controller = dir_keyword("down", throttle=1.0, burn_days=None)
    sim.run(180, controller)
    for i,rk in enumerate(rockets):
        viz.animate(sim, rk, fname=f"rail_{i}.gif")


# =========================================================================
#  MAIN  – choose which scenario to run -----------------------------------
# =========================================================================
def main():
    os.makedirs("rocket_output", exist_ok=True)

    # ------------------------------------------------------------------
    # EXACTLY ONE of the following lines should be uncommented at a time
    # ------------------------------------------------------------------
    # demo_single_ship()
    # demo_retrograde_brake()
    demo_vertical_rail()

if __name__ == "__main__":
    mp.freeze_support(); main()
