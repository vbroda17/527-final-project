# rocket/demo.py
from rocket.sim import RocketSim
from rocket import viz
import numpy as np, os, multiprocessing as mp

# -----------------------------------------------------------
# CONTROLLER
#   • called once every time‑step
#   • must RETURN:   (throttle,  angle)
#
#     throttle  ∈ [0, 1]        → fraction of max_thrust to use
#     angle     (radians)       → direction of thrust in the inertial XY‑plane
#
#     0   rad  = +X  (to the right)
#     π/2 rad = +Y  (up)
#     π   rad = –X  (left)
#     3π/2     = –Y  (down)     [or  –π/2]
#
# EXAMPLE  ↓  — rocket coasts (no thrust)
def controller(step, rk, sim):
    # full throttle straight 'right' for first 7 days
    if step < 10:
        return 1.0, 0.0          # throttle 100 %, angle 0 rad (= +X)
    return 0.1, rk.angle

# -----------------------------------------------------------
def main():
    os.makedirs("rocket", exist_ok=True)
    years = 3
    # 1) build solar‑system ephemeris (1½ years, 1‑day step)
    sim = RocketSim(years=years, dt=1.0, elliptical=True)

    # 2) get Earth's initial position *and its inertial velocity*
    earth_idx, _ = sim.grav.get_body("Earth")
    r0 = sim.grav.traj[earth_idx, 0].copy()
    v0 = (sim.grav.traj[earth_idx, 1] - sim.grav.traj[earth_idx, 0]) / sim.dt

    # 3) create rocket (200 t, 40 MN max thrust)
    rk = sim.add_rocket(
            r0, v0,
            mass=2.0,               # EarthMass  (tiny test craft)
            max_thrust=0.001,       # EarthMass·AU/day²  (very small engine)
            max_v_kmh=30_000        # 30 000 km/h ≈ 0.0048 AU/day
        )

    # 4) propagate 365 steps (1 year) with the controller above
    sim.run(n_steps=365*years, controller=controller)

    # 5) outputs
    viz.static(sim, rk)       # rocket/start.png
    viz.snapshot(sim, rk)     # rocket/snapshot.png
    viz.animate(sim, rk)      # rocket/flight.gif
    print("created rocket/start.png, snapshot.png, flight.gif")

if __name__ == "__main__":
    mp.freeze_support()   # Windows safety
    main()
