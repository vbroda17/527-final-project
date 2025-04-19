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
    return 0.0000000001, rk.angle

# -----------------------------------------------------------
def main():
    os.makedirs("rocket", exist_ok=True)

    # 1) build solar‑system ephemeris (1½ years, 1‑day step)
    sim = RocketSim(years=1.5, dt=1.0, elliptical=True)

    # 2) get Earth's initial position *and its inertial velocity*
    earth_idx, _ = sim.grav.get_body("Earth")
    r0 = sim.grav.traj[earth_idx, 0].copy()
    v0 = (sim.grav.traj[earth_idx, 1] -    # simple finite difference
          sim.grav.traj[earth_idx, 0]) / sim.dt    # AU day⁻¹

    # 3) create rocket (200 t, 40 MN max thrust)
    rk = sim.add_rocket(r0, v0,          # start “sitting on” Earth
                        mass=2e5,
                        max_thrust=4e7)

    # 4) propagate 365 steps (1 year) with the controller above
    sim.run(n_steps=365, controller=controller)

    # 5) outputs
    viz.static(sim, rk)       # rocket/start.png
    viz.snapshot(sim, rk)     # rocket/snapshot.png
    viz.animate(sim, rk)      # rocket/flight.gif
    print("created rocket/start.png, snapshot.png, flight.gif")

if __name__ == "__main__":
    mp.freeze_support()   # Windows safety
    main()
