# rocket/demo.py
from rocket.sim import RocketSim
from rocket import viz
import numpy as np, os, multiprocessing as mp

def main():
    os.makedirs("rocket", exist_ok=True)

    sim = RocketSim(years=3, dt=1.0, elliptical=True)

    earth_idx, _ = sim.grav.get_body("Earth")
    r0 = sim.grav.traj[earth_idx, 0].copy()
    v0 = np.array([0.0, -0.01])   # 0.01 AU/day ≈ 1.7 km/s
    rk  = sim.add_rocket(r0, v0)

    sim.run(n_steps=365)

    viz.static(sim, rk)
    viz.snapshot(sim, rk)
    viz.animate(sim, rk)
    print("rocket/start.png  rocket/snapshot.png  rocket/flight.gif  created.")

if __name__ == "__main__":
    mp.freeze_support()      # <-- vital on Windows
    main()
