# rocket/demo.py
from rocket.sim import RocketSim
from rocket import viz
import numpy as np, os, multiprocessing as mp

def controller(step, rk, sim):
    # first 5 steps: 100% throttle straight down
    # if step < 5:
    #     return 1.0, -np.pi/2           # huge downward kick
    # afterwards let optimiser decide; here just coast
    return 0.0, rk.angle

def main():
    os.makedirs("rocket", exist_ok=True)
    sim = RocketSim(years=3, dt=1.0, elliptical=True)

    earth_idx,_ = sim.grav.get_body("Earth")
    r0 = sim.grav.traj[earth_idx,0].copy()
    v0 = np.array([0.0, 0.0])            # start at rest wrt Earth
    rk  = sim.add_rocket(r0, v0,
                         mass=2e5,        # 200 t ship
                         max_thrust=4e7)  # 40 MN (~SLS core)
    sim.run(n_steps=365, controller=controller)

    viz.static(sim, rk)
    viz.snapshot(sim, rk)
    viz.animate(sim, rk)
    print("created rocket/start.png snapshot.png flight.gif")

if __name__ == "__main__":
    mp.freeze_support()      # <-- vital on Windows
    main()
