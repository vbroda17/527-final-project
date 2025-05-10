# rocket/demo.py
"""
Run with  python -m rocket.demo
All GIF/PNG go to rocket_output/  (viz.py handles the folder).

Five demos:
    demo_single_ship        – one rocket, lead‑pursuit on a body
    demo_retrograde_brake   – single rocket, retro‑burn
    demo_vertical_rail      – N rockets dropped from a rail
    demo_vertical_rail_multi– same but rockets simulated together
    demo_no_gravity_ship    – rocket under 'no gravity' controller

Change RUN_YEARS or DT once at the top to affect every scenario.
"""

import os, multiprocessing as mp, numpy as np
from types import SimpleNamespace
from rocket.sim import RocketSim
from rocket import viz

# global knobs
RUN_YEARS = 3.5                   # length of each sim
DT        = .5                    # days per step
OUT_DIR   = "rocket_output"

# helpers
def start_pos(sim, start):
    if isinstance(start, str):
        idx,_ = sim.grav.get_body(start)
        r0 = sim.grav.traj[idx, 0].copy()
        v0 = (sim.grav.traj[idx,1]-sim.grav.traj[idx,0])/sim.dt
    else:
        r0 = np.asarray(start,float); v0=np.zeros(2)
    return r0,v0

def add_ship(sim, start, vmax_kmh=30_000, mass=2.0, max_thrust=1e-3):
    r0,v0 = start_pos(sim,start)
    return sim.add_rocket(r0,v0,mass=mass,max_thrust=max_thrust,
                          max_v_kmh=vmax_kmh)

# controllers
def retrograde(throttle=1.0):
    def c(_, rk, __):
        if np.allclose(rk.v,0): return 0.0, rk.angle
        return throttle, np.arctan2(-rk.v[1],-rk.v[0])
    return c

def point_at(body, throttle=1.0):
    def c(_,rk,sim):
        idx,_=sim.grav.get_body(body)
        dx,dy=sim.grav.traj[idx,sim.grav.i]-rk.r
        return throttle, np.arctan2(dy,dx)
    return c

def lead_body(body, lead=0.05, thr=1.0):
    def c(_,rk,sim):
        idx,_=sim.grav.get_body(body)
        pos = sim.grav.traj[idx,sim.grav.i]
        vel = (sim.grav.traj[idx,(sim.grav.i+1)%sim.grav.N_steps]-pos)/sim.dt
        d   = np.linalg.norm(pos-rk.r)
        tgt = pos+vel*d*lead
        return thr, np.arctan2(*(tgt-rk.r)[::-1])
    return c

def dir_keyword(name, thr=1.0):
    return lambda *_: (thr, {"right":0,"up":np.pi/2,
                             "left":np.pi,"down":-np.pi/2}[name])

# def no_gravity_controller(angle=0.0, thr=1.0):
#     """Controller ignoring gravity: we zero gravity for this sim."""
#     def factory(sim):
#         sim.grav.gravity_accel = lambda r: np.zeros(2)   # monkey‑patch
#         return lambda *_: (thr, angle)
#     return factory

# --- controller that zeroes gravity & keeps constant inertial velocity ---
def no_grav_const(body, speed_kmh=30_000):
    def factory(sim):
        sim.grav.gravity_accel = lambda r: np.zeros(2)          # kill gravity
        vmax = speed_kmh
        idx,_ = sim.grav.get_body(body)
        def ctrl(step, rk, sim):
            dx,dy = sim.grav.traj[idx, sim.grav.i] - rk.r
            ang = np.arctan2(dy, dx)
            return 1.0, ang      # always full‑speed toward body
        return ctrl
    return factory

# DEMOS
def demo_single_ship(start="Earth", dest="Mars"):
    sim  = RocketSim(years=RUN_YEARS, dt=DT)
    rk   = add_ship(sim,start)
    ctrl = lead_body(dest, lead=0.08, thr=1.0)
    sim.run(int(RUN_YEARS*365/DT), ctrl)
    viz.snapshot(sim,rk,fname=f"single_{start}_{dest}.png")
    viz.animate(sim,rk,fname=f"single_{start}_{dest}.gif")

def demo_retrograde_brake(start="Earth"):
    sim  = RocketSim(years=RUN_YEARS, dt=DT)
    rk   = add_ship(sim,start,vmax_kmh=20_000)
    ctrl = retrograde(throttle=0.8)
    sim.run(int(RUN_YEARS*365/DT), ctrl)
    viz.animate(sim,rk,fname=f"retro_{start}.gif")

def demo_vertical_rail(N=5,y=1.8,thr=1.0):
    sim = RocketSim(years=RUN_YEARS, dt=DT)
    xs  = np.linspace(-1,1,N)
    rockets=[add_ship(sim,(x,y)) for x in xs]
    ctrl = dir_keyword("down",thr)
    sim.run(int(RUN_YEARS*365/DT), ctrl)
    for i,rk in enumerate(rockets):
        viz.animate(sim,rk,fname=f"rail_sep_{i}.gif")

def demo_vertical_rail_multi(N=5,y=1.8,thr=1.0):
    sim = RocketSim(years=RUN_YEARS, dt=DT)
    xs  = np.linspace(-1,1,N)
    rockets=[add_ship(sim,(x,y)) for x in xs]
    ctrl = dir_keyword("down",thr)
    sim.run(int(RUN_YEARS*365/DT), ctrl)
    viz.animate_multi(sim, rockets, fname="rail_multi.gif")     

def demo_no_gravity_ship(start="Earth", dest="Mars"):
    sim  = RocketSim(years=RUN_YEARS, dt=DT)
    rk   = add_ship(sim,start)
    ctrl = no_grav_const(dest, 30_000)(sim) 
    sim.run(int(RUN_YEARS*365/DT), ctrl)
    viz.animate(sim,rk,fname=f"no_grav.gif")

# MAIN
def main():
    os.makedirs(OUT_DIR,exist_ok=True)

    # run every demo; comment out to skip
    # demo_single_ship("Venus","Earth")
    # demo_retrograde_brake("Earth")
    # demo_vertical_rail()
    demo_vertical_rail_multi()
    # demo_no_gravity_ship("Earth","Mars")

if __name__ == "__main__":
    mp.freeze_support(); main()
