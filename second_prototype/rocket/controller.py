import numpy as np
from rocket.sim import RocketSim
from rocket.core import Rocket

AU_KM  = 1.495978707e8
DAY_S  = 86400.0
km_to_AU   = 1.0 / AU_KM
kms_to_AUday = DAY_S / AU_KM

# -----------------------------------------------------------
def create_rocket(sim: RocketSim,
                  start,                  # "Earth" or (x_AU, y_AU)
                  v0_kms=(0,0),
                  mass=2e5,               # kg
                  max_thrust_kN=40e3):    # kN   (give in kN for readability)
    """
    Returns a new Rocket already inserted into `sim`.
    `start`  ->  planet name   OR  tuple of AU coordinates.
    `v0_kms` -> inertial velocity in km/s.
    """
    if isinstance(start, str):
        idx,_ = sim.grav.get_body(start)
        r0 = sim.grav.traj[idx, 0].copy()
    else:
        r0 = np.asarray(start, float)

    v0 = np.asarray(v0_kms)*kms_to_AUday
    rocket = sim.add_rocket(r0, v0, mass=mass, max_thrust=max_thrust_kN*1000)
    return rocket

# -----------------------------------------------------------
def burn_toward_body(duration_days, target_name):
    """
    Controller: full throttle for `duration_days`, gimbal each tick toward
    the instantaneous position of `target_name`.  Then throttle=0, keep angle.
    """
    def ctrl(step, rk: Rocket, sim: RocketSim):
        throttle = 1.0 if step < duration_days else 0.0
        idx,_ = sim.grav.get_body(target_name)
        tgt = sim.grav.traj[idx, sim.grav.i]
        dx, dy = tgt - rk.r
        angle  = np.arctan2(dy, dx)
        return throttle, angle
    return ctrl

# -----------------------------------------------------------
def coast(controller_after_burn):
    """
    Tiny wrapper so you can chain:  full burn N days -> delegate controller.
    """
    def chain(step, rk, sim):
        if step < controller_after_burn[0]:   # first element is N_days
            return 1.0, rk.angle
        return controller_after_burn[1](step, rk, sim)
    return chain