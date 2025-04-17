#!/usr/bin/env python3
"""
rocket.py – simple rocket integrator that uses solar_system.GravSim
"""

import numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from solar_system import build_sim, G_AU3_EM_day2   # reuse planet code

class Rocket:
    def __init__(self, sim, r0, v0, thrust=np.zeros(2)):
        """
        sim    : GravSim instance
        r0,v0  : initial position (AU) & velocity (AU/day)
        thrust : constant acceleration (AU/day²) in *inertial* frame
        """
        self.sim     = sim
        self.r = r0.astype(float)
        self.v = v0.astype(float)
        self.a_thrust = thrust.astype(float)
        self.history  = [self.r.copy()]

    def step(self, dt):
        # gravitational acceleration from planets + Sun
        a_g = self.sim.gravity_accel(self.r)
        a   = a_g + self.a_thrust
        # simple RK2 (mid‑point) = much better than Euler for little cost
        v_half = self.v + 0.5 * a * dt
        r_half = self.r + 0.5 * self.v * dt
        a_half = self.sim.gravity_accel(r_half) + self.a_thrust
        self.v += a_half * dt
        self.r += v_half * dt
        self.history.append(self.r.copy())

    # ------------------------------------------------------------------
    def run(self, n_steps, dt):
        for _ in tqdm(range(n_steps), desc="integrating rocket"):
            self.step(dt);  self.sim.step()     # step planets same dt

# ─────────────────────  Plot helpers  ───────────────────────────
def plot_static(sim, rocket):
    pos0 = sim.traj[:,0]
    fig,ax = plt.subplots(figsize=(5,5))
    # planet orbits
    for j,b in enumerate(sim.bodies):
        ax.plot(sim.traj[j,:,0], sim.traj[j,:,1], '--', alpha=0.3)
        ax.scatter(pos0[j,0], pos0[j,1], s=20)
    ax.scatter(0,0,color='y',s=80)
    ax.scatter(rocket.r[0], rocket.r[1], color='magenta', marker='X', s=40)
    ax.set_aspect('equal'); ax.set_xlabel("AU"); ax.set_ylabel("AU")
    fig.tight_layout(); fig.savefig("rocket_start.png",dpi=160); plt.close(fig)

def plot_snapshot(sim, rocket, fname="rocket_snapshot.png"):
    hist = np.vstack(rocket.history)
    fig,ax = plt.subplots(figsize=(5,5))
    for j in range(len(sim.bodies)):
        ax.plot(sim.traj[j,:,0], sim.traj[j,:,1], '--', alpha=0.3)
    ax.plot(hist[:,0], hist[:,1], '-', color='magenta')
    ax.scatter(hist[-1,0], hist[-1,1], marker='X', color='magenta')
    ax.scatter(0,0,color='y',s=80)
    ax.set_aspect('equal'); ax.set_xlabel("AU"); ax.set_ylabel("AU")
    fig.tight_layout(); fig.savefig(fname,dpi=160); plt.close(fig)

def animate(sim, rocket, fname="rocket_anim.gif"):
    hist = np.vstack(rocket.history)
    fig,ax = plt.subplots(figsize=(5,5));  ax.set_facecolor('k')
    for j in range(len(sim.bodies)):
        ax.plot(sim.traj[j,:,0], sim.traj[j,:,1], '--', alpha=0.2)

    scat_r, = ax.plot([],[],'X',color='magenta')
    trail_r,= ax.plot([],[],'-',color='magenta',lw=1)
    ax.scatter(0,0,color='gold',s=80,zorder=6)

    lim = np.max(np.abs(sim.traj[:,:,0:2]))*1.2
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_aspect('equal')
    ax.set_xlabel("AU"); ax.set_ylabel("AU")

    def init():
        scat_r.set_data([],[]); trail_r.set_data([],[])
        return scat_r, trail_r

    def update(i):
        scat_r.set_data([hist[i, 0]], [hist[i, 1]])
        trail_r.set_data(hist[:i+1,0], hist[:i+1,1])
        return scat_r, trail_r

    ani = FuncAnimation(fig, update, frames=len(hist),
                        init_func=init, interval=25, blit=True)
    ani.save(fname, writer='pillow', fps=30)
    plt.close(fig)

# ─────────────────────  demo  ───────────────────────────
if __name__ == "__main__":
    sim = build_sim(years=3, dt=1.0, elliptical=True)
    # place rocket at Earth start & give small pro‑grade burn
    earth_idx,_ = sim.get_body("Earth")
    r0 = sim.traj[earth_idx,0].copy()
    v0 = np.array([0, 0.03])          # AU/day  (~5 km/s) tweak as you like
    rocket = Rocket(sim, r0, v0, thrust=np.zeros(2))

    N = 365                            # simulate 1 year at dt=1 day
    rocket.run(N, dt=1.0)

    plot_static(sim, rocket)
    plot_snapshot(sim, rocket, fname="rocket_after_1yr.png")
    animate(sim, rocket, fname="rocket_anim.gif")
    print("Images saved.")
