# simulation.py
import numpy as np
from helpers import G  # gravitational constant in chosen units

class Rocket:
    def __init__(self, position, velocity, mass=1.0):
        # position and velocity as numpy arrays (e.g., [x, y] in AU and AU/day)
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass  # rocket mass (not used in gravity calc, but could be used for fuel/thrust calculations)

class Simulation:
    def __init__(self, solar_system, rocket, time_step=1.0):
        """
        Initialize a simulation with a SolarSystem and a Rocket.
        time_step: integration time step in days (or chosen time unit).
        """
        self.sys = solar_system       # SolarSystem object with bodies and star
        self.rocket = rocket         # Rocket object with initial state
        self.dt = time_step
        self.current_time = 0.0
        # Storage for trajectory
        self.trajectory = []  # will store (time, position) for rocket

    def step(self):
        """Advance the simulation by one time step (dt)."""
        # Get current rocket state for readability
        r = self.rocket.position
        v = self.rocket.velocity
        # Compute acceleration due to gravity from all bodies:
        accel = np.zeros_like(r)
        # Include sun:
        sun_pos = np.array([0.0, 0.0])  # assume sun at origin
        rel_vec = sun_pos - r
        dist3 = (np.linalg.norm(rel_vec) ** 3) if np.linalg.norm(rel_vec) != 0 else np.inf
        accel += G * self.sys.star.mass * rel_vec / dist3
        # Include each planet:
        for body in self.sys.bodies:
            # If precomputed positions are available:
            if hasattr(self.sys, "orbit_cache"):
                # Determine index corresponding to current time:
                t_index = int(self.current_time / self.dt)
                body_pos = self.sys.orbit_cache["positions"][self.sys.bodies.index(body), t_index]
            else:
                # If no cache, compute planet position on the fly (e.g., advance by orbital motion)
                # (For simplicity, assume small dt so using constant angular velocity approximation)
                # In a real implementation, update body.position based on its orbital velocity or mean motion.
                body_pos = body.position  
            rel_vec = body_pos - r
            dist3 = (np.linalg.norm(rel_vec) ** 3) if np.linalg.norm(rel_vec) != 0 else np.inf
            accel += G * body.mass * rel_vec / dist3
        # Now accel is the total gravitational acceleration vector on the rocket.
        # Update rocket's velocity and position (Euler integration):
        v = v + accel * self.dt
        r = r + v * self.dt
        # Update rocket state:
        self.rocket.velocity = v
        self.rocket.position = r
        self.current_time += self.dt
        # Record trajectory:
        self.trajectory.append((self.current_time, r.copy()))

    def run(self, total_time):
        """Run the simulation for the given total time (in days, or chosen unit)."""
        steps = int(total_time / self.dt)
        # If using cached planet orbits, perhaps ensure they cover needed steps:
        if hasattr(self.sys, "precompute_orbits"):
            self.sys.precompute_orbits(total_time, self.dt)
        # Append initial state to trajectory
        self.trajectory = [(0.0, self.rocket.position.copy())]
        for _ in range(steps):
            self.step()
        return np.array([pos for _, pos in self.trajectory])
