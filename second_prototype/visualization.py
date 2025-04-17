# visualization.py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from helpers import G, compute_gravitational_param
import numpy as np
def plot_orbits(solar_system, rocket=None, time_index=None, show_orbit_count=True):
    """
    Plot a static view of the orbits and positions at a given time index.
    """
    bodies = solar_system.bodies
    star = solar_system.star
    # Determine positions to plot (use time_index or default to 0)
    if time_index is None:
        time_index = 0
    if hasattr(solar_system, "orbit_cache"):
        positions = solar_system.orbit_cache["positions"][:, time_index]  # all bodies positions
    else:
        # use current positions stored in bodies
        positions = np.array([body.position for body in bodies])
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    # Plot star (sun) at origin:
    ax.scatter(0, 0, color='yellow', s=80, label=star.name)
    # Plot each planet
    colors = plt.cm.tab10(range(len(bodies)))  # color map for distinct colors
    for i, body in enumerate(bodies):
        pos = positions[i]
        # Draw orbit ellipse (approximate) or full trail if available
        if hasattr(solar_system, "orbit_cache"):
            # Use all cached positions for body i to draw orbit path
            orbit_path = solar_system.orbit_cache["positions"][i, :time_index+1]
            ax.plot(orbit_path[:,0], orbit_path[:,1], ls='--', color=colors[i], linewidth=0.8)
        else:
            # Draw an ellipse based on orbital parameters if available
            if body.a and body.e is not None:
                # parametric ellipse for orbit:
                theta = np.linspace(0, 2*np.pi, 200)
                r = (body.a*(1-body.e**2)) / (1 + body.e * np.cos(theta))
                orbit_x = r * np.cos(theta)
                orbit_y = r * np.sin(theta)
                ax.plot(orbit_x, orbit_y, ls='--', color=colors[i], linewidth=0.8)
        # Plot planet position
        ax.scatter(pos[0], pos[1], color=colors[i], s=40)
        # Label planet name (and orbits count if requested)
        label = body.name
        if show_orbit_count and body.a:
            # Calculate orbits completed ~ (current_time / period)
            mu = solar_system.star.mass * G  # G*M (in appropriate units)
            period = 2*np.pi * np.sqrt(body.a**3 / mu)
            orbits = (solar_system.orbit_cache["times"][time_index] / period) if hasattr(solar_system, "orbit_cache") else 0
            label += f" ({int(orbits)} orbits)"
        ax.text(pos[0]*1.02, pos[1]*1.02, label, fontsize=8)
    # Plot rocket if provided
    if rocket is not None:
        rpos = rocket.position if isinstance(rocket, (list, np.ndarray)) else rocket.position
        ax.scatter(rpos[0], rpos[1], color='magenta', s=25, marker='X', label='Rocket')
        # Draw rocket trail if trajectory available
        if hasattr(solar_system, "orbit_cache") and time_index > 0:
            # If we have rocket trajectory separately passed or accessible:
            # (Assume rocket trajectory is an array of positions corresponding to times)
            pass  # This can be implemented by passing rocket trajectory as parameter
    ax.set_xlabel(f"Distance [{solar_system.units['distance']}]")
    ax.set_ylabel(f"Distance [{solar_system.units['distance']}]")
    ax.legend()
    plt.title("Solar System Orbits")
    return fig, ax

def animate_orbits(solar_system, rocket_trajectory=None, interval=50, frames=None):
    """
    Animate the motion of planets (and rocket if given) over time.
    Returns a Matplotlib FuncAnimation object.
    """
    bodies = solar_system.bodies
    star = solar_system.star
    # Use cached positions for all time steps if available
    times = solar_system.orbit_cache["times"] if hasattr(solar_system, "orbit_cache") else None
    total_frames = len(times) if times is not None else frames
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    # Initialize plot elements:
    star_point, = ax.plot([], [], 'yo', markersize=8)  # sun
    planet_points = [ax.plot([], [], 'o', color=plt.cm.tab10(i), markersize=5)[0] for i in range(len(bodies))]
    rocket_point, = ax.plot([], [], 'X', color='magenta', markersize=6)  # rocket
    # Optionally, lines for trails
    trails = [ax.plot([], [], '--', color=plt.cm.tab10(i), linewidth=0.8)[0] for i in range(len(bodies))]
    rocket_trail, = ax.plot([], [], '--', color='magenta', linewidth=0.8)

    def init():
        ax.set_xlim(-1.5, 1.5)  # Set a reasonable initial view, can be adjusted or dynamic
        ax.set_ylim(-1.5, 1.5)
        star_point.set_data([0], [0])
        for p in planet_points:
            p.set_data([], [])
        rocket_point.set_data([], [])
        for t in trails:
            t.set_data([], [])
        rocket_trail.set_data([], [])
        return [star_point, *planet_points, rocket_point, *trails, rocket_trail]

    def update(frame):
        # Update planet positions
        if times is not None:
            t_idx = frame
            positions = solar_system.orbit_cache["positions"][:, t_idx]
        else:
            # If no precomputed positions, this should compute positions at time = frame*dt
            positions = np.array([body.position for body in bodies])
            # (In a real scenario, we would update each body's position here)
        for i, p in enumerate(planet_points):
            pos = positions[i]
            p.set_data(pos[0], pos[1])          # <-- floats, so “x must be a sequence”
            rocket_point.set_data(rocket_pos[0], rocket_pos[1])
            # update trail up to this frame
            trails[i].set_data(solar_system.orbit_cache["positions"][i, :t_idx+1, 0],
                                solar_system.orbit_cache["positions"][i, :t_idx+1, 1])
        # Update rocket position if trajectory provided
        if rocket_trajectory is not None:
            rocket_pos = rocket_trajectory[frame]
            rocket_point.set_data([rocket_pos[0]], [rocket_pos[1]])
            rocket_trail.set_data(rocket_trajectory[:frame+1, 0], rocket_trajectory[:frame+1, 1])
        return [star_point, *planet_points, rocket_point, *trails, rocket_trail]

    anim = animate_orbits(solar_system,
                      rocket_trajectory=trajectory,
                      frames=len(trajectory),   # force same length
                      interval=50)

    return anim
