# main.py
from solar_system import SolarSystem
from simulation import Simulation, Rocket
from visualization import plot_orbits, animate_orbits
import os

def main():
    # ensure the output folder exists
    os.makedirs("outputs", exist_ok=True)

    # Initialize solar system data
    solar_system = SolarSystem(data_folder="systems/solar_system", use_cache=True)
    
    # Setup rocket initial state (e.g., at Earth's position, heading outward)
    # Find Earth's initial position and velocity from solar_system data:
    earth = next(body for body in solar_system.bodies if body.name.lower() == "earth")
    rocket_start_pos = earth.position.copy()
    rocket_start_vel = earth.velocity.copy() if earth.velocity is not None else np.array([0.0, 0.0])
    # For example, give the rocket a slight delta-v in Earth's orbital direction to raise its orbit
    rocket_start_vel *= 1.1  # 10% faster than Earth orbital velocity
    rocket = Rocket(position=rocket_start_pos, velocity=rocket_start_vel, mass=1.0)
    
    # Run simulation for a given duration (e.g., 2 Earth years)
    sim = Simulation(solar_system, rocket, time_step=1.0)  # 1 day per step
    total_time = 2 * 365  # days
    trajectory = sim.run(total_time)
    
    # Generate visualizations
    # 1. Static plot of final orbits and positions
    fig, ax = plot_orbits(solar_system, rocket=sim.rocket, time_index=len(solar_system.orbit_cache["times"]) - 1)
    fig.savefig("outputs/orbits_plot.png")
    # 2. Animated GIF of the journey
    anim = animate_orbits(solar_system, rocket_trajectory=trajectory, interval=50)
    anim.save("outputs/orbits_animation.gif", writer='pillow')
    
    # Print final state of rocket
    final_pos = sim.rocket.position
    final_vel = sim.rocket.velocity
    print(f"Final rocket position: {final_pos}, velocity: {final_vel}")
    # Example: evaluate if rocket reached Mars proximity
    mars = next(body for body in solar_system.bodies if body.name.lower() == "mars")
    dist_to_mars = np.linalg.norm(final_pos - mars.position)
    print(f"Distance from Mars at end: {dist_to_mars:.3f} AU")

if __name__ == "__main__":
    main()
