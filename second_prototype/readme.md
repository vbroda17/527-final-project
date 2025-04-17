# Files
sun.txt: Contains the star's (sun's) parameters.
bodies.csv: A CSV listing each orbiting body (planet) with columns like:
name, radius, mass, aphelion, perihelion, mean_anomaly.
Units are consistent (e.g. radius and distance in AU, mass in Earth-masses, angle in degrees or radians). 
metadata.txt: Optional file specifying metadata like epoch and units. 
The code will default to AU, days, and Earth masses if not provided. The epoch can be used to interpret the starting mean anomalies.
orbit_cache.npz: Optional NumPy binary file used for caching precomputed orbital positions (and velocities) for each body. This can store arrays for planet positions over time to avoid recalculation on each run. For example, it might contain arrays like times, positions (shape [N_bodies, N_steps, dim]), and velocities (if needed). The framework will generate this cache if not present or if recalculation is forced.

solar_system.py – System Data Loading and Orbit Preparation
Purpose: Load planetary system data from the specified folder (by default systems/solar_system/) and prepare orbital information. This module reads the star and planet data, converts units if needed, computes orbital parameters, and (optionally) precomputes orbital positions/velocities for fast access. It also handles caching of these computations in NumPy files. Key responsibilities:
Data Loading: Read sun.txt for the central star’s mass and radius, and parse bodies.csv for planet data. Use the metadata file if present to adjust units or epoch.
Orbital Parameter Calculation: For each planet, derive orbital parameters such as semi-major axis and eccentricity from aphelion and perihelion (e.g. a = (aphelion + perihelion) / 2, e = (aphelion - perihelion) / (aphelion + perihelion)). These parameters, along with the starting mean anomaly, define the orbit.
Initial State Computation: Compute each planet’s initial position (and velocity if needed) at the start epoch. This may involve converting the mean anomaly to the true anomaly for elliptical orbits. For simplicity, one can assume the orbits lie in a common plane (2D simulation) with the Sun at the origin. Using Kepler’s laws, determine the planet’s position in Cartesian coordinates (x, y) relative to the Sun. (For example, if mean anomaly = 0 corresponds to perihelion, place the planet at distance = perihelion on the +x axis; otherwise, solve Kepler’s Equation to find the true anomaly).
Orbit Precomputation and Caching: To improve performance, the module can precompute the positions (and possibly velocities) of each planet over the simulation time span. For instance, generate positions for each day (or time step) up to the desired duration and save this to orbit_cache.npz. On future runs, if the cache file exists (and matches the requested time span/resolution), load it instead of recalculating. This caching mechanism uses NumPy’s binary format for efficiency.
Data Structures: Provide convenient structures for simulation use. For example, define a SolarSystem class that holds a list of bodies (with their parameters and possibly precomputed trajectory tables). The class can offer methods like get_position(body_name, t) or provide arrays of all positions at time t. The star (sun) can be stored separately or as a special body.

simulation.py – Time-Step Simulation of Rocket Trajectory
Purpose: Simulate the motion of the rocket (spacecraft) over time under the gravitational influence of all massive bodies (sun and planets). Planetary bodies move in their orbits (either precomputed or calculated on the fly), but they do not influence each other (only the central star affects them). The rocket is influenced by all bodies’ gravity, while the rocket’s influence on other bodies is ignored (its mass is negligible in simulation). Key responsibilities:
Rocket Representation: Define a rocket (or spacecraft) state, including its position, velocity, and possibly mass (mass is not needed for gravitational acceleration calculations since $a = GM/r^2$ is independent of the rocket’s mass, but it could be used for computing fuel usage or just for completeness).
Integration Mechanics: Update the rocket’s position and velocity over small time steps, accumulating the gravitational acceleration from all bodies at each step. Use an efficient integration method:
For simplicity, one might start with Euler integration (update velocity and position in a straightforward way each step). For better accuracy, methods like Verlet or Runge-Kutta can be implemented.
Each step:
Determine positions of all celestial bodies at the current time (from the SolarSystem module, e.g. using precomputed positions or by advancing their orbit analytically).
Compute the gravitational acceleration on the rocket: sum contributions from the sun and each planet. Each contribution is $\mathbf{a}_i = G M_i \frac{\mathbf{r}i - \mathbf{r}{rocket}}{|\mathbf{r}i - \mathbf{r}{rocket}|^3}$, where $M_i$ is the mass of body i and $\mathbf{r}_i$ its position. Summing gives total $\mathbf{a}$ on the rocket.
Update rocket velocity: $\mathbf{v} ;+=; \mathbf{a} \cdot \Delta t$.
Update rocket position: $\mathbf{r} ;+=; \mathbf{v} \cdot \Delta t$.
Advance time and repeat.
Because planets do not interact with each other or the rocket, their orbits can be updated independently (or fetched from cache) without needing to recalc mutual gravitation.
Performance Considerations: The simulation loop should be optimized since it may run for many time steps. Techniques to improve performance:
Use NumPy vectorization for computing gravitational forces from all bodies at once. For example, store all planet positions in an array and use array operations to compute the acceleration contribution of each in a single vectorized calculation (taking advantage of fast NumPy internals written in C). Utilizing NumPy can significantly speed up computations in Python​
stackoverflow.com
.
Use Numba (a JIT compiler) to compile the Python update loop to machine code. Numba can accelerate numerical loops to approach C-speed performance​
numba.pydata.org
. For instance, decorating the integration function with @njit can yield 10x-50x speedups by eliminating Python overhead.
Multiprocessing or parallelization: If multiple rocket trajectories need to be simulated (e.g., evaluating many candidate paths in a Bee Colony Optimization), run them in parallel across multiple CPU cores. Each simulation is independent, so using Python’s multiprocessing.Pool or concurrent futures to run simulations in separate processes can linearly speed up batch computations. (Within a single trajectory simulation, parallelizing the time steps is not straightforward since each step depends on the previous, but multiple simulations or possibly multiple rockets at once could be parallelized.)
If using Numba, one can also explore @njit(parallel=True) for internal parallel loops if applicable, though for a single trajectory the loop is sequential. Large vector operations (NumPy) are anyway executed in C and can utilize low-level optimizations.
Output: Record the rocket’s trajectory over time. This could be stored as arrays of positions (and velocities) at each time step for later analysis or visualization. For example, keep a list or NumPy array of rocket positions for the whole simulation duration.

visualization.py – Visualization of Orbits and Trajectories
Purpose: Provide functions to visualize the solar system orbits and the rocket’s journey, both as static plots and animations. This includes plotting planet orbits (with trails) and an animated view of the simulation over time, optionally saving as a GIF. Key features:
Static Orbit Plot: A function to plot a snapshot of the solar system with orbits and current positions. For example, plot_orbits(solar_system, rocket=None, time_index=None) could:
Draw the orbits of planets around the sun as dotted or faint lines. These orbits can be drawn using the semi-major axis and eccentricity (e.g., plotting an ellipse for each planet). If precomputed position history is available, one could also plot the path covered by each planet up to the current time (this would appear as a portion of the orbit, perhaps dotted to distinguish it).
Plot the sun at the center and each planet at its position for the given time (or at epoch if time_index is 0). Use circles or points scaled to reflect their relative sizes (the actual scale of radii vs distances will be hugely different, so for visualization the sizes can be exaggerated or fixed small markers).
If a rocket is present and rocket state is given, mark the rocket’s position. Also, draw the rocket’s trajectory trail as a dotted line behind it. For instance, if the rocket has completed some orbits or traveled a path, use the recorded trajectory points (from simulation) to draw a line or scatter plot of where it has been.
Orbit Counters: Indicate how many orbits each body has completed in the simulated timeframe. This could be shown by a label near each planet or in a legend. For example, if Mercury completed 4 orbits in 2 Earth years of simulation, show "Mercury (4 orbits)" next to its path. The number of orbits can be computed as elapsed_time / orbital_period (using each planet’s period from its semi-major axis via Kepler’s third law), rounded down to an integer. The rocket’s “orbits” around the sun (if it ends up orbiting) can be counted similarly if needed.
Aesthetic touches: use different colors for each planet’s orbit/path, use a distinct marker (star symbol) for the sun, maybe annotate planet names.
Animated Orbits: A function to produce an animation of the planetary motion and rocket trajectory over time. For example, animate_orbits(solar_system, trajectory, interval=50, frames=None) could use matplotlib’s FuncAnimation:
Update the positions of each planet and the rocket at each frame (incrementing the time index).
Draw trailing paths by plotting segments of the trajectory up to the current frame (to create a motion trail effect).
Possibly show an updating text for time (like "Day 100"), and update orbit counters dynamically (though that might be better left for static summary).
Ensure the axes are scaled appropriately (e.g., equal aspect ratio for x and y so orbits are not distorted).
Once animation is created, it can be saved as a GIF or MP4. Matplotlib can save GIFs using Pillow, or one can capture frames and use an image library. For example, animation.save('orbits_animation.gif', writer='pillow') would produce a GIF file.
Optional Visualization Tools: The module could also support alternative visualization backends:
Pygame for a more interactive or real-time simulation display (which could allow panning/zooming or user control, but requires more manual drawing code).
Plotly for interactive browser-based visualization with the ability to zoom and play through time. Plotly can create an animation by updating figure frames and can be quite powerful for 3D or interactive needs. These are optional; the default implementation can use Matplotlib for simplicity.
Saving Plots: Provide functions to save the static plots (e.g., as PNG) and the animations. E.g., save_plot(fig, filename) and save_animation(anim, filename) or simply integrate saving into the above functions via parameters.

helpers.py – Physics and Utility Functions
Purpose: Provide helper functions for physics calculations and file I/O operations to keep other modules clean. This includes computations for gravitational forces, orbital mechanics utilities, and data reading/writing routines. Functions and utilities:
Gravitational Constants and Units: Define the gravitational constant G in the chosen units. Since the simulation might use Astronomical Units, days, and Earth masses, G needs to be in those units. For instance, in units of AU, days, and Earth-masses, one can compute G ~ 2.96×10<sup>-4</sup> AU<sup>3</sup>/(Earth-mass·day<sup>2</sup>) (derived from the SI value). Alternatively, one could choose units such that $G=1$ for simplicity (for example, using appropriate normalizing units like the Sun's mass and year as in astronomical units systems). In this framework, we will define G once and use it consistently in the simulation.
Orbital Mechanics: Functions to assist with orbital calculations:
compute_orbital_elements(aphelion, perihelion) – returns semi-major axis a and eccentricity e.
mean_to_eccentric_anomaly(M, e) – solves Kepler’s equation to find the eccentric anomaly E given mean anomaly M and eccentricity e (use iterative numerical method).
true_anomaly(E, e) – computes the true anomaly θ from eccentric anomaly.
orbital_radius(a, e, theta) – gives distance from focus at true anomaly θ (using $r = \frac{a(1-e^2)}{1+e\cos\theta}$).
orbital_period(a, M_central) – returns orbital period given semi-major axis a and central mass (using Kepler’s third law).
These can be used by solar_system.py for precise position calculations instead of duplicating code in the class.
Gravitational Force/Acceleration:
gravity_acceleration(body_pos, body_mass, target_pos) – returns the acceleration vector on a target at target_pos due to a body at body_pos with mass body_mass. This would implement $\mathbf{a} = G M \frac{\mathbf{r}{body} - \mathbf{r}{target}}{|\mathbf{r}{body} - \mathbf{r}{target}|^3}$.
Potentially a variant that sums over multiple bodies: total_gravity(positions, masses, target_pos) to compute combined acceleration (this could be vectorized with NumPy).
File I/O Utilities:
read_sun_file(path) – reads the sun.txt and returns a dict or object with mass and radius (parsing the simple format).
read_bodies_file(path) – reads bodies.csv and returns a list of dicts or a structured numpy array for planet data.
read_metadata(path) – reads metadata.txt if present, returns a dict of settings.
save_orbit_cache(path, data) – saves the orbit_cache dictionary (with arrays) to an .npz file. For example, use np.savez_compressed(path, **data_dict).
load_orbit_cache(path) – loads the .npz file and returns the dictionary of arrays.
These abstract out the file handling from the logic in solar_system.py.
Unit Conversion Helpers: If supporting flexible units, provide conversion factors, e.g. AU_to_km, EarthMass_to_kg, etc., and functions to apply conversions on arrays of data. This ties in with reading metadata and converting data to internal units.

main.py – Running the Simulation and Generating Output
Purpose: The entry point of the application. It ties together all components: loading the system, running the simulation for a configured duration, and invoking visualization. This script can be executed to see the simulation results or generate outputs. Typical workflow in main.py:
Initialize Solar System Data: Create a SolarSystem object (from solar_system.py) for the desired system (default is the Solar System data in systems/solar_system). This will load planet data and prepare orbits (and possibly load or compute the orbit cache).
Configure Rocket Initial State: Define the initial conditions for the rocket. For example, you might start the rocket on Earth’s orbit. E.g., position = Earth’s position, velocity = Earth’s orbital velocity plus some delta (to simulate a spacecraft launch). For a simple test, one could start the rocket at Earth with the same velocity (so it stays in Earth’s orbit), or give it a slight boost to set it on a transfer trajectory.
Run Simulation: Create a Simulation object with the solar system and rocket, set the desired time step (e.g., 1 day or 0.1 day for finer resolution), and call run(total_time). The total simulation time might be, say, a few years in days. The result is the rocket’s trajectory (and internally the simulation has recorded it).
Output Results: After simulation, use visualization.py to produce outputs:
Call plot_orbits to generate a static figure of the orbits and positions at the end of the simulation (or at a specific snapshot). Save this figure to a file, e.g., orbits_plot.png.
Call animate_orbits with the rocket’s trajectory to create an animation of the journey. Save this as orbits_animation.gif for viewing.
Optionally, print some summary data, such as final state of the rocket, or how close it came to a target planet, etc., which might be relevant for assessing an interplanetary trajectory.
Integration with BCO: While not explicitly shown in code, the main.py (or rather the simulation module) could be used by a Bee Colony Optimization algorithm. For example, the BCO could vary the rocket’s initial velocity or departure date, run the simulation each time, and evaluate a fitness (like how close it gets to a target or fuel usage). This framework supports that by separating simulation logic (which can be called multiple times with different parameters) from the data loading (done once) and visualization (for after optimization runs).