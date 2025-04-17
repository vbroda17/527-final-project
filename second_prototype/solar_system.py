# solar_system.py
import numpy as np
import csv
from helpers import read_sun_file, read_bodies_file, read_metadata, save_orbit_cache, load_orbit_cache
from helpers import G, compute_gravitational_param

class CelestialBody:
    """Class to store basic data for a celestial body (planet or star)."""
    def __init__(self, name, mass, radius, aphelion=None, perihelion=None, mean_anomaly=None):
        self.name = name
        self.mass = mass            # in Earth masses (or specified unit)
        self.radius = radius        # in AU (if distance unit is AU)
        # Orbital parameters (for planets; None for the central star)
        self.aphelion = aphelion    
        self.perihelion = perihelion
        self.mean_anomaly = mean_anomaly  # at epoch (in radians)
        if aphelion and perihelion:
            # Compute semi-major axis and eccentricity for elliptical orbit
            self.a = 0.5 * (aphelion + perihelion)
            self.e = (aphelion - perihelion) / (aphelion + perihelion)
        else:
            self.a = self.e = None

        # Placeholder for dynamic state (position, velocity) if needed
        self.position = None
        self.velocity = None

class SolarSystem:
    def __init__(self, data_folder="systems/solar_system", use_cache=True):
        self.data_folder = data_folder
        self.bodies = []      # list of CelestialBody objects for planets
        self.star = None      # CelestialBody for the sun/star
        self.epoch = None
        self.units = {"distance": "AU", "time": "day", "mass": "EarthMass"}  # default units
        # Load data and prepare orbits:
        self._load_system_data()
        if use_cache:
            loaded = self._load_cached_orbits()
        else:
            loaded = False
        if not loaded:
            self._compute_initial_states()
            # Optionally precompute orbital trajectories for a default duration if needed
            # (The actual duration and step could be determined by simulation needs)
            # e.g., self.precompute_orbits(total_days=365*5, step=1.0)
            # Save to cache for future use
            self._save_orbit_cache()

    def _load_system_data(self):
        """Read sun file, bodies file, and metadata (if exists)."""
        # Read star data
        star_data = read_sun_file(f"{self.data_folder}/sun.txt")
        # Example: star_data could be dict with keys 'mass' and 'radius'
        self.star = CelestialBody(name="Sun", mass=star_data["mass"], radius=star_data["radius"])
        # Read optional metadata (units, epoch)
        meta = read_metadata(f"{self.data_folder}/metadata.txt")
        if meta:
            self.epoch = meta.get("epoch", None)
            # Update units if provided (e.g., use different distance unit and adjust values accordingly)
            self._apply_unit_conversions(meta)
        # Read planets data from CSV
        bodies_data = read_bodies_file(f"{self.data_folder}/bodies.csv")
        for body in bodies_data:
            name = body["name"]
            mass = body["mass"]
            radius = body["radius"]
            ap = body["aphelion"]
            per = body["perihelion"]
            anomaly = body["mean_anomaly"]
            # Convert anomaly to radians for internal use if it’s in degrees
            mean_anomaly_rad = np.deg2rad(anomaly) if meta and meta.get("angle_unit","deg") == "deg" else anomaly
            planet = CelestialBody(name, mass, radius, ap, per, mean_anomaly_rad)
            self.bodies.append(planet)

    def _apply_unit_conversions(self, meta):
        """Convert loaded values to internal base units (AU, day, EarthMass)."""
        # For example, if metadata specifies distance in km, convert all distances to AU.
        # (1 AU ~ 149,597,870.7 km). Similarly for mass (if given in SolarMass, convert to EarthMass, etc.)
        # Pseudocode:
        if meta.get("distance_unit") and meta["distance_unit"] != "AU":
            factor = ...  # determine conversion factor to AU
            self.star.radius *= factor
            # Also convert each body's radius, aphelion, perihelion
            for body in self.bodies:
                body.radius *= factor
                body.aphelion *= factor
                body.perihelion *= factor
        if meta.get("mass_unit") and meta["mass_unit"] != "EarthMass":
            m_factor = ...  # conversion to Earth masses
            self.star.mass *= m_factor
            for body in self.bodies:
                body.mass *= m_factor
        # time_unit can be handled in simulation (affects gravitational constant, etc.)
        self.units.update({k: meta[k] for k in ["distance_unit","time_unit","mass_unit"] if k in meta})

    def _compute_initial_states(self):
        """Compute initial position (and velocity) for each planet at epoch."""
        for planet in self.bodies:
            # Calculate initial position in 2D plane.
            # Assume orbit lies in XY-plane with Sun at (0,0).
            # Determine true anomaly from mean anomaly (solve Kepler's equation if needed for high accuracy).
            M = planet.mean_anomaly
            e = planet.e
            a = planet.a
            if e and e < 1e-6:
                # nearly circular, treat true anomaly ~ mean anomaly for simplicity
                true_anom = M
            elif e:
                # Solve Kepler's equation for E (eccentric anomaly) via iteration (Newton's method)
                E = M if e < 0.8 else np.pi  # initial guess
                for _ in range(100):
                    dE = (M - (E - e*np.sin(E))) / (1 - e*np.cos(E))
                    E += dE
                    if abs(dE) < 1e-8:
                        break
                true_anom = 2 * np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
            else:
                true_anom = M  # If circular or e not defined (for sun)
            # Distance from focus at true anomaly (for ellipse: r = a*(1 - e^2) / (1 + e*cos(theta)))
            r = planet.a * (1 - planet.e**2) / (1 + planet.e * np.cos(true_anom)) if planet.e is not None else 0
            # Compute coordinates (x, y)
            x = r * np.cos(true_anom)
            y = r * np.sin(true_anom)
            planet.position = np.array([x, y])
            # Compute orbital velocity vector for planet (if needed for integration):
            # For elliptical orbit, orbital speed v = sqrt(G*M_sun*(2/r - 1/a)).
            # Direction is perpendicular to radius vector at perihelion.
            # (For simplicity, assume initial velocity is perpendicular to position vector.)
            if planet.e is not None:
                mu = compute_gravitational_param(self.star.mass)  # G*M_sun in appropriate units
                speed = np.sqrt(mu * (2/r - 1/planet.a))
                # Velocity direction: perpendicular to radius (90 degrees ahead of true anomaly for prograde orbit)
                vx = -speed * np.sin(true_anom)
                vy =  speed * np.cos(true_anom)
                planet.velocity = np.array([vx, vy])
        # (After this, each planet has initial position and velocity set)

    def precompute_orbits(self, total_time_days, time_step=1.0):
        """Precompute positions (and velocities) for each planet at each time step up to total_time."""
        num_steps = int(total_time_days / time_step) + 1
        times = np.linspace(0, total_time_days, num_steps)  # time 0 to total_time
        # Initialize arrays for positions
        all_positions = np.zeros((len(self.bodies), num_steps, 2))
        # (If velocity tracking needed: all_velocities = np.zeros((len(self.bodies), num_steps, 2)))
        for i, planet in enumerate(self.bodies):
            # Using orbital elements to compute position at each time
            if planet.e is not None:
                # mean motion n = 2π / T (T can be derived from a^3 ~ (M_sun) and time unit)
                mu = compute_gravitational_param(self.star.mass)
                # Kepler's third law: T^2 = 4π^2 * a^3 / mu  -> T in days
                period = 2*np.pi * np.sqrt(planet.a**3 / mu)
                n = 2 * np.pi / period  # mean motion (rad/day)
                M0 = planet.mean_anomaly
                for j, t in enumerate(times):
                    M = M0 + n * t
                    # solve for true anomaly similarly as above
                    # ... (compute E, then true_anom, then r, then x,y)
                    # assign to all_positions[i, j, :] = [x, y]
            else:
                # If no orbital data (e.g., star), just skip or set position [0,0].
                all_positions[i, :, :] = 0
        self.orbit_cache = {"times": times, "positions": all_positions}
        # (One could also store velocities if needed)

    def _load_cached_orbits(self):
        """Attempt to load precomputed orbit positions from file."""
        cache_path = f"{self.data_folder}/orbit_cache.npz"
        try:
            data = load_orbit_cache(cache_path)
            if data:
                # Assume cache contains 'positions' and possibly 'times'
                self.orbit_cache = {"times": data["times"], "positions": data["positions"]}
                return True
        except FileNotFoundError:
            return False
        return False

    def _save_orbit_cache(self):
        """Save the precomputed orbits to a cache file for faster reload next time."""
        if hasattr(self, "orbit_cache"):
            save_orbit_cache(f"{self.data_folder}/orbit_cache.npz", self.orbit_cache)

    def get_body_position(self, name, time_index):
        """Retrieve position of a given body at a specific time index (from precomputed cache)."""
        if not hasattr(self, "orbit_cache"):
            raise RuntimeError("Orbit cache not computed. Call precompute_orbits first.")
        # Find index of body in list
        for i, body in enumerate(self.bodies):
            if body.name.lower() == name.lower():
                return self.orbit_cache["positions"][i, time_index]
        return None
