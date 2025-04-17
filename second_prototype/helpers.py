# helpers.py
import numpy as np
import csv
import re
# Define gravitational constant G in desired units (AU^3 / (Earth mass * day^2)):
# Using G = 6.67430e-11 m^3/(kg s^2),
# 1 AU = 1.4959787e11 m, 1 day = 86400 s, 1 Earth mass = 5.9722e24 kg.
# G_in_AU_day_Emass = G_SI * (day^2) * (1 / AU^3) * Earth_mass (since Earth_mass in denominator of units).
G = 6.67430e-11 * (86400**2) / ((1.4959787e11)**3) * 5.9722e24
def compute_gravitational_param(mass):
    """Return μ = G * M   (mass in the same units used for G)."""
    return G * mass

# For simplicity, one might alternatively use canonical units where G=1 and adjust masses accordingly.

def compute_orbital_elements(aphelion, perihelion):
    """Return semi-major axis a and eccentricity e from aphelion and perihelion distances."""
    a = 0.5 * (aphelion + perihelion)
    e = (aphelion - perihelion) / (aphelion + perihelion)
    return a, e

def mean_to_eccentric_anomaly(M, e, tol=1e-8):
    """Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E given mean anomaly M."""
    E = M  # initial guess
    if e < 0.001:
        return M  # nearly circular
    for _ in range(100):
        f = E - e*np.sin(E) - M
        if abs(f) < tol:
            break
        E = E - f / (1 - e*np.cos(E))
    return E

def true_anomaly(E, e):
    """Compute true anomaly from eccentric anomaly."""
    return 2 * np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))

def orbital_radius(a, e, theta):
    """Calculate orbital radius (distance from focus) at true anomaly theta."""
    return a * (1 - e**2) / (1 + e * np.cos(theta))

def orbital_period(a, M_central):
    """Calculate orbital period (in days) for semi-major axis a (AU) and central mass M_central (Earth masses)."""
    # Using Kepler's third law: T^2 = 4π^2 a^3 / (G * M_central)
    mu = G * M_central  # G*M (in AU^3/day^2)
    T = 2 * np.pi * np.sqrt(a**3 / mu)
    return T

def gravity_acceleration(body_pos, body_mass, target_pos):
    """Compute gravitational acceleration vector on target due to a body."""
    r_vec = body_pos - target_pos
    dist = np.linalg.norm(r_vec)
    if dist == 0:
        return np.zeros_like(r_vec)
    a_vec = G * body_mass * r_vec / (dist**3)
    return a_vec

def total_gravity(positions, masses, target_pos):
    """Compute total gravitational acceleration on target due to multiple bodies."""
    # Vectorized calculation: positions is an array of shape (N, dim), masses shape (N,)
    r_vecs = positions - target_pos  # shape (N, dim)
    dist_vec = np.linalg.norm(r_vecs, axis=1)
    # Avoid division by zero for any zero-distance (replace zeros with inf so those accel = 0)
    dist_cubed = np.where(dist_vec == 0, np.inf, dist_vec**3)
    # Compute acceleration for each body and sum
    acc_components = (G * masses)[:, None] * (r_vecs / dist_cubed[:, None])
    total_acc = np.nansum(acc_components, axis=0)
    return total_acc

def read_sun_file(path):
    """Read sun data file and return a dict with mass and radius."""
    data = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                # Support formats like "Mass = 332946"
                parts = line.split('=')
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    value = float(parts[1].strip().split()[0])
                    if "mass" in key:
                        data["mass"] = value
                    if "radius" in key:
                        data["radius"] = value
                else:
                    # If it's just numbers in known order (mass then radius)
                    vals = line.strip().split()
                    if len(vals) >= 2:
                        data["mass"] = float(vals[0])
                        data["radius"] = float(vals[1])
                        break
    except FileNotFoundError:
        raise
    return data

def _clean_field(s: str) -> str:
    """
    Lower‑case, strip spaces/underscores and anything in parentheses/brackets.
    Examples:
        'radius(AU)'      -> 'radius'
        'mass  [Earth]'   -> 'mass'
        ' mean_anomaly '  -> 'mean_anomaly'
    """
    s = s.strip().lower()
    # throw away everything from the first '(' or '[' onward
    s = re.split(r'[\(\[]', s)[0]
    s = s.replace(' ', '').replace('_', '')
    return s

_EXPECTED = {
    "name": "name",
    "radius": "radius",
    "mass": "mass",
    "aphelion": "aphelion",
    "perihelion": "perihelion",
    "meananomaly": "mean_anomaly",
}

def read_bodies_file(path):
    """Load bodies.csv and return a list of dicts with canonical keys."""
    bodies = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        # Build a mapping from cleaned header -> real header name
        cleaned = {_clean_field(h): h for h in reader.fieldnames}
        missing = [k for k in _EXPECTED if k not in cleaned]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        for row in reader:
            body = {
                "name":           row[cleaned["name"]],
                "radius":         float(row[cleaned["radius"]]),
                "mass":           float(row[cleaned["mass"]]),
                "aphelion":       float(row[cleaned["aphelion"]]),
                "perihelion":     float(row[cleaned["perihelion"]]),
                "mean_anomaly":   float(row[cleaned["meananomaly"]]),
            }
            bodies.append(body)
    return bodies
def read_metadata(path):
    """Read metadata file if it exists and return a dictionary of settings."""
    meta = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, val = line.strip().split('=', 1)
                    meta[key.strip()] = val.strip()
    except FileNotFoundError:
        return None
    return meta

def save_orbit_cache(path, data):
    """Save orbit cache data (e.g., positions, times) to a NumPy .npz file."""
    np.savez_compressed(path, **data)

def load_orbit_cache(path):
    """Load orbit cache from a .npz file, returning a dict of arrays."""
    npz = np.load(path, allow_pickle=True)
    data = {key: npz[key] for key in npz.files}
    return data
