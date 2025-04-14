import numpy as np

G = 6.67430e-11  # m^3 kg^-1 s^-2 (kept for completeness)

class Planet:
    def __init__(self, name, x, y, mass, radius=0):
        self.name = name
        self.pos = np.array([x, y], dtype=float)
        self.mass = mass
        self.radius = radius  # purely for drawing

def gravity_acc(pos, planet):
    """Return gravitational acceleration vector at 'pos' due to 'planet'."""
    r_vec = planet.pos - pos
    r = np.linalg.norm(r_vec)
    if r == 0:
        return np.zeros(2)
    return G * planet.mass * r_vec / r**3
