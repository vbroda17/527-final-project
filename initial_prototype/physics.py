import numpy as np

G = 6.67430e-11  # gravitational constant (m^3 kg^-1 s^-2)

class Planet:
    def __init__(self, name: str, x: float, y: float, mass: float, radius: float = 0):
        self.name = name
        self.pos = np.array([x, y], dtype=float)
        self.mass = mass
        self.radius = radius  # for drawing only

def gravity_acc(pos: np.ndarray, planet: "Planet") -> np.ndarray:
    """Return gravitational acceleration at *pos* due to *planet*."""
    r_vec = planet.pos - pos
    r = np.linalg.norm(r_vec)
    if r == 0:
        return np.zeros(2)
    return G * planet.mass * r_vec / r ** 3