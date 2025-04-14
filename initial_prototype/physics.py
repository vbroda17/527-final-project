import numpy as np
G = 6.67430e-11  # m^3 kg^-1 s^-2 the gravitational constant

def circle_orbit(a, w):
    """Return a function pos(t) that traces a circle of radius *a* rad/s *w*."""
    def _f(t):
        return np.array([a*np.cos(w*t), a*np.sin(w*t)])
    return _f

class Planet:
    def __init__(self, name, mass, radius_draw=0, static_pos=None, orbit_func=None):
        self.name = name
        self.mass = mass
        self.radius_draw = radius_draw
        self.static_pos = np.array(static_pos) if static_pos is not None else None
        self.orbit_func = orbit_func  # callable t→np.array([x,y])
    def pos(self, t):
        return self.static_pos if self.orbit_func is None else self.orbit_func(t)

def gravity(pos, bodies, t):
    """Return total gravitational acceleration at *pos* from *bodies* at time *t*."""
    a = np.zeros(2)
    for b in bodies:
        r_vec = b.pos(t) - pos
        r = np.linalg.norm(r_vec)
        if r == 0: continue
        a += G * b.mass * r_vec / r**3
    return a