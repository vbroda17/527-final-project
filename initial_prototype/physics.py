import numpy as np
G = 6.67430e-11

def circle_orbit(a, w):
    def _f(t):
        return np.array([a*np.cos(w*t), a*np.sin(w*t)])
    return _f

class Planet:
    def __init__(self, name, mass, static_pos=None, orbit_func=None, radius_draw=0):
        self.name=name; self.mass=mass; self.static_pos=np.array(static_pos) if static_pos is not None else None; self.orbit_func=orbit_func; self.radius_draw=radius_draw
    def pos(self,t):
        return self.static_pos if self.orbit_func is None else self.orbit_func(t)

def gravity(pos,bodies,t):
    a=np.zeros(2)
    for b in bodies:
        r_vec=b.pos(t)-pos; r=np.linalg.norm(r_vec)
        if r==0: continue
        a+=G*b.mass*r_vec/r**3
    return a