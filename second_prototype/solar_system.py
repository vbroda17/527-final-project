#!/usr/bin/env python3
"""
solar_system.py  –  static overview *and* animated orbits with progress bar.

CLI
----
python solar_system.py [--ellipse | --circular] [--years N] [--dt DAYS]
                       [--system PATH] [--static]

    --ellipse / --circular   choose orbit model   (default elliptical)
    --years    N             simulate this many Earth years (default 5)
    --dt       DAYS          time step in days    (default 1)
    --static                 skip the animation, only build static PNG
    --system   PATH          data folder (default systems/solar_system)

Requires: numpy, matplotlib, tqdm  (concurrent.futures is in std‑lib)
"""

from __future__ import annotations
import argparse, csv, math, os, sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed   # parallel

# ──────────────────────────────────────────────────────────────────────────────
#   Unit conversion helpers (same as before)
# ──────────────────────────────────────────────────────────────────────────────
AU_KM          = 1.495_978_707e8          # km
EARTH_MASS_KG  = 5.9722e24
SOLAR_MASS_EM  = 332_946.0
DEG2RAD        = math.pi / 180.0

def dist_to_au(val: float, unit: str) -> float:
    u = unit.lower()
    if u.startswith("au"):  return val
    if u in ("km", "kilometre", "kilometer"): return val / AU_KM
    if u == "m": return val / (AU_KM * 1e3)
    raise ValueError(f"unknown distance unit '{unit}'")

def mass_to_earth(val: float, unit: str) -> float:
    u = unit.lower()
    if "earth" in u: return val
    if "solar" in u or "sun" in u: return val * SOLAR_MASS_EM
    if u == "kg": return val / EARTH_MASS_KG
    raise ValueError(f"unknown mass unit '{unit}'")

def angle_to_rad(val: float, unit: str) -> float:
    u = unit.lower()
    if u.startswith("rad"): return val
    if u.startswith("deg"): return val * DEG2RAD
    raise ValueError(f"unknown angle unit '{unit}'")

# ──────────────────────────────────────────────────────────────────────────────
#   File IO
# ──────────────────────────────────────────────────────────────────────────────
def read_simple_kv(path: Path) -> Dict[str,str]:
    d={}
    with path.open() as f:
        for ln in f:
            ln=ln.split("#")[0].strip()
            if "=" in ln:
                k,v=[x.strip() for x in ln.split("=",1)]
                d[k.lower()]=v
    return d

def load_sun(folder: Path, meta: Dict[str,str]):
    d=read_simple_kv(folder/"sun.txt")
    mass_u   = d.get("mass_unit",   meta.get("mass_unit","EarthMass"))
    radius_u = d.get("radius_unit", meta.get("distance_unit","AU"))
    return {
        "mass":   mass_to_earth(float(d["mass"]),   mass_u),
        "radius": dist_to_au   (float(d["radius"]), radius_u),
    }

def load_bodies(folder: Path, meta: Dict[str,str]) -> List[Dict]:
    bodies=[]
    with (folder/"bodies.csv").open(newline="") as f:
        rdr=csv.DictReader(f)
        for r in rdr:
            radius_u = meta.get("radius_unit",   meta.get("distance_unit","AU"))
            mass_u   = meta.get("mass_unit","EarthMass")
            ap_u     = meta.get("aphelion_unit", meta.get("distance_unit","AU"))
            per_u    = meta.get("perihelion_unit",meta.get("distance_unit","AU"))
            ang_u    = meta.get("mean_anomaly_unit",meta.get("angle_unit","deg"))
            bodies.append({
                "name":       r["name"].strip(),
                "radius":     dist_to_au(float(r["radius"]), radius_u),
                "mass":       mass_to_earth(float(r["mass"]), mass_u),
                "aphelion":   dist_to_au(float(r["aphelion"]), ap_u),
                "perihelion": dist_to_au(float(r["perihelion"]), per_u),
                "M0":         angle_to_rad(float(r["mean_anomaly"]), ang_u)
            })
    return bodies

# ──────────────────────────────────────────────────────────────────────────────
#   Orbit maths
# ──────────────────────────────────────────────────────────────────────────────
G_AU3_EM_day2 = 6.67430e-11 * (86400**2) / (AU_KM*1000)**3 * EARTH_MASS_KG  # AU^3 / (EM * day^2)

def elements(body:Dict):
    a = 0.5*(body["aphelion"]+body["perihelion"])
    e = (body["aphelion"]-body["perihelion"])/(body["aphelion"]+body["perihelion"])
    return a,e

def mean_motion(a, M_central):
    return 2*math.pi/period(a,M_central)

def period(a, M_central):
    return 2*math.pi*math.sqrt(a**3/(G_AU3_EM_day2*M_central))

def kepler_E(M,e, tol=1e-10):
    E=M if e<0.8 else math.pi
    for _ in range(100):
        d=(E-e*math.sin(E)-M)/(1-e*math.cos(E))
        E-=d
        if abs(d)<tol:break
    return E

def true_from_E(E,e):
    return 2*math.atan2(math.sqrt(1+e)*math.sin(E/2),
                        math.sqrt(1-e)*math.cos(E/2))

def position_at_time(body, t, M_central):
    a,e=elements(body)
    n = mean_motion(a, M_central)
    M = body["M0"] + n*t
    E = kepler_E(M% (2*math.pi), e)
    theta = true_from_E(E,e)
    r = a*(1-e**2)/(1+e*math.cos(theta))
    return np.array([r*math.cos(theta), r*math.sin(theta)])

def position_circle(body,t,M_central):
    a,_=elements(body)
    T=period(a,M_central)
    theta=body["M0"]+2*math.pi*t/T
    return np.array([a*math.cos(theta),a*math.sin(theta)])

# ──────────────────────────────────────────────────────────────────────────────
#   Trajectory pre‑compute (parallel with progress bar)
# ──────────────────────────────────────────────────────────────────────────────
def compute_one(body, t_array, M_central, elliptical):
    if elliptical:
        return np.stack([position_at_time(body,t,M_central) for t in t_array])
    return np.stack([position_circle(body,t,M_central) for t in t_array])

def precompute(bodies,total_days,dt, M_central, elliptical):
    t_arr=np.arange(0,total_days+dt,dt)
    traj=np.zeros((len(bodies),len(t_arr),2))
    with ProcessPoolExecutor() as ex:
        futures={ex.submit(compute_one,b,t_arr,M_central,elliptical):i
                 for i,b in enumerate(bodies)}
        for fut in tqdm(as_completed(futures),
                        total=len(futures),
                        desc="computing orbits",
                        unit="planet"):
            i=futures[fut]
            traj[i]=fut.result()
    return t_arr,traj

# ──────────────────────────────────────────────────────────────────────────────
#   Plotting & animation
# ──────────────────────────────────────────────────────────────────────────────
def static_overview(ax, bodies, traj0, elliptical):
    cmap   = plt.cm.plasma
    colors = cmap(np.linspace(0.15, 0.95, len(bodies)))

    lim = 0
    for (body, start_xy, col) in zip(bodies, traj0, colors):
        a, e = elements(body)
        if elliptical:
            th = np.linspace(0, 2 * math.pi, 400)
            r  = a * (1 - e ** 2) / (1 + e * np.cos(th))
            ax.plot(r * np.cos(th), r * np.sin(th), "--", color=col, alpha=0.45)
        else:
            circ = plt.Circle((0, 0), a, fill=False, ls="--", color=col, alpha=0.45)
            ax.add_patch(circ)

        # checkpoint dot (hollow) sitting at the true start position
        ax.scatter(*start_xy, s=90, facecolors="none",
                   edgecolors=col, linewidths=1.3, zorder=5)
        lim = max(lim, body["aphelion"])

    ax.scatter(0, 0, color="gold", s=120, zorder=6, label="Sun")
    ax.set_aspect("equal")
    ax.set_xlabel("AU"); ax.set_ylabel("AU")
    ax.set_xlim(-1.1 * lim, 1.1 * lim); ax.set_ylim(-1.1 * lim, 1.1 * lim)

def animate(bodies, t_arr, traj, elliptical, save=False):
    """
    Increment counter exactly when the planet passes its start‐point.
    Counter *always* starts at 0.
    """
    import math, numpy as np
    cmap   = plt.cm.plasma
    colors = cmap(np.linspace(0.15, 0.95, len(bodies)))

    start_xy = traj[:, 0]                               # (N,2) first position
    theta0   = np.arctan2(start_xy[:, 1], start_xy[:, 0])
    prev_phi = np.zeros(len(bodies))                    # previous wrapped φ
    counter  = np.zeros(len(bodies), dtype=int)         # integer orbits

    # ── Figure and static backdrop ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="k")
    fig.patch.set_facecolor("k");  ax.set_facecolor("k")
    static_overview(ax, bodies, start_xy, elliptical)   # shows hollow ring

    movers  = [ax.plot([], [], "o", color=c, zorder=4)[0] for c in colors]
    trails  = [ax.plot([], [], "-", color=c, lw=1)[0]      for c in colors]
    labels  = [ax.text(xy[0], xy[1], "0", color="w", fontsize=8,
                       ha="center", va="center", zorder=6)
               for xy in start_xy]

    # ── animator helpers ────────────────────────────────────────────────
    def init():
        for m, t in zip(movers, trails):
            m.set_data([], [])
            t.set_data([], [])
        return movers + trails + labels

    def update(frame):
        for i, (mov, trl, lbl) in enumerate(zip(movers, trails, labels)):
            x, y = traj[i, frame]
            mov.set_data([x], [y])
            trl.set_data(traj[i, :frame + 1, 0], traj[i, :frame + 1, 1])

            # current polar angle, then wrapped φ in [0, 2π)
            theta = math.atan2(y, x)
            phi   = (theta - theta0[i]) % (2 * math.pi)

            # detect wrap‑around (passed the start point)
            if phi < prev_phi[i] - 1e-6:       # crossed from high (~6.28) to low (~0)
                counter[i] += 1                # completed an orbit
                lbl.set_text(str(counter[i]))  # update number

            prev_phi[i] = phi                  # store for next frame
        return movers + trails + labels

    ani = FuncAnimation(fig, update, frames=len(t_arr),
                        init_func=init, interval=30, blit=True)

    if save:
        ani.save("orbit_anim.gif", writer="pillow", fps=30)
        print("GIF saved to orbit_anim.gif")

    plt.show()

# ──────────────────────────────────────────────────────────────────────────────
#   Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap=argparse.ArgumentParser(epilog="example: python solar_system.py --ellipse --years 10")
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument('--ellipse',dest='ellip',action='store_true')
    mode.add_argument('--circular',dest='ellip',action='store_false')
    ap.set_defaults(ellip=True)
    ap.add_argument('--years',type=float,default=5,help="simulation length in Earth years")
    ap.add_argument('--dt',type=float,default=1,help="time step (days)")
    ap.add_argument('--system',default='systems/solar_system',help="data folder")
    ap.add_argument('--static',action='store_true',help="only build static PNG")
    args=ap.parse_args()

    folder=Path(args.system)
    if not folder.exists(): sys.exit(f"folder {folder} not found")

    meta=read_simple_kv(folder/"metadata.txt")
    meta.setdefault("distance_unit","AU")
    meta.setdefault("mass_unit","EarthMass")
    meta.setdefault("angle_unit","deg")

    sun=load_sun(folder,meta)
    bodies=load_bodies(folder,meta)

    # 1. static figure always
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 6))
    static_overview(ax, bodies, 
                    traj0=np.array([position_at_time(b, 0, 1) if args.ellip else position_circle(b, 0, 1) for b in bodies]),
                    elliptical=args.ellip)

    fig.savefig("orbit_overview.png",dpi=200,bbox_inches='tight')
    print("Static plot saved to orbit_overview.png")
    if args.static:
        return

    # 2. pre‑compute trajectories
    total_days=args.years*365
    t_arr,traj=precompute(bodies,total_days,args.dt,
                          M_central=sun["mass"],elliptical=args.ellip)

    # 3. animate
    animate(bodies,t_arr,traj,args.ellip,save=True)

if __name__=="__main__":
    main()
