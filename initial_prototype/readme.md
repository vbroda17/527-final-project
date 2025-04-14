# Some information about this 
Optemizes low fule, short time trajectories

Uses a 2d sandbox

Each bee careies a rocket that preforms a sswquence of thrust impulses

Grativity is calculated from all celesital bodies

Runs saved like run_YYYYMMDD-HHMMSS

Initialisation – every bee starts exactly at the Start body’s coordinates with zero velocity.  A random list of thrust impulses (way‑points) is generated.

Physics – For each segment we integrate Newton’s law with Euler steps:

v  += (thrust_vec + Σ_i G m_i (r_i−p)/|…|^3) * dt
p  += v * dt

Fuel ∝ |thrust_vec|·dt.  Time is simulated time.
3.  CostJ = α·flight_time  +  β·fuel_used  (default α = 1, β = 0.01).
4.  ABC phases – employed, on‑looker, scout.
5.  Output – best trajectory + convergence plots saved to a fresh run folder.


# Things to improve
Automatic run naming system
fixing how the planet system works
- Starting with where it actually starts at
- Haaving like a system of planets. All of which have some kinda of gravity
- Picking one as a start planet
- picking one as a target planet
- The rest will be just normal, kinda sever as obsicalls, but this should not need to be specified as all should have gravity impacting orbits

- color code paths to start like blue then become red slowly at end point to show like time being passed ( can be any color)

- probably ignore the runs, have a special folder to save the best runs that might be used in like report or poster

- IMporve this to explain whats happening in scripts better