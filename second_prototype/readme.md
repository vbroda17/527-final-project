# Orginization
There are two main compenets right now. There is a solar_system.py script that does all of the general planet calcultions, including orbiits and gravity pulls. There is a folder called rocket that does a lot of the rocket components. This includes initialization, vizulization, a sim, and a demo(s).
# Solar System
This reads in from a given csv file and system information. Adjustable but right now only have up to mars.

# Rocket
## Init
Does physics per ticks
Computes velocity, adds the gravity, and a bit more 
## Sim
Kinda works as an orchestration. Has table of positions and such (ephemeris), loops given steps and calls the controller and updates planets
## Viz
Creates the vizulizattions
## Demo
Used to test out diferent controllers and to try and fine tune the rocket. Runs the sims


# FOR POSTER AND FUTURE WORK
1. Variables: Speed of bees, number of bees (inlcluding role split), Food source, trail decay
2. Implement Bee Colony Opt
3. Make the variables
    A. Implement trail decay
    B. Implement Food sources
    C. Implement Fitness function for food source

## Obj
fintess have entire orbit as food source, calculate actuall fintess based on real location of the planet
Random food as like astroids appear on and near orbit as well, more static value
Try without graviity on all bees, then consider trying with later

# Consider add calculations being saved to like npy files
Need to fix best path. It is being found in there but not displaying on best path properly.

# General info about bee
## Class
Speed conversion from km/h → AU/day.

Planet sim: spin up a RocketSim to get the planets’ precomputed trajectories.

Distance threshold

Compute the straight‐line distance D from start to target.

Set a path‐length threshold at 1.05*D. Any path longer than that is penalized.

Orbit weights

Sample M points around the destination planet’s orbit.

Compute each point’s angular difference Δ from the true‐planet position, then so that the point exactly at the planet has weight 1, and the opposite side of the circle has weight 0.

Bonus patches

Always include the true‐planet index.

Randomly pick up to extra_food additional orbit indices.

Those points each get the same angular weight as above (so the planet spot is the highest).

Initialize the bees into three buckets:

Employed (search around good spots),

Onlookers (follow employed proportionally to their fitness),

Scouts (random wanderers that “abandon” a bad patch and relocate).
## Step bee function
Each time a bee steps:

Move by its current velocity, optionally add gravity.

Accumulate the incremental path length.

Angular proximity:
Find the orbit point whose distance to the bee is minimal → index i_near.
Look up its w_orbit = orbit_weights[i_near].

Path-length bonus:
Let L be its cumulative path length so far, and thresh = 1.05·D.
Define frac = (thresh − L)/thresh.
If L ≤ thresh, bonus = 1 + frac (so at exactly L=0, bonus=2; at L=thresh, bonus=1).
If L>thresh, then bonus = frac<0, penalizing overly long routes.

Combine
b.fit = w_orbit * path_bonus

Patch bonus
If within 0.02 AU of any bonus point → add that point’s angular weight to b.fit.

Record the new position into b.path.

## TLDR
Putting it all together
Angular proximity drives bees toward the portion of the orbit closest to the actual planet.

Path‐length bonus penalizes overly long, winding routes.

Patch bonuses give extra spikes of fitness at random orbit spots (so the colony can explore “shortcuts”).

Role dynamics (employed ↔ onlooker ↔ scout) implement the standard ABC flow—employed probe good spots, onlookers converge on the best of those, scouts randomly re-seed.


# TO DO TOMMOROW
Fix the best path. Make it stop earlier
change the param search to do it all with no command line arguments
check how the graphs look