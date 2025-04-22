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

# add calculations being saved to like npy files