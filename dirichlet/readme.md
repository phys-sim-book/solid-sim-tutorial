# Mass-Spring Solids Simulation

A square hanging under gravity with its right and left top nodes fixed is simulated with mass-spring elasticity potential and implicit Euler time integration.
Each time step is solved by minimizing the Incremental Potential with the projected Newton method.

## Dependencies
```
pip install numpy scipy pygame
```

## Run
```
python simulator.py
```