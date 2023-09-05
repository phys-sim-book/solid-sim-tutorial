# Mass-Spring Solids Simulation

A square falling onto the ground under gravity is simulated with mass-spring elasticity potential and implicit Euler time integration.
Each time step is solved by minimizing the Incremental Potential with the projected Newton method.

## Dependencies
```
pip install numpy scipy pygame
```

## Run
```
python simulator.py
```