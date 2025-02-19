# Reduced Simulation of Neo-Hookean Solids 

A square falling onto the ground under gravity is simulated using neo-Hookean elasticity and implicit Euler time integration in the reduced solution space constructed via polynomial functions or modal-order reduction.
Each time step is solved by minimizing the Incremental Potential with the projected Newton method.

## Dependencies
```
pip install numpy scipy pygame
```

## Run
```
python simulator.py
```