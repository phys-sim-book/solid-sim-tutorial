# Inversion-free Hyperelastic Solids Simulation

A square falling onto a ground under gravity and then compressed by a ceiling is simulated with an inversion-free hyperelastic potential and implicit Euler time integration.
Each time step is solved by minimizing the Incremental Potential with the projected Newton method.

## Dependencies
```
pip install numpy scipy pygame
```

## Run
```
python simulator.py
```