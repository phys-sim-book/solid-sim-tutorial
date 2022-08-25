# Inversion-free Hyperelastic Solids Simulation

Two squares falling onto the ground under gravity, contacting with each other, is simulated with an inversion-free hyperelastic potential and IPC with implicit Euler time integration.
Each time step is solved by minimizing the Incremental Potential with the projected Newton method.

## Dependencies
```
pip install numpy scipy pygame
```

## Run
```
python simulator.py
```