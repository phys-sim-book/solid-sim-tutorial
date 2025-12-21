"""
Viscosity Module - XSPH Viscosity

This module implements XSPH (eXtended Smoothed Particle Hydrodynamics) viscosity,
which smooths particle velocities by making neighboring particles move more coherently.
This reduces numerical noise and creates more stable, visually pleasing simulations.
"""

import taichi as ti
import sys
import os
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from sph_base import poly6_value


@ti.kernel
def apply_xsph_viscosity(
    positions: ti.template(),
    velocities: ti.template(),
    particle_num_neighbors: ti.template(),
    particle_neighbors: ti.template(),
    velocity_deltas_xsph: ti.template(),
    h_: float,
    poly6_factor_: float,
    xsph_viscosity_c_: float
):
    # Step 1: Calculate velocity corrections in parallel
    for i in positions:
        vel_i = velocities[i]
        delta_v = ti.Vector([0.0, 0.0, 0.0])
        for j in range(particle_num_neighbors[i]):
            p_j = particle_neighbors[i, j]
            pos_ji = positions[i] - positions[p_j]
            vel_ij = velocities[p_j] - vel_i
            w_ij = poly6_value(pos_ji.norm(), h_, poly6_factor_)
            delta_v += vel_ij * w_ij
        velocity_deltas_xsph[i] = xsph_viscosity_c_ * delta_v

    # Step 2: Apply corrections
    for i in positions:
        velocities[i] += velocity_deltas_xsph[i]
