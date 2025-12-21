"""
Vorticity Confinement Module

This module restores energy lost to numerical damping by detecting and amplifying
vorticity (rotational motion) in the fluid. This keeps the simulation visually
interesting and physically plausible.
"""

import taichi as ti
import sys
import os
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from sph_base import spiky_gradient


@ti.kernel
def calculate_vorticity(
    positions: ti.template(),
    velocities: ti.template(),
    particle_num_neighbors: ti.template(),
    particle_neighbors: ti.template(),
    vorticity: ti.template(),
    h_: float,
    spiky_grad_factor_: float
):

    for i in positions:
        vort_i = ti.Vector([0.0, 0.0, 0.0])
        for j in range(particle_num_neighbors[i]):
            p_j = particle_neighbors[i, j]
            pos_ji = positions[i] - positions[p_j]
            
            # Relative velocity: v_ij = v_j - v_i
            vel_ij = velocities[p_j] - velocities[i]
            
            # Cross product of relative velocity and spiky gradient
            vort_i += vel_ij.cross(spiky_gradient(pos_ji, h_, spiky_grad_factor_))
            
        vorticity[i] = vort_i


@ti.kernel
def apply_vorticity_confinement(
    positions: ti.template(),
    velocities: ti.template(),
    particle_num_neighbors: ti.template(),
    particle_neighbors: ti.template(),
    vorticity: ti.template(),
    vorticity_force: ti.template(),
    h_: float,
    spiky_grad_factor_: float,
    epsilon_: float,
    vorticity_epsilon_: float,
    time_delta_: float
):

    # Step 1: Calculate the corrective force `f_vorticity` for each particle
    for i in positions:
        # First, calculate eta = \nabla|\omega|
        # This is the gradient of the magnitude of the vorticity field.
        eta = ti.Vector([0.0, 0.0, 0.0])
        vort_i_norm = vorticity[i].norm()

        for j in range(particle_num_neighbors[i]):
            p_j = particle_neighbors[i, j]
            pos_ji = positions[i] - positions[p_j]
            
            vort_j_norm = vorticity[p_j].norm()
            
            # Difference in vorticity magnitude multiplied by the kernel gradient
            eta += (vort_j_norm - vort_i_norm) * spiky_gradient(pos_ji, h_, spiky_grad_factor_)
        
        # Calculate the direction vector N = eta / |eta|
        eta_norm = eta.norm()
        if eta_norm > epsilon_: # Avoid division by zero
            N = eta / eta_norm
            # Calculate the final force: f_vorticity = epsilon * (N x omega_i)
            vorticity_force[i] = vorticity_epsilon_ * N.cross(vorticity[i])
        else:
            vorticity_force[i] = ti.Vector([0.0, 0.0, 0.0])

    # Step 2: Apply the calculated force to the velocities
    for i in velocities:
        # The paper applies a force, so we integrate it over the timestep
        velocities[i] += vorticity_force[i] * time_delta_
