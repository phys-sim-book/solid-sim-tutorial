"""
This module contains:
- Smoothing kernels (poly6, spiky_gradient)
- Grid hashing and neighbor search
- Boundary collision handling
- Time integration (pre_solve, post_solve)
"""

import taichi as ti


@ti.func
def poly6_value(s, h_, poly6_factor_):
    """Poly6 smoothing kernel for density estimation."""
    result = 0.0
    if 0 < s and s < h_:
        x = (h_ * h_ - s * s) / (h_ * h_ * h_)
        result = poly6_factor_ * x * x * x
    return result


@ti.func
def spiky_gradient(r, h_, spiky_grad_factor_):
    """Spiky gradient kernel for pressure forces."""
    result = ti.Vector([0.0, 0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h_:
        x = (h_ - r_len) / (h_ * h_ * h_)
        g_factor = spiky_grad_factor_ * x * x
        result = r * g_factor / r_len
    return result


@ti.func
def get_cell(pos, cell_recpr_):
    """Convert world position to grid cell coordinates."""
    return int(pos * cell_recpr_)


@ti.func
def is_in_grid(c, grid_size_x: int, grid_size_y: int, grid_size_z: int):
    """Check if a grid cell coordinate is within bounds."""
    return 0 <= c[0] < grid_size_x and 0 <= c[1] < grid_size_y and 0 <= c[2] < grid_size_z


@ti.func
def confine_position_to_boundary(p, wall_states_, boundary_x: float, boundary_y: float, boundary_z: float, particle_radius_in_world_, epsilon_):
    """Confine particle position to simulation boundary with moving walls."""
    left_wall_x = wall_states_[None][0]
    right_wall_x = wall_states_[None][1]
    bmax = ti.Vector([right_wall_x, boundary_y, boundary_z]) - particle_radius_in_world_
    bmin = ti.Vector([left_wall_x, 0.0, 0.0]) + particle_radius_in_world_
    for i in ti.static(range(3)):
        if p[i] <= bmin[i]:
            p[i] = bmin[i] + epsilon_ * ti.random()
        elif p[i] >= bmax[i]:
            p[i] = bmax[i] - epsilon_ * ti.random()
    return p


@ti.kernel
def pre_solve(
    old_positions: ti.template(),
    positions: ti.template(),
    velocities: ti.template(),
    grid_num_particles: ti.template(),
    grid2particles: ti.template(),
    particle_num_neighbors: ti.template(),
    particle_neighbors: ti.template(),
    time_delta_: float,
    cell_recpr_: float,
    grid_size_x: int,
    grid_size_y: int,
    grid_size_z: int,
    max_num_particles_per_cell_: int,
    max_num_neighbors_: int,
    neighbor_radius_: float
):
    """
    Pre-solve step: time integration and neighbor search.
    
    Steps:
    1. Save old positions
    2. Apply gravity and update positions/velocities
    3. Build spatial hash grid
    4. Find neighbors for each particle
    """
    # Save old positions
    for i in positions:
        old_positions[i] = positions[i]

    # Apply gravity and integrate
    for i in positions:
        g = ti.Vector([0.0, -9.8, 0.0])
        vel = velocities[i] + g * time_delta_
        velocities[i] = vel
        positions[i] = positions[i] + vel * time_delta_

    # Clear and rebuild spatial hash grid
    grid_num_particles.fill(0)
    
    for p_i in positions:
        cell = get_cell(positions[p_i], cell_recpr_)
        offs = ti.atomic_add(grid_num_particles[cell], 1)
        if offs < max_num_particles_per_cell_:
            grid2particles[cell, offs] = p_i
            
    # Find neighbors for each particle
    for p_i in positions:
        pos_i = positions[p_i]
        cell = get_cell(pos_i, cell_recpr_)
        nb_i = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
            cell_to_check = cell + offs
            if is_in_grid(cell_to_check, grid_size_x, grid_size_y, grid_size_z):
                for j in range(grid_num_particles[cell_to_check]):
                    p_j = grid2particles[cell_to_check, j]
                    if nb_i < max_num_neighbors_ and p_j != p_i and (pos_i - positions[p_j]).norm() < neighbor_radius_:
                        particle_neighbors[p_i, nb_i] = p_j
                        nb_i += 1
        particle_num_neighbors[p_i] = nb_i


@ti.kernel
def post_solve(
    positions: ti.template(),
    old_positions: ti.template(),
    velocities: ti.template(),
    wall_states: ti.template(),
    boundary_x: float,
    boundary_y: float,
    boundary_z: float,
    particle_radius_in_world_: float,
    epsilon_: float,
    time_delta_: float
):
    """
    Post-solve step: boundary collision and velocity update.
    
    Steps:
    1. Confine particles to boundary
    2. Update velocities based on position changes
    """
    for i in positions:
        positions[i] = confine_position_to_boundary(positions[i], wall_states, boundary_x, boundary_y, boundary_z, particle_radius_in_world_, epsilon_)
        
    for i in positions:
        velocities[i] = (positions[i] - old_positions[i]) / time_delta_
