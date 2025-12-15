"""
This module contains:
- Helper functions (tet volume, barycentric coords, grid cell, box clamping)
- Time integration (pre_solve, post_solve)
- Physics initialization (init_physics)
"""

import taichi as ti


@ti.func
def get_tet_volume(p_indices, pos):
    """Calculate the volume of a tetrahedron given vertex indices and positions."""
    p0, p1, p2, p3 = pos[p_indices[0]], pos[p_indices[1]], pos[p_indices[2]], pos[p_indices[3]]
    return (p1 - p0).cross(p2 - p0).dot(p3 - p0) / 6.0


@ti.func
def get_barycentric_coords(p, a, b, c, d):
    """Compute barycentric coordinates of point p relative to tetrahedron (a, b, c, d)."""
    mat = ti.Matrix.cols([a - d, b - d, c - d])
    weights = ti.Vector([0.0, 0.0, 0.0])
    if abs(mat.determinant()) > 1e-9:
        weights = mat.inverse() @ (p - d)
    w4 = 1.0 - weights.sum()
    return ti.Vector([weights[0], weights[1], weights[2], w4])


@ti.func
def get_grid_cell(p, inv_cell_spacing, grid_size):
    """Convert world position to grid cell coordinates."""
    return ti.max(0, ti.min(grid_size - 1, ti.floor(p * inv_cell_spacing, ti.i32)))


@ti.func
def clamp_to_box(p, box_min, box_max):
    """Clamp a position to stay within the collision box bounds."""
    return ti.Vector([
        ti.max(box_min[0], ti.min(box_max[0], p[0])),
        ti.max(box_min[1], ti.min(box_max[1], p[1])),
        ti.max(box_min[2], ti.min(box_max[2], p[2]))
    ])


@ti.kernel
def init_physics(
    num_edges: ti.i32, num_tets: ti.i32, num_particles: ti.i32,
    edge_ids: ti.template(), edge_lengths: ti.template(),
    pos: ti.template(), tet_ids: ti.template(), rest_vol: ti.template(),
    inv_mass: ti.template(), prev_pos: ti.template(), vel: ti.template()
):
    """
    1. Compute rest lengths for all edges
    2. Compute rest volumes and inverse masses for all tetrahedra
    3. Initialize positions, previous positions, and velocities
    """
    for i in range(num_edges):
        id0, id1 = edge_ids[i, 0], edge_ids[i, 1]
        edge_lengths[i] = (pos[id0] - pos[id1]).norm()

    inv_mass.fill(0.0)
    for i in range(num_tets):
        p_indices = ti.Vector([tet_ids[i, 0], tet_ids[i, 1], tet_ids[i, 2], tet_ids[i, 3]])
        vol = get_tet_volume(p_indices, pos)
        rest_vol[i] = vol
        if vol > 0.0:
            p_inv_mass = 1.0 / (vol / 4.0)
            for j in ti.static(range(4)):
                inv_mass[p_indices[j]] += p_inv_mass

    for i in range(num_particles):
        pos[i] += ti.Vector([0.0, 1.0, 0.0])
        prev_pos[i] = pos[i]
        vel[i] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def pre_solve(
    dt: ti.f64, use_gravity: ti.i32, num_particles: ti.i32,
    gravity: ti.template(), box_min: ti.template(), box_max: ti.template(),
    pos: ti.template(), prev_pos: ti.template(), vel: ti.template(), inv_mass: ti.template()
):
    """
    1. Apply gravity to velocities
    2. Save current positions
    3. Integrate positions forward
    4. Clamp positions to box bounds
    """
    for i in range(num_particles):
        if inv_mass[i] == 0.0: continue
        if use_gravity != 0:
            vel[i] += gravity * dt
        prev_pos[i] = pos[i]
        pos[i] += vel[i] * dt
        
        # Box collision - clamp position to box bounds
        pos[i] = clamp_to_box(pos[i], box_min, box_max)


@ti.kernel
def post_solve(
    dt: ti.f64, num_particles: ti.i32,
    pos: ti.template(), prev_pos: ti.template(), vel: ti.template(), inv_mass: ti.template()
):
    for i in range(num_particles):
        if inv_mass[i] == 0.0: continue
        vel[i] = (pos[i] - prev_pos[i]) / dt
