"""
This module contains: 
- Time integration (pre_solve, post_solve)
- Physics initialization (init_physics)
"""

import taichi as ti


@ti.kernel
def init_physics(
    num_particles: ti.i32, num_tris: ti.i32,
    num_stretching_constraints: ti.i32, num_bending_constraints: ti.i32,
    pos: ti.template(), prev_pos: ti.template(), vel: ti.template(),
    inv_mass: ti.template(), original_inv_mass: ti.template(),
    tri_ids: ti.template(), stretching_ids: ti.template(), bending_ids: ti.template(),
    stretching_lengths: ti.template(), bending_lengths: ti.template()
):
    """
    1. Initialize previous positions and velocities
    2. Compute inverse masses from triangle areas
    3. Compute rest lengths for stretching and bending constraints
    4. Store original inverse masses for later restoration
    """
    for i in range(num_particles):
        prev_pos[i] = pos[i]
        vel[i] = ti.Vector([0.0, 0.0, 0.0])
    
    inv_mass.fill(0.0)
    for i in range(num_tris):
        id0, id1, id2 = tri_ids[i * 3], tri_ids[i * 3 + 1], tri_ids[i * 3 + 2]
        p0, p1, p2 = pos[id0], pos[id1], pos[id2]
        area = 0.5 * (p1 - p0).cross(p2 - p0).norm()
        p_inv_mass = 1.0 / (area / 3.0) if area > 0 else 0.0
        inv_mass[id0] += p_inv_mass
        inv_mass[id1] += p_inv_mass
        inv_mass[id2] += p_inv_mass
    
    for i in range(num_stretching_constraints):
        id0, id1 = stretching_ids[i, 0], stretching_ids[i, 1]
        stretching_lengths[i] = (pos[id0] - pos[id1]).norm()
    
    for i in range(num_bending_constraints):
        id0, id1 = bending_ids[i, 0], bending_ids[i, 1]
        bending_lengths[i] = (pos[id0] - pos[id1]).norm()
    
    for i in range(num_particles):
        original_inv_mass[i] = inv_mass[i]


@ti.kernel
def pre_solve(
    dt: ti.f64, num_particles: ti.i32,
    gravity: ti.template(),
    pos: ti.template(), prev_pos: ti.template(), vel: ti.template(), inv_mass: ti.template()
):
    """
    1. Apply gravity to velocities
    2. Save current positions
    3. Integrate positions forward
    4. Handle ground collision (simple reflection at y=0)
    """
    for i in range(num_particles):
        if inv_mass[i] == 0.0: continue
        vel[i] += gravity * dt
        prev_pos[i] = pos[i]
        pos[i] += vel[i] * dt
        if pos[i].y < 0.0:
            pos[i] = prev_pos[i]
            pos[i].y = 0.0


@ti.kernel
def post_solve(
    dt: ti.f64, num_particles: ti.i32,
    pos: ti.template(), prev_pos: ti.template(), vel: ti.template(), inv_mass: ti.template()
):
    for i in range(num_particles):
        if inv_mass[i] == 0.0: continue
        vel[i] = (pos[i] - prev_pos[i]) / dt
