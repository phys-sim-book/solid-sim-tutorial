import taichi as ti


@ti.kernel
def pin_top_vertices(
    num_particles: ti.i32,
    pos: ti.template(), inv_mass: ti.template()
):
 
    max_y = -1e9
    for i in range(num_particles):
        if pos[i].y > max_y:
            max_y = pos[i].y
    for i in range(num_particles):
        if pos[i].y >= max_y - 1e-6:
            inv_mass[i] = 0.0


@ti.kernel
def initialize_colors(
    num_particles: ti.i32,
    pos: ti.template(), vertex_colors: ti.template()
):
  
    for i in range(num_particles):
        x_coord = int((pos[i].x + 1.0) * 5) % 2
        z_coord = int((pos[i].z + 1.0) * 5) % 2
        
        if (x_coord + z_coord) % 2 == 0:
            vertex_colors[i] = ti.Vector([1.0, 1.0, 1.0])
        else:
            vertex_colors[i] = ti.Vector([0.1, 0.1, 0.1])


@ti.kernel
def apply_grab(
    particle_idx: ti.i32, target_x: ti.f64, target_y: ti.f64, target_z: ti.f64,
    pos: ti.template(), vel: ti.template(), grab_indicator_pos: ti.template()
):
 
    if particle_idx != -1:
        target_pos = ti.Vector([target_x, target_y, target_z])
        pos[particle_idx] = target_pos
        vel[particle_idx] = ti.Vector([0.0, 0.0, 0.0])
        grab_indicator_pos[0] = target_pos
