import taichi as ti


@ti.kernel
def init_particles(
    positions: ti.template(),
    velocities: ti.template(),
    wall_states: ti.template(),
    num_particles_: int,
    particle_radius_in_world_: float,
    boundary_x: float,
    boundary_y: float,
    boundary_z: float,
    epsilon_: float
):
    padding = particle_radius_in_world_
    
    for i in range(num_particles_):
        x = padding + ti.random() * (boundary_x - 2 * padding)
        y = padding + ti.random() * (boundary_y - 2 * padding)
        z = padding + ti.random() * (boundary_z - 2 * padding)
        
        positions[i] = ti.Vector([x, y, z])
        velocities[i] = ti.Vector([0.0, 0.0, 0.0])
    
    # Initialize wall states: [left_wall_x, right_wall_x, movement_time]
    wall_states[None] = ti.Vector([epsilon_, boundary_x - epsilon_, 0.0])


@ti.kernel
def move_wall(
    wall_states: ti.template(),
    wall_moving: ti.template(),
    simulation_running: ti.template(),
    speed_multiplier: float,
    time_delta_: float,
    boundary_x: float,
    epsilon_: float
):
    w = wall_states[None]
    
    if wall_moving[None] and simulation_running[None]:
        w[2] += speed_multiplier
        period = 90
        vel_strength = 5.0
        movement_time = w[2]
        if movement_time >= 2 * period:
            w[2] = 0
            movement_time = 0
        
        movement_offset = ti.sin(movement_time * 3.14159 / period) * vel_strength * time_delta_ * speed_multiplier
        
        w[0] += movement_offset
        w[1] -= movement_offset
        
        w[0] = ti.max(w[0], epsilon_)
        w[1] = ti.min(w[1], boundary_x - epsilon_)
    else:
        w[0] = epsilon_
        w[1] = boundary_x - epsilon_
        w[2] = 0
    
    wall_states[None] = w


@ti.kernel
def update_particle_colors(
    positions: ti.template(),
    velocities: ti.template(),
    particle_colors: ti.template()
):
    max_velocity = 10.0

    blue = ti.Vector([0.0941, 0.2784, 0.6314])
    teal = ti.Vector([0.1392, 0.7226, 0.5441])
    yellow = ti.Vector([0.9922, 0.9216, 0.0431])
    red = ti.Vector([0.9804, 0.1961, 0.0196])

    for i in positions:
        vel_magnitude = velocities[i].norm()
        normalized_vel = ti.min(vel_magnitude / max_velocity, 1.0)

        if normalized_vel < 0.35:
            t = normalized_vel / 0.35
            particle_colors[i] = blue + t * (teal - blue)
        elif normalized_vel < 0.65:
            t = (normalized_vel - 0.35) / 0.30
            particle_colors[i] = teal + t * (yellow - teal)
        elif normalized_vel < 0.85:
            t = (normalized_vel - 0.65) / 0.20
            particle_colors[i] = yellow + t * (red - yellow)
        else:
            particle_colors[i] = red


def create_box_edges(wall_states, boundary):
    left_wall_x = wall_states[None][0]
    right_wall_x = wall_states[None][1]
    
    corners = [
        [left_wall_x, 0.0, 0.0],
        [right_wall_x, 0.0, 0.0],
        [right_wall_x, boundary[1], 0.0],
        [left_wall_x, boundary[1], 0.0],
        [left_wall_x, 0.0, boundary[2]],
        [right_wall_x, 0.0, boundary[2]],
        [right_wall_x, boundary[1], boundary[2]],
        [left_wall_x, boundary[1], boundary[2]]
    ]
    
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    
    return corners, edges


def create_original_box_edges(boundary, epsilon):
    left_wall_x = epsilon
    right_wall_x = boundary[0] - epsilon
    
    corners = [
        [left_wall_x, 0.0, 0.0],
        [right_wall_x, 0.0, 0.0],
        [right_wall_x, boundary[1], 0.0],
        [left_wall_x, boundary[1], 0.0],
        [left_wall_x, 0.0, boundary[2]],
        [right_wall_x, 0.0, boundary[2]],
        [right_wall_x, boundary[1], boundary[2]],
        [left_wall_x, boundary[1], boundary[2]]
    ]
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    return corners, edges
