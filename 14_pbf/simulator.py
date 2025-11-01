
import math
import numpy as np
import taichi as ti

ti.init(arch=ti.gpu, default_fp=ti.f32)

SUBSTEPS_PER_FRAME = 2

dim = 3
screen_res = (800, 800)
screen_to_world_ratio = 20.0
boundary = (screen_res[0] / screen_to_world_ratio * 2.2,
            screen_res[1] / screen_to_world_ratio * 1.0,
            screen_res[0] / screen_to_world_ratio * 1.0)

cell_size = 1.8

cell_recpr = 1.0 / cell_size

def round_up(f, s):
    return (math.floor(f * cell_recpr / s) + 1) * s

grid_size = (round_up(boundary[0], 1), round_up(boundary[1], 1), round_up(boundary[2], 1))

num_particles = 100000
max_num_particles_per_cell = 100
max_num_neighbors = 100
time_delta = 1.0 / 30.0
epsilon = 1e-5

particle_radius = 4.0
particle_radius_in_world = particle_radius / screen_to_world_ratio

particle_color = (0.4, 0.7, 1.0)
boundary_color = (0.9, 0.7, 0.6)

h = 1.1
mass = 1.0
rho0 = 1.0
lambda_epsilon = 50.0
pbf_num_iters = 7
corr_deltaQ_coeff = 0.3
corrK = 0.001
neighbor_radius = h * 1.05
xsph_viscosity_c = 0.01
vorticity_epsilon = 0.01

poly6_factor = 315.0 / (64.0 * math.pi)
spiky_grad_factor = -45.0 / math.pi

old_positions = ti.Vector.field(dim, float, shape=num_particles)
positions = ti.Vector.field(dim, float, shape=num_particles)
velocities = ti.Vector.field(dim, float, shape=num_particles)
grid_num_particles = ti.field(int, shape=grid_size)
grid2particles = ti.field(int, shape=(*grid_size, max_num_particles_per_cell))
particle_num_neighbors = ti.field(int, shape=num_particles)
particle_neighbors = ti.field(int, shape=(num_particles, max_num_neighbors))
lambdas = ti.field(float, shape=num_particles)
position_deltas = ti.Vector.field(dim, float, shape=num_particles)
velocity_deltas_xsph = ti.Vector.field(dim, float, shape=num_particles)
wall_states = ti.Vector.field(3, float, shape=())
particle_colors = ti.Vector.field(3, float, shape=num_particles)
vorticity = ti.Vector.field(dim, float, shape=num_particles)
vorticity_force = ti.Vector.field(dim, float, shape=num_particles)
wall_moving = ti.field(bool, shape=())
simulation_running = ti.field(bool, shape=())


@ti.func
def poly6_value(s, h_):
    result = 0.0
    if 0 < s and s < h_:
        x = (h_ * h_ - s * s) / (h_ * h_ * h_)
        result = poly6_factor * x * x * x
    return result

@ti.func
def spiky_gradient(r, h_):
    result = ti.Vector([0.0, 0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h_:
        x = (h_ - r_len) / (h_ * h_ * h_)
        g_factor = spiky_grad_factor * x * x
        result = r * g_factor / r_len
    return result

@ti.func
def compute_scorr(pos_ji):
    x = poly6_value(pos_ji.norm(), h) / poly6_value(corr_deltaQ_coeff * h, h)
    x = x * x; x = x * x
    return -corrK * x

@ti.func
def get_cell(pos):
    return int(pos * cell_recpr)

@ti.func
def is_in_grid(c):
    return 0 <= c[0] < grid_size[0] and 0 <= c[1] < grid_size[1] and 0 <= c[2] < grid_size[2]

@ti.func
def confine_position_to_boundary(p):
    left_wall_x = wall_states[None][0]
    right_wall_x = wall_states[None][1]
    bmax = ti.Vector([right_wall_x, boundary[1], boundary[2]]) - particle_radius_in_world
    bmin = ti.Vector([left_wall_x, 0.0, 0.0]) + particle_radius_in_world
    for i in ti.static(range(dim)):
        if p[i] <= bmin[i]:
            p[i] = bmin[i] + epsilon * ti.random()
        elif p[i] >= bmax[i]:
            p[i] = bmax[i] - epsilon * ti.random()
    return p

@ti.kernel
def pre_solve():
    for i in positions:
        old_positions[i] = positions[i]

    for i in positions:
        g = ti.Vector([0.0, -9.8, 0.0])
        vel = velocities[i] + g * time_delta
        velocities[i] = vel
        positions[i] = positions[i] + vel * time_delta

    grid_num_particles.fill(0)
    
    for p_i in positions:
        cell = get_cell(positions[p_i])
        offs = ti.atomic_add(grid_num_particles[cell], 1)
        if offs < max_num_particles_per_cell:
            grid2particles[cell, offs] = p_i
            
    for p_i in positions:
        pos_i = positions[p_i]
        cell = get_cell(pos_i)
        nb_i = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
            cell_to_check = cell + offs
            if is_in_grid(cell_to_check):
                for j in range(grid_num_particles[cell_to_check]):
                    p_j = grid2particles[cell_to_check, j]
                    if nb_i < max_num_neighbors and p_j != p_i and (pos_i - positions[p_j]).norm() < neighbor_radius:
                        particle_neighbors[p_i, nb_i] = p_j
                        nb_i += 1
        particle_num_neighbors[p_i] = nb_i

@ti.kernel
def solve_density_constraints():
    for p_i in positions:
        pos_i = positions[p_i]
        grad_i, sum_gradient_sqr, density_constraint = ti.Vector([0.0, 0.0, 0.0]), 0.0, 0.0

        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            pos_ji = pos_i - positions[p_j]
            grad_j = spiky_gradient(pos_ji, h)
            grad_i += grad_j
            sum_gradient_sqr += grad_j.dot(grad_j)
            density_constraint += poly6_value(pos_ji.norm(), h)

        density_constraint = (mass * density_constraint / rho0) - 1.0
        sum_gradient_sqr += grad_i.dot(grad_i)
        
        denominator = (1.0 / mass) * sum_gradient_sqr + lambda_epsilon
        # lambdas[p_i] = -density_constraint / (sum_gradient_sqr + lambda_epsilon)
        lambdas[p_i] = -density_constraint / denominator if denominator > epsilon else 0.0

    for p_i in positions:
        pos_i = positions[p_i]
        lambda_i = lambdas[p_i]
        pos_delta_i = ti.Vector([0.0, 0.0, 0.0])

        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            lambda_j = lambdas[p_j]
            pos_ji = pos_i - positions[p_j]
            scorr_ij = compute_scorr(pos_ji)
            pos_delta_i += (lambda_i + lambda_j + scorr_ij) * spiky_gradient(pos_ji, h)
        
        position_deltas[p_i] = pos_delta_i / rho0

    for i in positions:
        positions[i] += position_deltas[i]

@ti.kernel
def calculate_vorticity():
    for i in positions:
        vort_i = ti.Vector([0.0, 0.0, 0.0])
        for j in range(particle_num_neighbors[i]):
            p_j = particle_neighbors[i, j]
            pos_ji = positions[i] - positions[p_j]
            
            # Relative velocity: v_ij = v_j - v_i
            vel_ij = velocities[p_j] - velocities[i]
            
            # Cross product of relative velocity and spiky gradient
            vort_i += vel_ij.cross(spiky_gradient(pos_ji, h))
            
        vorticity[i] = vort_i

@ti.kernel
def apply_vorticity_confinement():
    # 1. Calculate the corrective force `f_vorticity` for each particle
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
            eta += (vort_j_norm - vort_i_norm) * spiky_gradient(pos_ji, h)
        
        # Calculate the direction vector N = eta / |eta|
        eta_norm = eta.norm()
        if eta_norm > epsilon: # Avoid division by zero
            N = eta / eta_norm
            # Calculate the final force: f_vorticity = epsilon * (N x omega_i)
            vorticity_force[i] = vorticity_epsilon * N.cross(vorticity[i])
        else:
            vorticity_force[i] = ti.Vector([0.0, 0.0, 0.0])

    # 2. Apply the calculated force to the velocities
    for i in velocities:
        # The paper applies a force, so we integrate it over the timestep
        velocities[i] += vorticity_force[i] * time_delta

@ti.kernel
def apply_xsph_viscosity():
    # 1. Calculate velocity corrections in parallel
    for i in positions:
        vel_i = velocities[i]
        delta_v = ti.Vector([0.0, 0.0, 0.0])
        for j in range(particle_num_neighbors[i]):
            p_j = particle_neighbors[i, j]
            pos_ji = positions[i] - positions[p_j]
            vel_ij = velocities[p_j] - vel_i
            w_ij = poly6_value(pos_ji.norm(), h)
            delta_v += vel_ij * w_ij
        velocity_deltas_xsph[i] = xsph_viscosity_c * delta_v

    # 2. Apply corrections
    for i in positions:
        velocities[i] += velocity_deltas_xsph[i]

@ti.kernel
def post_solve():
    for i in positions:
        positions[i] = confine_position_to_boundary(positions[i])
        
    for i in positions:
        velocities[i] = (positions[i] - old_positions[i]) / time_delta

def substep():
    if simulation_running[None]:
        pre_solve()
        for _ in range(pbf_num_iters):
            solve_density_constraints()
        post_solve()
        
        calculate_vorticity()
        apply_vorticity_confinement()
        apply_xsph_viscosity()

@ti.kernel
def init_particles():
    padding = particle_radius_in_world
    
    for i in range(num_particles):
        x = padding + ti.random() * (boundary[0] - 2 * padding)
        y = padding + ti.random() * (boundary[1] - 2 * padding)
        z = padding + ti.random() * (boundary[2] - 2 * padding)
        
        positions[i] = ti.Vector([x, y, z])
        velocities[i] = ti.Vector([0.0, 0.0, 0.0])
    
    wall_states[None] = ti.Vector([epsilon, boundary[0] - epsilon, 0.0])

@ti.kernel
def update_particle_colors():
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

@ti.kernel
def move_wall(speed_multiplier: float):
    w = wall_states[None]
    
    if wall_moving[None] and simulation_running[None]:
        w[2] += speed_multiplier
        period = 90
        vel_strength = 5.0
        movement_time = w[2]
        if movement_time >= 2 * period:
            w[2] = 0
            movement_time = 0
        
        movement_offset = ti.sin(movement_time * 3.14159 / period) * vel_strength * time_delta * speed_multiplier
        
        w[0] += movement_offset
        w[1] -= movement_offset
        
        w[0] = ti.max(w[0], epsilon)
        w[1] = ti.min(w[1], boundary[0] - epsilon)
    else:
        w[0] = epsilon
        w[1] = boundary[0] - epsilon
        w[2] = 0
    
    wall_states[None] = w

def create_box_edges():
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

def create_original_box_edges():
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

def main():
    global SUBSTEPS_PER_FRAME
    
    particle_color = (0.4, 0.7, 1.0)
    boundary_color = (0.9, 0.7, 0.6)
    
    wall_moving[None] = False
    simulation_running[None] = False
    
    fps_counter = 0
    fps_timer = 0.0
    current_fps = 0.0
    
    init_particles()
    update_particle_colors()
    window = ti.ui.Window("PBF3D with XSPH Viscosity", screen_res, vsync=True)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    gui = window.get_gui()
    
    cam_pos = ti.Vector([20.0, 25.0, 120.0])
    cam_yaw = 0.0
    cam_pitch = 0.0
    cam_move_speed = 1.5
    cam_rot_speed = 0.02
    
    def get_forward():
        cy = math.cos(cam_yaw); sy = math.sin(cam_yaw)
        cp = math.cos(cam_pitch); sp = math.sin(cam_pitch)
        return ti.Vector([sy * cp, -sp, -cy * cp])
    
    def get_right():
        f = get_forward()
        up = ti.Vector([0.0, 1.0, 0.0])
        r = f.cross(up)
        n = r.norm()
        return r / (n if n > 1e-8 else 1.0)
    
    def get_up():
        r = get_right()
        f = get_forward()
        return r.cross(f) * -1.0


    while window.running:
        fps_counter += 1
        fps_timer += time_delta
        if fps_timer >= 1.0:
            current_fps = fps_counter / fps_timer
            fps_counter = 0
            fps_timer = 0.0
        
        gui.begin("Simulation Controls", 0.05, 0.05, 0.3, 0.2)
        
        gui.text(f"FPS: {current_fps:.1f}")
        gui.text(f"Substeps per frame: {SUBSTEPS_PER_FRAME}")
        gui.text(f"Effective simulation speed: {SUBSTEPS_PER_FRAME}x")
        gui.text(f"Wall movement: Synchronized")
        
        gui.text("Performance:")
        if gui.button("1x Speed (1 substep)"):
            SUBSTEPS_PER_FRAME = 1
        if gui.button("2x Speed (2 substeps)"):
            SUBSTEPS_PER_FRAME = 2
        if gui.button("3x Speed (3 substeps)"):
            SUBSTEPS_PER_FRAME = 3
        if gui.button("4x Speed (4 substeps)"):
            SUBSTEPS_PER_FRAME = 4
        
        if gui.button("Run Simulation"):
            simulation_running[None] = True
            wall_moving[None] = True
            
        gui.end()
        
        if simulation_running[None]:
            move_wall(SUBSTEPS_PER_FRAME)
            
            for _ in range(SUBSTEPS_PER_FRAME):
                substep()
            
            update_particle_colors()

        if window.is_pressed(ti.ui.LEFT):
            cam_yaw -= cam_rot_speed
        if window.is_pressed(ti.ui.RIGHT):
            cam_yaw += cam_rot_speed
        if window.is_pressed(ti.ui.UP):
            cam_pitch = max(cam_pitch - cam_rot_speed, -1.4)
        if window.is_pressed(ti.ui.DOWN):
            cam_pitch = min(cam_pitch + cam_rot_speed, 1.4)
        
        move = ti.Vector([0.0, 0.0, 0.0])
        if window.is_pressed('w'):
            move += get_forward()
        if window.is_pressed('s'):
            move -= get_forward()
        if window.is_pressed('a'):
            move -= get_right()
        if window.is_pressed('d'):
            move += get_right()
        if window.is_pressed('q'):
            move -= ti.Vector([0.0, 1.0, 0.0])
        if window.is_pressed('e'):
            move += ti.Vector([0.0, 1.0, 0.0])
        if move.norm() > 1e-8:
            move = move / move.norm()
        cam_pos += move * cam_move_speed
        
        cam_forward = get_forward()
        cam_target = cam_pos + cam_forward
        camera.position(cam_pos[0], cam_pos[1], cam_pos[2])
        camera.lookat(cam_target[0], cam_target[1], cam_target[2])
        
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(30, 30, 30), color=(1, 1, 1))

        scene.particles(positions, radius=particle_radius_in_world, per_vertex_color=particle_colors)
        
        orig_corners, orig_edges = create_original_box_edges()
        for i in range(12):
            start_corner = np.array(orig_corners[orig_edges[i][0]], dtype=np.float32)
            end_corner = np.array(orig_corners[orig_edges[i][1]], dtype=np.float32)
            scene.lines(np.array([start_corner, end_corner], dtype=np.float32), color=(0.6, 0.6, 0.6), width=1)

        corners, edges = create_box_edges()
        for i in range(12):
            start_corner = np.array(corners[edges[i][0]], dtype=np.float32)
            end_corner = np.array(corners[edges[i][1]], dtype=np.float32)
            scene.lines(np.array([start_corner, end_corner], dtype=np.float32), color=(0.85, 0.85, 0.85), width=2)
        
        canvas.scene(scene)
        
        window.show()

if __name__ == "__main__":
    main()