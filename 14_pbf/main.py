import math
import numpy as np
import taichi as ti

# Import all modules
from sph_base import pre_solve, post_solve
from constraints import solve_density_constraints, calculate_vorticity, apply_vorticity_confinement, apply_xsph_viscosity
from scene import init_particles, move_wall, update_particle_colors, create_box_edges, create_original_box_edges

ti.init(arch=ti.gpu, default_fp=ti.f32)

# ============================================================================
# Simulation Constants
# ============================================================================

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

# SPH parameters
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

# ============================================================================
# Taichi Field Allocations
# ============================================================================

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


# ============================================================================
# Simulation Loop
# ============================================================================

def substep():
    if simulation_running[None]:
        # Pre-solve: time integration and neighbor search
        pre_solve(
            old_positions, positions, velocities, grid_num_particles, grid2particles,
            particle_num_neighbors, particle_neighbors,
            time_delta, cell_recpr, grid_size[0], grid_size[1], grid_size[2],
            max_num_particles_per_cell, max_num_neighbors, neighbor_radius
        )
        
        # Solve density constraints (iterated for stability)
        for _ in range(pbf_num_iters):
            solve_density_constraints(
                positions, particle_num_neighbors, particle_neighbors, lambdas, position_deltas,
                h, poly6_factor, spiky_grad_factor, mass, rho0, lambda_epsilon, epsilon,
                corr_deltaQ_coeff, corrK
            )
        
        # Post-solve: boundary collision and velocity update
        post_solve(
            positions, old_positions, velocities, wall_states,
            boundary[0], boundary[1], boundary[2],
            particle_radius_in_world, epsilon, time_delta
        )
        
        # Vorticity confinement: restore lost rotational energy
        calculate_vorticity(
            positions, velocities, particle_num_neighbors, particle_neighbors, vorticity,
            h, spiky_grad_factor
        )
        apply_vorticity_confinement(
            positions, velocities, particle_num_neighbors, particle_neighbors,
            vorticity, vorticity_force, h, spiky_grad_factor, epsilon, vorticity_epsilon, time_delta
        )
        
        # XSPH viscosity: smooth velocities
        apply_xsph_viscosity(
            positions, velocities, particle_num_neighbors, particle_neighbors, velocity_deltas_xsph,
            h, poly6_factor, xsph_viscosity_c
        )


# ============================================================================
# Main Application
# ============================================================================

def main():
    global SUBSTEPS_PER_FRAME
    
    wall_moving[None] = False
    simulation_running[None] = False
    
    fps_counter = 0
    fps_timer = 0.0
    current_fps = 0.0
    
    # Initialize particles and visualization
    init_particles(positions, velocities, wall_states, num_particles, particle_radius_in_world, boundary[0], boundary[1], boundary[2], epsilon)
    update_particle_colors(positions, velocities, particle_colors)
    
    # Setup window and rendering
    window = ti.ui.Window("PBF3D with XSPH Viscosity", screen_res, vsync=True)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    gui = window.get_gui()
    
    # Camera setup
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

    # Main loop
    while window.running:
        fps_counter += 1
        fps_timer += time_delta
        if fps_timer >= 1.0:
            current_fps = fps_counter / fps_timer
            fps_counter = 0
            fps_timer = 0.0
        
        # GUI
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
        
        # Simulation step
        if simulation_running[None]:
            move_wall(wall_states, wall_moving, simulation_running, SUBSTEPS_PER_FRAME, time_delta, boundary[0], epsilon)
            
            for _ in range(SUBSTEPS_PER_FRAME):
                substep()
            
            update_particle_colors(positions, velocities, particle_colors)

        # Camera controls
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
        
        # Update camera
        cam_forward = get_forward()
        cam_target = cam_pos + cam_forward
        camera.position(cam_pos[0], cam_pos[1], cam_pos[2])
        camera.lookat(cam_target[0], cam_target[1], cam_target[2])
        
        # Render scene
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(30, 30, 30), color=(1, 1, 1))

        scene.particles(positions, radius=particle_radius_in_world, per_vertex_color=particle_colors)
        
        # Render bounding boxes
        orig_corners, orig_edges = create_original_box_edges(boundary, epsilon)
        for i in range(12):
            start_corner = np.array(orig_corners[orig_edges[i][0]], dtype=np.float32)
            end_corner = np.array(orig_corners[orig_edges[i][1]], dtype=np.float32)
            scene.lines(np.array([start_corner, end_corner], dtype=np.float32), color=(0.6, 0.6, 0.6), width=1)

        corners, edges = create_box_edges(wall_states, boundary)
        for i in range(12):
            start_corner = np.array(corners[edges[i][0]], dtype=np.float32)
            end_corner = np.array(corners[edges[i][1]], dtype=np.float32)
            scene.lines(np.array([start_corner, end_corner], dtype=np.float32), color=(0.85, 0.85, 0.85), width=2)
        
        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
