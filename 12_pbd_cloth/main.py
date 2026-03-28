import taichi as ti
import numpy as np
import time
import warp as wp
import warp.render

# Import all modules
from xpbd_base import init_physics, pre_solve, post_solve
from constraints import solve_stretching_constraints, solve_bending_constraints
from utils import (
    load_cloth_data_from_json, find_constraint_indices, find_bottom_corner, calculate_quadratic_path_pos,
    pin_top_vertices, apply_grab, initialize_colors
)

ti.init(arch=ti.cpu, default_fp=ti.f64)


class USDExporter:
    """USD animation exporter for Blender compatibility."""
    
    def __init__(self, output_path="cloth_animation.usd", fps=60, scale=10.0):
        self.output_path = output_path
        self.fps = fps
        self.scale = scale
        self.frame_dt = 1.0 / fps
        self.sim_time = 0.0
        self.frame_count = 0
        
        self.renderer = wp.render.UsdRenderer(output_path, scaling=scale)
        self.renderer.render_ground()
        
        print(f"USD exporter initialized: {output_path}")
        print(f"FPS: {fps}")
        print(f"Scale: {scale}x")
    
    def export_frame(self, positions, indices):
        """Export a single frame to the USD file."""
        self.renderer.begin_frame(self.sim_time)
        self.renderer.render_mesh(
            name="cloth_mesh",
            points=positions,
            indices=indices
        )
        self.renderer.end_frame()
        
        self.frame_count += 1
        self.sim_time += self.frame_dt
        
        if self.frame_count % 60 == 0:
            print(f"Exported frame {self.frame_count} at time {self.sim_time:.3f}s")
        
    def save(self):
        """Finalize and save the USD file."""
        self.renderer.save()
        print(f"\nUSD animation saved: {self.output_path}")
        print(f"Total frames: {self.frame_count}")
        print(f"Duration: {self.sim_time:.3f} seconds")
        print(f"\nTo import into Blender:")
        print(f"1. File > Import > Universal Scene Description (.usd)")
        print(f"2. Select '{self.output_path}'")
        print(f"3. The animation will be imported as a single animated sequence!")


# ============================================================================
# Load Mesh Data
# ============================================================================

verts_np, face_tri_ids_np = load_cloth_data_from_json()
tris_np = face_tri_ids_np.reshape((-1, 3))
stretching_ids_np, bending_ids_np = find_constraint_indices(tris_np)

# ============================================================================
# Simulation Constants
# ============================================================================

paused = True
gravity = ti.Vector([0.0, -9.8, 0.0])
dt = 1.0 / 60.0
num_substeps = 15
sdt = dt / num_substeps
solver_iterations = 1 # can increase

export_enabled = False
export_frame_count = 0
usd_exporter = None

num_particles = len(verts_np)
num_tris = len(tris_np)
num_stretching_constraints = len(stretching_ids_np)
num_bending_constraints = len(bending_ids_np)

stretching_compliance = 0.0
bending_compliance = 1.0

# ============================================================================
# Taichi Field Allocations
# ============================================================================

# Simulation fields
pos = ti.Vector.field(3, dtype=ti.f64, shape=num_particles)
prev_pos = ti.Vector.field(3, dtype=ti.f64, shape=num_particles)
vel = ti.Vector.field(3, dtype=ti.f64, shape=num_particles)
inv_mass = ti.field(dtype=ti.f64, shape=num_particles)
original_inv_mass = ti.field(dtype=ti.f64, shape=num_particles)

# Mesh topology fields
tri_ids = ti.field(ti.i32, shape=num_tris * 3)
stretching_ids = ti.field(ti.i32, shape=(num_stretching_constraints, 2))
bending_ids = ti.field(ti.i32, shape=(num_bending_constraints, 2))

# Constraint rest lengths
stretching_lengths = ti.field(dtype=ti.f64, shape=num_stretching_constraints)
bending_lengths = ti.field(dtype=ti.f64, shape=num_bending_constraints)

# XPBD accumulated lambdas (one per constraint, reset each substep)
stretching_lambdas = ti.field(dtype=ti.f64, shape=num_stretching_constraints)
bending_lambdas = ti.field(dtype=ti.f64, shape=num_bending_constraints)

# Visualization fields
ground_vertices = ti.Vector.field(3, dtype=ti.f64, shape=4)
ground_indices = ti.field(ti.i32, shape=6)
grab_indicator_pos = ti.Vector.field(3, dtype=ti.f64, shape=1)
vertex_colors = ti.Vector.field(3, dtype=ti.f64, shape=num_particles)

# ============================================================================
# Initialize Fields from Mesh Data
# ============================================================================

pos.from_numpy(verts_np)
tri_ids.from_numpy(tris_np.flatten())
stretching_ids.from_numpy(stretching_ids_np)
bending_ids.from_numpy(bending_ids_np)
ground_vertices.from_numpy(np.array([[-10, 0, -10], [10, 0, -10], [10, 0, 10], [-10, 0, 10]], dtype=np.float64))
ground_indices.from_numpy(np.array([0, 1, 2, 0, 2, 3], dtype=np.int32))

# ============================================================================
# Simulation Substep Function
# ============================================================================

@ti.kernel
def reset_constraint_lambdas(
    num_s: ti.i32, num_b: ti.i32,
    s_lambdas: ti.template(), b_lambdas: ti.template()
):
    for i in range(num_s):
        s_lambdas[i] = 0.0
    for i in range(num_b):
        b_lambdas[i] = 0.0


def substep(grab_id, grab_x, grab_y, grab_z):
    """Perform one simulation substep."""
    pre_solve(sdt, num_particles, gravity, pos, prev_pos, vel, inv_mass)
    # XPBD: reset lambdas once per substep (they accumulate across solver iterations)
    reset_constraint_lambdas(
        num_stretching_constraints, num_bending_constraints,
        stretching_lambdas, bending_lambdas
    )
    for _ in range(solver_iterations):
        solve_stretching_constraints(stretching_compliance, sdt, num_stretching_constraints, pos, stretching_ids, stretching_lengths, inv_mass, stretching_lambdas)
        solve_bending_constraints(bending_compliance, sdt, num_bending_constraints, pos, bending_ids, bending_lengths, inv_mass, bending_lambdas)
    apply_grab(grab_id, grab_x, grab_y, grab_z, pos, vel, grab_indicator_pos)
    post_solve(sdt, num_particles, pos, prev_pos, vel, inv_mass)


# ============================================================================
# Initialize Simulation
# ============================================================================

init_physics(num_particles, num_tris, num_stretching_constraints, num_bending_constraints,
             pos, prev_pos, vel, inv_mass, original_inv_mass,
             tri_ids, stretching_ids, bending_ids, stretching_lengths, bending_lengths)
pin_top_vertices(num_particles, pos, inv_mass)
initialize_colors(num_particles, pos, vertex_colors)

# ============================================================================
# Setup UI and Camera
# ============================================================================

window = ti.ui.Window("Taichi XPBD Cloth", (1024, 1024), vsync=True)
canvas = window.get_canvas()
scene = window.get_scene()
camera = ti.ui.Camera()
camera.position(0, 1.0, 3.0)
camera.lookat(0, 0.5, 0)

# ============================================================================
# Animation State
# ============================================================================

is_grabbing = False
grabbed_particle_id = -1
grab_distance = 0.0
bottom_corner_id = find_bottom_corner(pos.to_numpy())

is_animating = False
animation_start_time = 0.0
animation_duration = 2.0
original_corner_pos = None
target_corner_pos = None
start_time = time.time()

# ============================================================================
# Main Simulation and Rendering Loop
# ============================================================================

while window.running:
    gui = window.GUI
    with gui.sub_window("Controls", 0.05, 0.05, 0.3, 0.4):
        paused = gui.checkbox("Paused", paused)
        bending_compliance = gui.slider_float("Bending Compliance", bending_compliance, 0.0, 10.0)
        
        gui.text("Bottom Corner Control:")
        if paused:
            gui.text("(Unpause simulation to control corner)")
        else:
            if gui.button("Animate Corner"):
                if not is_animating and not is_grabbing:
                    is_animating = True
                    is_grabbing = True
                    grabbed_particle_id = bottom_corner_id
                    inv_mass[grabbed_particle_id] = 0.0
                    
                    original_corner_pos = pos[grabbed_particle_id]
                    target_corner_pos = ti.Vector([original_corner_pos.x - 0.3, original_corner_pos.y + 0.5, original_corner_pos.z + 0.2])
                    animation_start_time = 0.0
            
            if gui.button("Release Corner"):
                if is_grabbing and grabbed_particle_id != -1:
                    inv_mass[grabbed_particle_id] = original_inv_mass[grabbed_particle_id]
                is_grabbing = False
                is_animating = False
                grabbed_particle_id = -1
        
        if is_animating:
            current_time = time.time()
            elapsed_time = current_time - animation_start_time
            progress = min(1.0, elapsed_time / animation_duration)
            gui.text(f"Animation Progress: {progress:.1%}")
        
        gui.text("Export:")
        export_enabled = gui.checkbox("Export Animation", export_enabled)
        
        if export_enabled and usd_exporter is None:
            usd_exporter = USDExporter("cloth_animation.usd", fps=60, scale=10.0)
            print("USD exporter initialized!")
            
        gui.text(f"Exported Frames: {export_frame_count}")
        if usd_exporter:
            gui.text(f"USD File: {usd_exporter.output_path}")
        
        if gui.button("Reset"):
            pos.from_numpy(verts_np)
            init_physics(num_particles, num_tris, num_stretching_constraints, num_bending_constraints,
                        pos, prev_pos, vel, inv_mass, original_inv_mass,
                        tri_ids, stretching_ids, bending_ids, stretching_lengths, bending_lengths)
            pin_top_vertices(num_particles, pos, inv_mass)
            is_grabbing = False
            is_animating = False
            grabbed_particle_id = -1

    grab_target_position = ti.Vector([0.0, 0.0, 0.0])
    
    if is_animating and grabbed_particle_id != -1:
        if animation_start_time == 0.0:
            animation_start_time = time.time()
        
        current_time = time.time()
        elapsed_time = current_time - animation_start_time
        
        if elapsed_time < animation_duration:
            t = elapsed_time / animation_duration
            t = max(0.0, min(1.0, t))
            
            grab_target_position = calculate_quadratic_path_pos(original_corner_pos, target_corner_pos, t)
        else:
            grab_target_position = target_corner_pos
            is_animating = False
    
    elif is_grabbing and grabbed_particle_id != -1:
        grab_target_position = pos[grabbed_particle_id]
    
    if not paused:
        for _ in range(num_substeps):
            substep(grabbed_particle_id, grab_target_position.x, grab_target_position.y, grab_target_position.z)

    if export_enabled and usd_exporter:
        particle_positions = pos.to_numpy()
        triangle_indices = tri_ids.to_numpy()
        usd_exporter.export_frame(particle_positions, triangle_indices)
        export_frame_count += 1

    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(2, 3, 4), color=(1, 1, 1))
    
    scene.mesh(pos, indices=tri_ids, per_vertex_color=vertex_colors, two_sided=True)
    scene.mesh(ground_vertices, indices=ground_indices, color=(0.8, 0.8, 0.8))

    if is_grabbing:
        scene.particles(grab_indicator_pos, radius=0.01, color=(1.0, 1.0, 0.0))

    canvas.scene(scene)
    window.show()

if usd_exporter and export_frame_count > 0:
    usd_exporter.save()
    print(f"\nAnimation export complete!")
    print(f"Total frames exported: {export_frame_count}")
    print(f"USD file: {usd_exporter.output_path}")
