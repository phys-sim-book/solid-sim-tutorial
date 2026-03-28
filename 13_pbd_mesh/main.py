"""
Main Orchestrator - XPBD Mesh Simulation

This module ties everything together:
- Allocates all Taichi fields (memory)
- Sets simulation constants (time step, compliance, etc.)
- Runs the substep() loop which calls functions from the modules
- Handles the GUI window, camera, and user input
- Manages USD export for animation
"""

import taichi as ti
import numpy as np
import json
import time
import warp as wp
import warp.render

# Import all modules
from xpbd_base import init_physics, pre_solve, post_solve
from constraints import solve_edges, solve_volumes
from skinning import build_hash_grid, compute_skinning_info_hashed, update_vis_mesh

ti.init(arch=ti.cpu, default_fp=ti.f64)


def load_dual_mesh_from_json(filepath):
    """Load simulation and visual mesh data from JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading or parsing {filepath}: {e}")
        exit()

    sim_verts_np = np.array(data['verts'], dtype=np.float64).reshape((-1, 3))
    tet_ids_np = np.array(data['tetIds'], dtype=np.int32).reshape((-1, 4))
    edge_ids_np = np.array(data['edgeIds'], dtype=np.int32).reshape((-1, 2))

    vis_data = data.get('dragonVis')
    if not vis_data:
        print("Error: JSON file must contain a 'dragonVis' object for the visual mesh.")
        exit()
    vis_verts_np = np.array(vis_data['verts'], dtype=np.float64).reshape((-1, 3))
    vis_tri_ids_np = np.array(vis_data['triIds'], dtype=np.int32)
    
    return {
        "sim_verts": sim_verts_np, "tet_ids": tet_ids_np, "edge_ids": edge_ids_np,
        "vis_verts": vis_verts_np, "vis_tri_ids": vis_tri_ids_np
    }


class USDExporter:
    """USD animation exporter for Blender compatibility."""
    
    def __init__(self, output_path="dragon_animation.usd", fps=60, scale=1.0):
        self.output_path = output_path
        self.fps = fps
        self.scale = scale
        self.frame_dt = 1.0 / fps
        self.sim_time = 0.0
        self.frame_count = 0
        
        # Initialize Warp USD renderer with scaling
        self.renderer = wp.render.UsdRenderer(output_path, scaling=scale)
        self.renderer.render_ground()
        
        print(f"USD exporter initialized: {output_path}")
        print(f"FPS: {fps}")
        print(f"Scale: {scale}x")
    
    def export_frame(self, vertices, faces):
        """Export a single frame to the USD file."""
        self.renderer.begin_frame(self.sim_time)
        self.renderer.render_mesh(
            name="dragon_mesh",
            points=vertices,
            indices=faces.flatten()
        )
        self.renderer.end_frame()
        
        self.frame_count += 1
        self.sim_time += self.frame_dt
        
        # Print progress every 60 frames
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

mesh = load_dual_mesh_from_json("dragon_data.json")

# ============================================================================
# Simulation Constants
# ============================================================================

paused = True
gravity = ti.Vector([0.0, -9.8, 0.0])
dt = 1.0 / 60.0
num_substeps = 10
solver_iterations = 5
sdt = dt / num_substeps

num_particles = len(mesh["sim_verts"])
num_tets = len(mesh["tet_ids"])
num_edges = len(mesh["edge_ids"])
num_vis_verts = len(mesh["vis_verts"])
num_vis_tris = len(mesh["vis_tri_ids"]) // 3

edge_compliance = 0.0
vol_compliance = 0.0
edge_compliance_slider = 0.0
vol_id_order_np = np.array([[1, 3, 2], [0, 2, 3], [0, 3, 1], [0, 1, 2]], dtype=np.int32)
vol_id_order = ti.field(ti.i32, shape=(4, 3))
vol_id_order.from_numpy(vol_id_order_np)

export_enabled = False
export_frame_count = 0
usd_exporter = None

CELL_SPACING = 0.05
inv_cell_spacing = 1.0 / CELL_SPACING
GRID_SIZE = 128

BOX_SIZE = 5.0
box_min = ti.Vector([-BOX_SIZE/2, 0.0, -BOX_SIZE/2])
box_max = ti.Vector([BOX_SIZE/2, BOX_SIZE, BOX_SIZE/2])

# ============================================================================
# Taichi Field Allocations
# ============================================================================

# Simulation mesh fields
pos = ti.Vector.field(3, dtype=ti.f64, shape=num_particles)
prev_pos = ti.Vector.field(3, dtype=ti.f64, shape=num_particles)
vel = ti.Vector.field(3, dtype=ti.f64, shape=num_particles)
inv_mass = ti.field(dtype=ti.f64, shape=num_particles)
rest_vol = ti.field(dtype=ti.f64, shape=num_tets)
edge_lengths = ti.field(dtype=ti.f64, shape=num_edges)
tet_ids = ti.field(ti.i32, shape=(num_tets, 4))
edge_ids = ti.field(ti.i32, shape=(num_edges, 2))

# XPBD accumulated lambdas (one per constraint, reset each substep)
edge_lambdas = ti.field(dtype=ti.f64, shape=num_edges)
vol_lambdas = ti.field(dtype=ti.f64, shape=num_tets)

# Visual mesh fields
vis_mesh_rest_pos = ti.Vector.field(3, dtype=ti.f64, shape=num_vis_verts)
vis_mesh_pos = ti.Vector.field(3, dtype=ti.f64, shape=num_vis_verts)
vis_mesh_indices = ti.field(ti.i32, shape=num_vis_tris * 3)
vis_mesh_colors = ti.Vector.field(3, dtype=ti.f32, shape=num_vis_verts)

# Skinning fields
skinning_info_tet_idx = ti.field(ti.i32, shape=num_vis_verts)
skinning_info_bary_weights = ti.Vector.field(3, dtype=ti.f64, shape=num_vis_verts)
min_dist = ti.field(ti.f64, shape=num_vis_verts)

# Spatial hash grid fields
grid_cell_counts = ti.field(ti.i32, shape=(GRID_SIZE, GRID_SIZE, GRID_SIZE))
grid_cell_starts = ti.field(ti.i32, shape=(GRID_SIZE, GRID_SIZE, GRID_SIZE))
sorted_vis_vert_ids = ti.field(ti.i32, shape=num_vis_verts)

# Ground plane fields
ground_vertices = ti.Vector.field(3, dtype=ti.f64, shape=4)
ground_indices = ti.field(ti.i32, shape=6)
ground_colors = ti.Vector.field(3, dtype=ti.f32, shape=4)

# ============================================================================
# Initialize Fields from Mesh Data
# ============================================================================

pos.from_numpy(mesh["sim_verts"])
tet_ids.from_numpy(mesh["tet_ids"])
edge_ids.from_numpy(mesh["edge_ids"])
vis_mesh_rest_pos.from_numpy(mesh["vis_verts"])
vis_mesh_indices.from_numpy(mesh["vis_tri_ids"])
vis_mesh_colors.fill(ti.Vector([0.968, 0.541, 0.113], dt=ti.f32))
ground_vertices.from_numpy(np.array([[-BOX_SIZE/2,0,-BOX_SIZE/2],[BOX_SIZE/2,0,-BOX_SIZE/2],[BOX_SIZE/2,0,BOX_SIZE/2],[-BOX_SIZE/2,0,BOX_SIZE/2]], dtype=np.float64))
ground_indices.from_numpy(np.array([0,1,2,0,2,3], dtype=np.int32))
ground_colors.fill(ti.Vector([0.8, 0.8, 0.8], dt=ti.f32))


# ============================================================================
# Simulation Substep Function
# ============================================================================

@ti.kernel
def reset_constraint_lambdas(
    num_e: ti.i32, num_v: ti.i32,
    e_lambdas: ti.template(), v_lambdas: ti.template()
):
    for i in range(num_e):
        e_lambdas[i] = 0.0
    for i in range(num_v):
        v_lambdas[i] = 0.0


def substep():
    pre_solve(sdt, 1, num_particles, gravity, box_min, box_max, pos, prev_pos, vel, inv_mass)
    # XPBD: reset lambdas once per substep (they accumulate across solver iterations)
    reset_constraint_lambdas(num_edges, num_tets, edge_lambdas, vol_lambdas)
    for _ in range(solver_iterations):
        solve_edges(edge_compliance, sdt, num_edges, pos, edge_ids, edge_lengths, inv_mass, edge_lambdas)
        solve_volumes(vol_compliance, sdt, num_tets, pos, tet_ids, rest_vol, inv_mass, vol_id_order, vol_lambdas)
    post_solve(sdt, num_particles, pos, prev_pos, vel, inv_mass)


# ============================================================================
# Initialize Simulation
# ============================================================================

print("Building spatial hash grid for skinning...")
start_time = time.time()
build_hash_grid(num_vis_verts, GRID_SIZE, inv_cell_spacing, vis_mesh_rest_pos, grid_cell_counts, grid_cell_starts, sorted_vis_vert_ids)
end_time = time.time()
print(f"Hash grid built in {end_time - start_time:.4f} seconds.")

print("Computing skinning information with hash grid...")
start_time = time.time()
compute_skinning_info_hashed(num_tets, num_vis_verts, CELL_SPACING, inv_cell_spacing, GRID_SIZE, pos, tet_ids, vis_mesh_rest_pos, grid_cell_counts, grid_cell_starts, sorted_vis_vert_ids, min_dist, skinning_info_tet_idx, skinning_info_bary_weights)
end_time = time.time()
print(f"Skinning computation finished in {end_time - start_time:.4f} seconds.")

init_physics(num_edges, num_tets, num_particles, edge_ids, edge_lengths, pos, tet_ids, rest_vol, inv_mass, prev_pos, vel)

# ============================================================================
# Setup UI and Camera
# ============================================================================

window = ti.ui.Window("Taichi XPBD - Dragon Simulation", (1024, 1024), vsync=True)
canvas = window.get_canvas()
scene = window.get_scene()
camera = ti.ui.Camera()
camera.position(0, 1.5, 4.5)
camera.lookat(0, 0.5, 0)

# ============================================================================
# Main Simulation and Rendering Loop
# ============================================================================

while window.running:
    gui = window.GUI
    with gui.sub_window("Controls", 0.05, 0.05, 0.3, 0.25):
        paused = gui.checkbox("Paused", paused)
        
        gui.text("Animation Export:")
        export_enabled = gui.checkbox("Export Animation", export_enabled)
        
        if export_enabled and usd_exporter is None:
            usd_exporter = USDExporter("dragon_animation.usd", fps=60, scale=1.0)
            print("USD exporter initialized!")
        
        edge_compliance_slider = gui.slider_float("Edge Compliance", edge_compliance_slider, 0.0, 50.0)
        edge_compliance = edge_compliance_slider
        vol_compliance = gui.slider_float("Volume Compliance", vol_compliance, 0.0, 10.0)
        
        gui.text(f"Exported Frames: {export_frame_count}")
        if usd_exporter:
            gui.text(f"USD File: {usd_exporter.output_path}")
        
        if gui.button("Reset"):
            pos.from_numpy(mesh["sim_verts"])
            init_physics(num_edges, num_tets, num_particles, edge_ids, edge_lengths, pos, tet_ids, rest_vol, inv_mass, prev_pos, vel)
            export_frame_count = 0
            if usd_exporter:
                usd_exporter = None

    if not paused:
        for _ in range(num_substeps):
            substep()

    # Update visual mesh before rendering
    update_vis_mesh(num_vis_verts, pos, tet_ids, vis_mesh_rest_pos, vis_mesh_pos, skinning_info_tet_idx, skinning_info_bary_weights)
    
    # Export mesh frame if enabled
    if export_enabled and not paused and usd_exporter:
        vertices = vis_mesh_pos.to_numpy()
        faces = vis_mesh_indices.to_numpy().reshape((-1, 3))
        usd_exporter.export_frame(vertices, faces)
        export_frame_count += 1

    # Rendering
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(2, 3, 4), color=(1, 1, 1))
    
    scene.mesh(vis_mesh_pos, indices=vis_mesh_indices, per_vertex_color=vis_mesh_colors)
    scene.mesh(ground_vertices, indices=ground_indices, per_vertex_color=ground_colors)
    canvas.set_background_color((1.0, 1.0, 1.0))  # white

    canvas.scene(scene)
    window.show()

# Save USD file when simulation ends
if usd_exporter and export_frame_count > 0:
    usd_exporter.save()
    print(f"\nAnimation export complete!")
    print(f"Total frames exported: {export_frame_count}")
    print(f"USD file: {usd_exporter.output_path}")
    print(f"\nTo import into Blender:")
    print(f"1. File > Import > Universal Scene Description (.usd)")
    print(f"2. Select '{usd_exporter.output_path}'")
    print(f"3. The animation will be imported as a single animated sequence!")
