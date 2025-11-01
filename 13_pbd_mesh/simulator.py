import taichi as ti
import numpy as np
import json
import time
import os
from tqdm import tqdm
import warp as wp
import warp.render

ti.init(arch=ti.cpu, default_fp=ti.f64)

def load_dual_mesh_from_json(filepath):
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
        # Begin frame
        self.renderer.begin_frame(self.sim_time)
        
        # Export dragon mesh (simplified, no colors for better Blender compatibility)
        self.renderer.render_mesh(
            name="dragon_mesh",
            points=vertices,
            indices=faces.flatten()
        )
        
        # End frame
        self.renderer.end_frame()
        
        self.frame_count += 1
        self.sim_time += self.frame_dt
        
        # Print progress every 60 frames
        if self.frame_count % 60 == 0:
            print(f"Exported frame {self.frame_count} at time {self.sim_time:.3f}s")
    
    def save(self):
        self.renderer.save()
        print(f"\nUSD animation saved: {self.output_path}")
        print(f"Total frames: {self.frame_count}")
        print(f"Duration: {self.sim_time:.3f} seconds")
        print(f"\nTo import into Blender:")
        print(f"1. File > Import > Universal Scene Description (.usd)")
        print(f"2. Select '{self.output_path}'")
        print(f"3. The animation will be imported as a single animated sequence!")

mesh = load_dual_mesh_from_json("dragon_data.json")

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
vol_id_order = [[1, 3, 2], [0, 2, 3], [0, 3, 1], [0, 1, 2]]

export_enabled = False
export_frame_count = 0
usd_exporter = None

CELL_SPACING = 0.05
inv_cell_spacing = 1.0 / CELL_SPACING
GRID_SIZE = 128

pos = ti.Vector.field(3, dtype=ti.f64, shape=num_particles)
prev_pos = ti.Vector.field(3, dtype=ti.f64, shape=num_particles)
vel = ti.Vector.field(3, dtype=ti.f64, shape=num_particles)
inv_mass = ti.field(dtype=ti.f64, shape=num_particles)
rest_vol = ti.field(dtype=ti.f64, shape=num_tets)
edge_lengths = ti.field(dtype=ti.f64, shape=num_edges)
tet_ids = ti.field(ti.i32, shape=(num_tets, 4))
edge_ids = ti.field(ti.i32, shape=(num_edges, 2))

vis_mesh_rest_pos = ti.Vector.field(3, dtype=ti.f64, shape=num_vis_verts)
vis_mesh_pos = ti.Vector.field(3, dtype=ti.f64, shape=num_vis_verts)
vis_mesh_indices = ti.field(ti.i32, shape=num_vis_tris * 3)
vis_mesh_colors = ti.Vector.field(3, dtype=ti.f32, shape=num_vis_verts)

skinning_info_tet_idx = ti.field(ti.i32, shape=num_vis_verts)
skinning_info_bary_weights = ti.Vector.field(3, dtype=ti.f64, shape=num_vis_verts)
min_dist = ti.field(ti.f64, shape=num_vis_verts)

grid_cell_counts = ti.field(ti.i32, shape=(GRID_SIZE, GRID_SIZE, GRID_SIZE))
grid_cell_starts = ti.field(ti.i32, shape=(GRID_SIZE, GRID_SIZE, GRID_SIZE))
sorted_vis_vert_ids = ti.field(ti.i32, shape=num_vis_verts)

ground_vertices = ti.Vector.field(3, dtype=ti.f64, shape=4)
ground_indices = ti.field(ti.i32, shape=6)
ground_colors = ti.Vector.field(3, dtype=ti.f32, shape=4)

BOX_SIZE = 5.0
box_min = ti.Vector([-BOX_SIZE/2, 0.0, -BOX_SIZE/2])
box_max = ti.Vector([BOX_SIZE/2, BOX_SIZE, BOX_SIZE/2])

pos.from_numpy(mesh["sim_verts"])
tet_ids.from_numpy(mesh["tet_ids"])
edge_ids.from_numpy(mesh["edge_ids"])
vis_mesh_rest_pos.from_numpy(mesh["vis_verts"])
vis_mesh_indices.from_numpy(mesh["vis_tri_ids"])
vis_mesh_colors.fill(ti.Vector([0.968, 0.541, 0.113], dt=ti.f32))
ground_vertices.from_numpy(np.array([[-BOX_SIZE/2,0,-BOX_SIZE/2],[BOX_SIZE/2,0,-BOX_SIZE/2],[BOX_SIZE/2,0,BOX_SIZE/2],[-BOX_SIZE/2,0,BOX_SIZE/2]], dtype=np.float64))
ground_indices.from_numpy(np.array([0,1,2,0,2,3], dtype=np.int32))
ground_colors.fill(ti.Vector([0.8, 0.8, 0.8], dt=ti.f32))



@ti.func
def get_tet_volume(p_indices):
    p0, p1, p2, p3 = pos[p_indices[0]], pos[p_indices[1]], pos[p_indices[2]], pos[p_indices[3]]
    return (p1 - p0).cross(p2 - p0).dot(p3 - p0) / 6.0

@ti.func
def get_barycentric_coords(p, a, b, c, d):
    mat = ti.Matrix.cols([a - d, b - d, c - d])
    weights = ti.Vector([0.0, 0.0, 0.0])
    if abs(mat.determinant()) > 1e-9:
        weights = mat.inverse() @ (p - d)
    w4 = 1.0 - weights.sum()
    return ti.Vector([weights[0], weights[1], weights[2], w4])

@ti.func
def get_grid_cell(p):
    return ti.max(0, ti.min(GRID_SIZE - 1, ti.floor(p * inv_cell_spacing, ti.i32)))

@ti.func
def clamp_to_box(p):
    """Clamp a position to stay within the collision box bounds."""
    return ti.Vector([
        ti.max(box_min[0], ti.min(box_max[0], p[0])),
        ti.max(box_min[1], ti.min(box_max[1], p[1])),
        ti.max(box_min[2], ti.min(box_max[2], p[2]))
    ])

@ti.kernel
def init_physics():
    for i in range(num_edges):
        id0, id1 = edge_ids[i, 0], edge_ids[i, 1]
        edge_lengths[i] = (pos[id0] - pos[id1]).norm()

    inv_mass.fill(0.0)
    for i in range(num_tets):
        p_indices = ti.Vector([tet_ids[i, 0], tet_ids[i, 1], tet_ids[i, 2], tet_ids[i, 3]])
        vol = get_tet_volume(p_indices)
        rest_vol[i] = vol
        if vol > 0.0:
            p_inv_mass = 1.0 / (vol / 4.0)
            for j in ti.static(range(4)):
                inv_mass[p_indices[j]] += p_inv_mass

    for i in range(num_particles):
        pos[i] += ti.Vector([0.0, 1.0, 0.0])
        prev_pos[i] = pos[i]
        vel[i] = ti.Vector([0.0, 0.0, 0.0])


def build_hash_grid():
    """Multi-step process to build the spatial hash grid."""
    grid_cell_counts.fill(0)
    
    @ti.kernel
    def count_verts_in_cells():
        for i in range(num_vis_verts):
            cell = get_grid_cell(vis_mesh_rest_pos[i])
            ti.atomic_add(grid_cell_counts[cell], 1)
    
    count_verts_in_cells()

    total_verts = 0
    counts_np = grid_cell_counts.to_numpy()
    starts_np = np.zeros_like(counts_np)
    # This prefix sum must be done serially on the CPU.
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            for k in range(GRID_SIZE):
                starts_np[i, j, k] = total_verts
                total_verts += counts_np[i, j, k]
    grid_cell_starts.from_numpy(starts_np)

    write_offsets = ti.field(ti.i32, shape=(GRID_SIZE, GRID_SIZE, GRID_SIZE))
    write_offsets.copy_from(grid_cell_starts)

    @ti.kernel
    def fill_sorted_ids():
        for i in range(num_vis_verts):
            cell = get_grid_cell(vis_mesh_rest_pos[i])
            write_idx = ti.atomic_add(write_offsets[cell], 1)
            sorted_vis_vert_ids[write_idx] = i

    fill_sorted_ids()

@ti.kernel
def compute_skinning_info_hashed():
    min_dist.fill(1e9)
    skinning_info_tet_idx.fill(-1)

    for tet_idx in range(num_tets):
        p_indices = ti.Vector([tet_ids[tet_idx, 0], tet_ids[tet_idx, 1], tet_ids[tet_idx, 2], tet_ids[tet_idx, 3]])
        p0, p1, p2, p3 = pos[p_indices[0]], pos[p_indices[1]], pos[p_indices[2]], pos[p_indices[3]]
        
        min_coord = min(p0, p1, p2, p3) - CELL_SPACING
        max_coord = max(p0, p1, p2, p3) + CELL_SPACING
        min_cell, max_cell = get_grid_cell(min_coord), get_grid_cell(max_coord)

        for i, j, k in ti.ndrange(
            (min_cell[0], max_cell[0] + 1), (min_cell[1], max_cell[1] + 1), (min_cell[2], max_cell[2] + 1)):
            
            start, end = grid_cell_starts[i, j, k], grid_cell_starts[i, j, k] + grid_cell_counts[i, j, k]
            
            for vert_offset in range(start, end):
                vis_vert_idx = sorted_vis_vert_ids[vert_offset]
                p_vis = vis_mesh_rest_pos[vis_vert_idx]
                
                bary = get_barycentric_coords(p_vis, p0, p1, p2, p3)
                dist = 0.0
                for c in ti.static(range(4)): dist = max(dist, -bary[c])
                
                if dist < ti.atomic_min(min_dist[vis_vert_idx], dist):
                    skinning_info_tet_idx[vis_vert_idx] = tet_idx
                    skinning_info_bary_weights[vis_vert_idx] = ti.Vector([bary[0], bary[1], bary[2]])

@ti.kernel
def update_vis_mesh():
    for i in range(num_vis_verts):
        tet_idx = skinning_info_tet_idx[i]
        if tet_idx < 0:
            vis_mesh_pos[i] = vis_mesh_rest_pos[i]
            continue
        p_indices = ti.Vector([tet_ids[tet_idx, 0], tet_ids[tet_idx, 1], tet_ids[tet_idx, 2], tet_ids[tet_idx, 3]])
        p0, p1, p2, p3 = pos[p_indices[0]], pos[p_indices[1]], pos[p_indices[2]], pos[p_indices[3]]
        b = skinning_info_bary_weights[i]
        b3 = 1.0 - b.sum()
        vis_mesh_pos[i] = b[0] * p0 + b[1] * p1 + b[2] * p2 + b3 * p3

@ti.kernel
def pre_solve(dt: ti.f64, use_gravity: ti.i32):
    for i in range(num_particles):
        if inv_mass[i] == 0.0: continue
        if use_gravity != 0:
            vel[i] += gravity * dt
        prev_pos[i] = pos[i]
        pos[i] += vel[i] * dt
        
        # Box collision - clamp position to box bounds
        pos[i] = clamp_to_box(pos[i])

@ti.kernel
def solve_edges(compliance: ti.f64, dt: ti.f64):
    alpha = compliance / (dt * dt)
    for i in range(num_edges):
        id0, id1 = edge_ids[i, 0], edge_ids[i, 1]
        w0, w1 = inv_mass[id0], inv_mass[id1]
        w_sum = w0 + w1
        if w_sum == 0.0: continue
        delta = pos[id0] - pos[id1]
        dist = delta.norm()
        if dist == 0.0: continue
        grad = delta / dist
        C = dist - edge_lengths[i]
        s = -C / (w_sum + alpha)
        pos[id0] += s * w0 * grad
        pos[id1] -= s * w1 * grad

@ti.kernel
def solve_volumes(compliance: ti.f64, dt: ti.f64):
    alpha = compliance / (dt * dt)
    for i in range(num_tets):
        p_indices = ti.Vector([tet_ids[i, 0], tet_ids[i, 1], tet_ids[i, 2], tet_ids[i, 3]])
        w_sum = 0.0
        grads = ti.Matrix.zero(ti.f64, 4, 3)
        for j in ti.static(range(4)):
            ids = ti.Vector([p_indices[vol_id_order[j][c]] for c in range(3)])
            p0, p1, p2 = pos[ids[0]], pos[ids[1]], pos[ids[2]]
            grad = (p1 - p0).cross(p2 - p0) / 6.0
            grads[j, :] = grad
            w_sum += inv_mass[p_indices[j]] * grad.norm_sqr()
        if w_sum == 0.0: continue
        C = get_tet_volume(p_indices) - rest_vol[i]
        s = -C / (w_sum + alpha)
        for j in ti.static(range(4)):
            pos[p_indices[j]] += s * inv_mass[p_indices[j]] * grads[j, :]

@ti.kernel
def post_solve(dt: ti.f64):
    for i in range(num_particles):
        if inv_mass[i] == 0.0: continue
        vel[i] = (pos[i] - prev_pos[i]) / dt

def substep():
    pre_solve(sdt, 1)
    for _ in range(solver_iterations):
        solve_edges(edge_compliance, sdt)
        solve_volumes(vol_compliance, sdt)
    post_solve(sdt)

print("Building spatial hash grid for skinning...")
start_time = time.time()
build_hash_grid()
end_time = time.time()
print(f"Hash grid built in {end_time - start_time:.4f} seconds.")

print("Computing skinning information with hash grid...")
start_time = time.time()
compute_skinning_info_hashed()
end_time = time.time()
print(f"Skinning computation finished in {end_time - start_time:.4f} seconds.")

init_physics()

# 2. Setup UI and Camera
window = ti.ui.Window("Taichi XPBD - Dragon Simulation", (1024, 1024), vsync=True)
canvas = window.get_canvas()
scene = window.get_scene()
camera = ti.ui.Camera()
camera.position(0, 1.5, 4.5)
camera.lookat(0, 0.5, 0)

# 3. Main simulation and rendering loop
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
            init_physics()
            export_frame_count = 0
            if usd_exporter:
                usd_exporter = None

    if not paused:
        for _ in range(num_substeps):
            substep()

    # Update visual mesh before rendering
    update_vis_mesh()
    
    # Export mesh frame if enabled
    if export_enabled and not paused and usd_exporter:
        # Get current mesh data
        vertices = vis_mesh_pos.to_numpy()
        faces = vis_mesh_indices.to_numpy().reshape((-1, 3))
        
        # Export the frame to USD
        usd_exporter.export_frame(vertices, faces)
        export_frame_count += 1

    # Rendering
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(2, 3, 4), color=(1, 1, 1))
    
    scene.mesh(vis_mesh_pos, indices=vis_mesh_indices, per_vertex_color=vis_mesh_colors)
    scene.mesh(ground_vertices, indices=ground_indices, per_vertex_color=ground_colors)

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
