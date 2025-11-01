import taichi as ti
import numpy as np
import json
import math
import time
import warp as wp
import warp.render

ti.init(arch=ti.cpu, default_fp=ti.f64)

class USDExporter:
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
        self.renderer.save()
        print(f"\nUSD animation saved: {self.output_path}")
        print(f"Total frames: {self.frame_count}")
        print(f"Duration: {self.sim_time:.3f} seconds")
        print(f"\nTo import into Blender:")
        print(f"1. File > Import > Universal Scene Description (.usd)")
        print(f"2. Select '{self.output_path}'")
        print(f"3. The animation will be imported as a single animated sequence!")

def load_cloth_data_from_json(filepath="cloth_data.json"):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        vertices = np.array(data['vertices'], dtype=np.float64).reshape((-1, 3))
        face_tri_ids = np.array(data['faceTriIds'], dtype=np.int32)
        print(f"Loaded cloth data: {len(vertices)} vertices, {len(face_tri_ids)//3} triangles")
        return vertices, face_tri_ids
    except Exception as e:
        print(f"Error loading cloth data from {filepath}: {e}")
        print("Falling back to procedural cloth generation...")
        verts_np, tris_np = create_cloth_mesh_data()
        return verts_np, tris_np.flatten()

def create_cloth_mesh_data(width=20, height=15, spacing=0.1):
    num_particles = width * height
    verts = np.zeros((num_particles, 3), dtype=np.float64)
    offset_x = - (width - 1) * spacing / 2.0
    offset_y = - (height - 1) * spacing / 2.0 + (height * spacing * 0.8)
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            verts[idx] = [x * spacing + offset_x, y * spacing + offset_y, 0]
    num_tris = 2 * (width - 1) * (height - 1)
    tri_ids = np.zeros((num_tris, 3), dtype=np.int32)
    tri_idx = 0
    for y in range(height - 1):
        for x in range(width - 1):
            p0, p1, p2, p3 = y*width+x, y*width+x+1, (y+1)*width+x, (y+1)*width+x+1
            tri_ids[tri_idx], tri_idx = [p0, p1, p2], tri_idx + 1
            tri_ids[tri_idx], tri_idx = [p1, p3, p2], tri_idx + 1
    return verts, tri_ids

def find_constraint_indices(tri_ids_np):
    edge_to_tri_map = {}
    for i, tri in enumerate(tri_ids_np):
        for j in range(3):
            v0_idx, v1_idx = tri[j], tri[(j + 1) % 3]
            edge = tuple(sorted((v0_idx, v1_idx)))
            if edge not in edge_to_tri_map:
                edge_to_tri_map[edge] = []
            edge_to_tri_map[edge].append(i)
    stretching_ids = list(edge_to_tri_map.keys())
    bending_ids = []
    for edge, tris in edge_to_tri_map.items():
        if len(tris) == 2:
            tri0_idx, tri1_idx = tris[0], tris[1]
            p2 = [v for v in tri_ids_np[tri0_idx] if v not in edge][0]
            p3 = [v for v in tri_ids_np[tri1_idx] if v not in edge][0]
            bending_ids.append([p2, p3])
    return np.array(stretching_ids, dtype=np.int32), np.array(bending_ids, dtype=np.int32)

paused = True
gravity = ti.Vector([0.0, -9.8, 0.0])
dt = 1.0 / 60.0
num_substeps = 15
sdt = dt / num_substeps
solver_iterations = 1

export_enabled = False
export_frame_count = 0
usd_exporter = None

verts_np, face_tri_ids_np = load_cloth_data_from_json()
tris_np = face_tri_ids_np.reshape((-1, 3))
stretching_ids_np, bending_ids_np = find_constraint_indices(tris_np)

num_particles = len(verts_np)
num_tris = len(tris_np)
num_stretching_constraints = len(stretching_ids_np)
num_bending_constraints = len(bending_ids_np)

stretching_compliance = 0.0
bending_compliance = 1.0

pos = ti.Vector.field(3, dtype=ti.f64, shape=num_particles)
prev_pos = ti.Vector.field(3, dtype=ti.f64, shape=num_particles)
vel = ti.Vector.field(3, dtype=ti.f64, shape=num_particles)
inv_mass = ti.field(dtype=ti.f64, shape=num_particles)
original_inv_mass = ti.field(dtype=ti.f64, shape=num_particles)

tri_ids = ti.field(ti.i32, shape=num_tris * 3)
stretching_ids = ti.field(ti.i32, shape=(num_stretching_constraints, 2))
bending_ids = ti.field(ti.i32, shape=(num_bending_constraints, 2))

stretching_lengths = ti.field(dtype=ti.f64, shape=num_stretching_constraints)
bending_lengths = ti.field(dtype=ti.f64, shape=num_bending_constraints)

ground_vertices = ti.Vector.field(3, dtype=ti.f64, shape=4)
ground_indices = ti.field(ti.i32, shape=6)
grab_indicator_pos = ti.Vector.field(3, dtype=ti.f64, shape=1)

vertex_colors = ti.Vector.field(3, dtype=ti.f64, shape=num_particles)

pos.from_numpy(verts_np)
tri_ids.from_numpy(tris_np.flatten())
stretching_ids.from_numpy(stretching_ids_np)
bending_ids.from_numpy(bending_ids_np)
ground_vertices.from_numpy(np.array([[-10, 0, -10], [10, 0, -10], [10, 0, 10], [-10, 0, 10]], dtype=np.float64))
ground_indices.from_numpy(np.array([0, 1, 2, 0, 2, 3], dtype=np.int32))

@ti.kernel
def init_physics():
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
def pin_top_vertices():
    max_y = -1e9
    for i in range(num_particles):
        if pos[i].y > max_y:
            max_y = pos[i].y
    for i in range(num_particles):
        if pos[i].y >= max_y - 1e-6:
            inv_mass[i] = 0.0

@ti.kernel
def initialize_colors():
    for i in range(num_particles):
        x_coord = int((pos[i].x + 1.0) * 5) % 2
        z_coord = int((pos[i].z + 1.0) * 5) % 2
        
        if (x_coord + z_coord) % 2 == 0:
            vertex_colors[i] = ti.Vector([1.0, 1.0, 1.0])
        else:
            vertex_colors[i] = ti.Vector([0.1, 0.1, 0.1])

@ti.kernel
def apply_grab(particle_idx: ti.i32, target_x: ti.f64, target_y: ti.f64, target_z: ti.f64):
    if particle_idx != -1:
        target_pos = ti.Vector([target_x, target_y, target_z])
        pos[particle_idx] = target_pos
        vel[particle_idx] = ti.Vector([0.0, 0.0, 0.0])
        grab_indicator_pos[0] = target_pos

@ti.kernel
def pre_solve(dt: ti.f64):
    for i in range(num_particles):
        if inv_mass[i] == 0.0: continue
        vel[i] += gravity * dt
        prev_pos[i] = pos[i]
        pos[i] += vel[i] * dt
        if pos[i].y < 0.0:
            pos[i] = prev_pos[i]
            pos[i].y = 0.0

@ti.kernel
def solve_stretching_constraints(compliance: ti.f64, dt: ti.f64):
    alpha = compliance / (dt * dt)
    for i in range(num_stretching_constraints):
        id0, id1 = stretching_ids[i, 0], stretching_ids[i, 1]
        w0, w1 = inv_mass[id0], inv_mass[id1]
        w_sum = w0 + w1
        if w_sum == 0.0: continue
        p0, p1 = pos[id0], pos[id1]
        delta = p0 - p1
        dist = delta.norm()
        if dist == 0.0: continue
        grad = delta / dist
        C = dist - stretching_lengths[i]
        s = -C / (w_sum + alpha)
        pos[id0] += s * w0 * grad
        pos[id1] -= s * w1 * grad

@ti.kernel
def solve_bending_constraints(compliance: ti.f64, dt: ti.f64):
    alpha = compliance / (dt * dt)
    for i in range(num_bending_constraints):
        id0, id1 = bending_ids[i, 0], bending_ids[i, 1]
        w0, w1 = inv_mass[id0], inv_mass[id1]
        w_sum = w0 + w1
        if w_sum == 0.0: continue
        p0, p1 = pos[id0], pos[id1]
        delta = p0 - p1
        dist = delta.norm()
        if dist == 0.0: continue
        grad = delta / dist
        C = dist - bending_lengths[i]
        s = -C / (w_sum + alpha)
        pos[id0] += s * w0 * grad
        pos[id1] -= s * w1 * grad

@ti.kernel
def post_solve(dt: ti.f64):
    for i in range(num_particles):
        if inv_mass[i] == 0.0: continue
        vel[i] = (pos[i] - prev_pos[i]) / dt

def substep(grab_id, grab_x, grab_y, grab_z):
    pre_solve(sdt)
    for _ in range(solver_iterations):
        solve_stretching_constraints(stretching_compliance, sdt)
        solve_bending_constraints(bending_compliance, sdt)
    apply_grab(grab_id, grab_x, grab_y, grab_z)
    post_solve(sdt)

def find_bottom_corner():
    positions = pos.to_numpy()
    min_y = float('inf')
    min_x = float('inf')
    corner_idx = -1
    
    for i in range(num_particles):
        if positions[i, 1] < min_y:
            min_y = positions[i, 1]
    
    for i in range(num_particles):
        if abs(positions[i, 1] - min_y) < 1e-6:
            if positions[i, 0] < min_x:
                min_x = positions[i, 0]
                corner_idx = i
    
    print(f"Found bottom corner particle: {corner_idx} at position ({min_x:.3f}, {min_y:.3f})")
    return corner_idx

def calculate_quadratic_path_pos(start_pos, end_pos, t):
    t_squared = t * t
    scaled_displacement = (end_pos - start_pos) * 0.3
    return start_pos + scaled_displacement * t_squared

init_physics()
pin_top_vertices()
initialize_colors()

window = ti.ui.Window("Taichi XPBD Cloth", (1024, 1024), vsync=True)
canvas = window.get_canvas()
scene = window.get_scene()
camera = ti.ui.Camera()
camera.position(0, 1.0, 3.0)
camera.lookat(0, 0.5, 0)

is_grabbing = False
grabbed_particle_id = -1
grab_distance = 0.0
bottom_corner_id = find_bottom_corner()

is_animating = False
animation_start_time = 0.0
animation_duration = 2.0
original_corner_pos = None
target_corner_pos = None
start_time = time.time()

while window.running:
    gui = window.GUI
    with gui.sub_window("Controls", 0.05, 0.05, 0.3, 0.4):
        paused = gui.checkbox("Paused", paused)
        bending_compliance = gui.slider_float("Bending Compliance", bending_compliance, 0.0, 10.0)
        
        gui.text("Bottom Corner Control:")
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
            init_physics()
            pin_top_vertices()
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
