import json
import numpy as np


def load_cloth_data_from_json(filepath="cloth_data.json"):
    """
    Load cloth mesh data from a JSON file.
    
    Expected JSON format:
    {
        "vertices": [[x, y, z], ...],
        "faceTriIds": [i0, i1, i2, ...]
    }
    
    """
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
