import numpy as np


def find_constraint_indices(tri_ids_np):
  
    edge_to_tri_map = {}
    
    # Build a map from edges to triangles
    for i, tri in enumerate(tri_ids_np):
        for j in range(3):
            v0_idx, v1_idx = tri[j], tri[(j + 1) % 3]
            edge = tuple(sorted((v0_idx, v1_idx)))
            if edge not in edge_to_tri_map:
                edge_to_tri_map[edge] = []
            edge_to_tri_map[edge].append(i)
    
    # All edges are stretching constraints
    stretching_ids = list(edge_to_tri_map.keys())
    
    # Bending constraints: diagonals between triangles sharing an edge
    bending_ids = []
    for edge, tris in edge_to_tri_map.items():
        if len(tris) == 2:  # Edge shared by exactly two triangles
            tri0_idx, tri1_idx = tris[0], tris[1]
            # Find the vertices not on the shared edge
            p2 = [v for v in tri_ids_np[tri0_idx] if v not in edge][0]
            p3 = [v for v in tri_ids_np[tri1_idx] if v not in edge][0]
            bending_ids.append([p2, p3])
    
    return np.array(stretching_ids, dtype=np.int32), np.array(bending_ids, dtype=np.int32)


def find_bottom_corner(pos_np):
   
    min_y = float('inf')
    min_x = float('inf')
    corner_idx = -1
    
    # Find minimum y
    for i in range(len(pos_np)):
        if pos_np[i, 1] < min_y:
            min_y = pos_np[i, 1]
    
    # Find minimum x among particles with minimum y
    for i in range(len(pos_np)):
        if abs(pos_np[i, 1] - min_y) < 1e-6:
            if pos_np[i, 0] < min_x:
                min_x = pos_np[i, 0]
                corner_idx = i
    
    print(f"Found bottom corner particle: {corner_idx} at position ({min_x:.3f}, {min_y:.3f})")
    return corner_idx


def calculate_quadratic_path_pos(start_pos, end_pos, t):
  
    t_squared = t * t
    scaled_displacement = (end_pos - start_pos) * 0.3
    return start_pos + scaled_displacement * t_squared
