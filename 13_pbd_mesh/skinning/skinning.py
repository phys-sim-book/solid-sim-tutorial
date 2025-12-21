"""
This module handles the computation of skinning weights and updates the visual mesh.
Skinning maps visual mesh vertices to simulation tetrahedra using barycentric coordinates,
allowing the visual mesh to deform smoothly with the simulation mesh.
"""

import taichi as ti
import sys
import os
# Add parent directory to path to import xpbd_base
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from xpbd_base import get_barycentric_coords, get_grid_cell


@ti.kernel
def compute_skinning_info_hashed(
    num_tets: ti.i32, num_vis_verts: ti.i32, cell_spacing: ti.f64,
    inv_cell_spacing: ti.f64, grid_size: ti.i32,
    pos: ti.template(), tet_ids: ti.template(), vis_mesh_rest_pos: ti.template(),
    grid_cell_counts: ti.template(), grid_cell_starts: ti.template(),
    sorted_vis_vert_ids: ti.template(), min_dist: ti.template(),
    skinning_info_tet_idx: ti.template(), skinning_info_bary_weights: ti.template()
):
    """
    For each simulation tetrahedron, this kernel:
    1. Determines which grid cells the tetrahedron overlaps
    2. Queries the hash grid to find nearby visual vertices
    3. Computes barycentric coordinates for each visual vertex
    4. Stores the closest tetrahedron and barycentric weights for each visual vertex
    """
    min_dist.fill(1e9)
    skinning_info_tet_idx.fill(-1)

    for tet_idx in range(num_tets):
        p_indices = ti.Vector([tet_ids[tet_idx, 0], tet_ids[tet_idx, 1], tet_ids[tet_idx, 2], tet_ids[tet_idx, 3]])
        p0, p1, p2, p3 = pos[p_indices[0]], pos[p_indices[1]], pos[p_indices[2]], pos[p_indices[3]]
        
        min_coord = min(p0, p1, p2, p3) - cell_spacing
        max_coord = max(p0, p1, p2, p3) + cell_spacing
        min_cell = get_grid_cell(min_coord, inv_cell_spacing, grid_size)
        max_cell = get_grid_cell(max_coord, inv_cell_spacing, grid_size)

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
def update_vis_mesh(
    num_vis_verts: ti.i32,
    pos: ti.template(), tet_ids: ti.template(),
    vis_mesh_rest_pos: ti.template(), vis_mesh_pos: ti.template(),
    skinning_info_tet_idx: ti.template(), skinning_info_bary_weights: ti.template()
):
    """
    For each visual vertex, this kernel:
    1. Looks up the associated simulation tetrahedron
    2. Retrieves the barycentric weights
    3. Interpolates the visual vertex position from the tetrahedron vertices
    """
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
