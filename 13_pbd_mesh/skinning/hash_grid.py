"""
This module builds a spatial hash grid for efficient nearest-neighbor queries used in the skinning computation.
"""

import taichi as ti
import numpy as np
import sys
import os
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from xpbd_base import get_grid_cell


def build_hash_grid(
    num_vis_verts: int, grid_size: int, inv_cell_spacing: float,
    vis_mesh_rest_pos: ti.template(), grid_cell_counts: ti.template(),
    grid_cell_starts: ti.template(), sorted_vis_vert_ids: ti.template()
):
    """
    Build a spatial hash grid for efficient skinning queries.
    
    1. Count how many visual vertices fall into each grid cell
    2. Compute prefix sums to determine starting indices for each cell
    3. Sort visual vertex IDs by their grid cell locations
    """
    grid_cell_counts.fill(0)
    
    @ti.kernel
    def count_verts_in_cells(
        num_vis_verts: ti.i32, inv_cell_spacing: ti.f64, grid_size: ti.i32,
        vis_mesh_rest_pos: ti.template(), grid_cell_counts: ti.template()
    ):
        for i in range(num_vis_verts):
            cell = get_grid_cell(vis_mesh_rest_pos[i], inv_cell_spacing, grid_size)
            ti.atomic_add(grid_cell_counts[cell], 1)
    
    count_verts_in_cells(num_vis_verts, inv_cell_spacing, grid_size, vis_mesh_rest_pos, grid_cell_counts)

    total_verts = 0
    counts_np = grid_cell_counts.to_numpy()
    starts_np = np.zeros_like(counts_np)
    # This prefix sum must be done serially on the CPU.
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                starts_np[i, j, k] = total_verts
                total_verts += counts_np[i, j, k]
    grid_cell_starts.from_numpy(starts_np)

    write_offsets = ti.field(ti.i32, shape=(grid_size, grid_size, grid_size))
    write_offsets.copy_from(grid_cell_starts)

    @ti.kernel
    def fill_sorted_ids(
        num_vis_verts: ti.i32, inv_cell_spacing: ti.f64, grid_size: ti.i32,
        vis_mesh_rest_pos: ti.template(), write_offsets: ti.template(),
        sorted_vis_vert_ids: ti.template()
    ):
        for i in range(num_vis_verts):
            cell = get_grid_cell(vis_mesh_rest_pos[i], inv_cell_spacing, grid_size)
            write_idx = ti.atomic_add(write_offsets[cell], 1)
            sorted_vis_vert_ids[write_idx] = i

    fill_sorted_ids(num_vis_verts, inv_cell_spacing, grid_size, vis_mesh_rest_pos, write_offsets, sorted_vis_vert_ids)
