import taichi as ti
import sys
import os
# Add parent directory to path to import xpbd_base
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from xpbd_base import get_tet_volume


@ti.kernel
def solve_volumes(
    compliance: ti.f64, dt: ti.f64, num_tets: ti.i32,
    pos: ti.template(), tet_ids: ti.template(), rest_vol: ti.template(), inv_mass: ti.template(),
    vol_id_order: ti.template()
):
    alpha = compliance / (dt * dt)
    for i in range(num_tets):
        p_indices = ti.Vector([tet_ids[i, 0], tet_ids[i, 1], tet_ids[i, 2], tet_ids[i, 3]])
        w_sum = 0.0
        grads = ti.Matrix.zero(ti.f64, 4, 3)
        for j in ti.static(range(4)):
            ids = ti.Vector([p_indices[vol_id_order[j, c]] for c in range(3)])
            p0, p1, p2 = pos[ids[0]], pos[ids[1]], pos[ids[2]]
            grad = (p1 - p0).cross(p2 - p0) / 6.0
            grads[j, :] = grad
            w_sum += inv_mass[p_indices[j]] * grad.norm_sqr()
        if w_sum == 0.0: continue
        C = get_tet_volume(p_indices, pos) - rest_vol[i]
        s = -C / (w_sum + alpha)
        for j in ti.static(range(4)):
            pos[p_indices[j]] += s * inv_mass[p_indices[j]] * grads[j, :]
