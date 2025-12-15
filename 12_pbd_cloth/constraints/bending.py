import taichi as ti


@ti.kernel
def solve_bending_constraints(
    compliance: ti.f64, dt: ti.f64, num_bending_constraints: ti.i32,
    pos: ti.template(), bending_ids: ti.template(), bending_lengths: ti.template(),
    inv_mass: ti.template()
):
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
