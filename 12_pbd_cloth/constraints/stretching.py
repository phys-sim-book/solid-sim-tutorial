import taichi as ti


@ti.kernel
def solve_stretching_constraints(
    compliance: ti.f64, dt: ti.f64, num_stretching_constraints: ti.i32,
    pos: ti.template(), stretching_ids: ti.template(), stretching_lengths: ti.template(),
    inv_mass: ti.template(), lambdas: ti.template()
):
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
        dlambda = -(C + alpha * lambdas[i]) / (w_sum + alpha)
        lambdas[i] += dlambda
        pos[id0] += dlambda * w0 * grad
        pos[id1] -= dlambda * w1 * grad
