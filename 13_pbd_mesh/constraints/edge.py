import taichi as ti


@ti.kernel
def solve_edges(
    compliance: ti.f64, dt: ti.f64, num_edges: ti.i32,
    pos: ti.template(), edge_ids: ti.template(), edge_lengths: ti.template(),
    inv_mass: ti.template(), lambdas: ti.template()
):
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
        dlambda = -(C + alpha * lambdas[i]) / (w_sum + alpha)
        lambdas[i] += dlambda
        pos[id0] += dlambda * w0 * grad
        pos[id1] -= dlambda * w1 * grad
