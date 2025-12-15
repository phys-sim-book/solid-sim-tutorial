import taichi as ti
import sys
import os
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from sph_base import poly6_value, spiky_gradient


@ti.func
def compute_scorr(pos_ji, h_, poly6_factor_, corr_deltaQ_coeff_, corrK_):
    x = poly6_value(pos_ji.norm(), h_, poly6_factor_) / poly6_value(corr_deltaQ_coeff_ * h_, h_, poly6_factor_)
    x = x * x; x = x * x
    return -corrK_ * x


@ti.kernel
def solve_density_constraints(
    positions: ti.template(),
    particle_num_neighbors: ti.template(),
    particle_neighbors: ti.template(),
    lambdas: ti.template(),
    position_deltas: ti.template(),
    h_: float,
    poly6_factor_: float,
    spiky_grad_factor_: float,
    mass_: float,
    rho0_: float,
    lambda_epsilon_: float,
    epsilon_: float,
    corr_deltaQ_coeff_: float,
    corrK_: float
):

    # Pass 1: Calculate lambda values (constraint multipliers)
    for p_i in positions:
        pos_i = positions[p_i]
        grad_i, sum_gradient_sqr, density_constraint = ti.Vector([0.0, 0.0, 0.0]), 0.0, 0.0

        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            pos_ji = pos_i - positions[p_j]
            grad_j = spiky_gradient(pos_ji, h_, spiky_grad_factor_)
            grad_i += grad_j
            sum_gradient_sqr += grad_j.dot(grad_j)
            density_constraint += poly6_value(pos_ji.norm(), h_, poly6_factor_)

        density_constraint = (mass_ * density_constraint / rho0_) - 1.0
        sum_gradient_sqr += grad_i.dot(grad_i)
        
        denominator = (1.0 / mass_) * sum_gradient_sqr + lambda_epsilon_
        lambdas[p_i] = -density_constraint / denominator if denominator > epsilon_ else 0.0

    # Pass 2: Calculate position corrections
    for p_i in positions:
        pos_i = positions[p_i]
        lambda_i = lambdas[p_i]
        pos_delta_i = ti.Vector([0.0, 0.0, 0.0])

        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            lambda_j = lambdas[p_j]
            pos_ji = pos_i - positions[p_j]
            scorr_ij = compute_scorr(pos_ji, h_, poly6_factor_, corr_deltaQ_coeff_, corrK_)
            pos_delta_i += (lambda_i + lambda_j + scorr_ij) * spiky_gradient(pos_ji, h_, spiky_grad_factor_)
        
        position_deltas[p_i] = pos_delta_i / rho0_

    # Pass 3: Apply position corrections
    for i in positions:
        positions[i] += position_deltas[i]
