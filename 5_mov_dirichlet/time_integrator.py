import copy
from cmath import inf

import numpy as np
import numpy.linalg as LA
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

import InertiaEnergy
import MassSpringEnergy
import GravityEnergy
import BarrierEnergy
import FrictionEnergy
import SpringEnergy

def step_forward(x, e, v, m, l2, k, n, o, contact_area, mu, is_DBC, DBC, DBC_v, DBC_limit, DBC_stiff, h, tol):
    x_tilde = x + v * h     # implicit Euler predictive position
    x_n = copy.deepcopy(x)
    mu_lambda = BarrierEnergy.compute_mu_lambda(x, n, o, contact_area, mu)  # compute mu * lambda for each node using x^n
    # ANCHOR: dbc_initialization
    DBC_target = [] # target position of each DBC in the current time step
    for i in range(0, len(DBC)):
        if (DBC_limit[i] - x_n[DBC[i]]).dot(DBC_v[i]) > 0:
            DBC_target.append(x_n[DBC[i]] + h * DBC_v[i])
        else:
            DBC_target.append(x_n[DBC[i]])
    # ANCHOR_END: dbc_initialization

    # Newton loop
    iter = 0
    E_last = IP_val(x, e, x_tilde, m, l2, k, n, o, contact_area, (x - x_n) / h, mu_lambda, DBC, DBC_target, DBC_stiff[0], h)
    # ANCHOR: convergence_criteria
    [p, DBC_satisfied] = search_dir(x, e, x_tilde, m, l2, k, n, o, contact_area, (x - x_n) / h, mu_lambda, is_DBC, DBC, DBC_target, DBC_stiff[0], tol, h)
    while (LA.norm(p, inf) / h > tol) | (sum(DBC_satisfied) != len(DBC)):   # also check whether all DBCs are satisfied
        print('Iteration', iter, ':')
        print('residual =', LA.norm(p, inf) / h)

        if (LA.norm(p, inf) / h <= tol) & (sum(DBC_satisfied) != len(DBC)):
            # increase DBC stiffness and recompute energy value record
            DBC_stiff[0] *= 2
            E_last = IP_val(x, e, x_tilde, m, l2, k, n, o, contact_area, (x - x_n) / h, mu_lambda, DBC, DBC_target, DBC_stiff[0], h)
        # ANCHOR_END: convergence_criteria

        # filter line search
        alpha = BarrierEnergy.init_step_size(x, n, o, p)  # avoid interpenetration and tunneling
        while IP_val(x + alpha * p, e, x_tilde, m, l2, k, n, o, contact_area, (x + alpha * p - x_n) / h, mu_lambda, DBC, DBC_target, DBC_stiff[0], h) > E_last:
            alpha /= 2
        print('step size =', alpha)

        x += alpha * p
        E_last = IP_val(x, e, x_tilde, m, l2, k, n, o, contact_area, (x - x_n) / h, mu_lambda, DBC, DBC_target, DBC_stiff[0], h)
        [p, DBC_satisfied] = search_dir(x, e, x_tilde, m, l2, k, n, o, contact_area, (x - x_n) / h, mu_lambda, is_DBC, DBC, DBC_target, DBC_stiff[0], tol, h)
        iter += 1

    v = (x - x_n) / h   # implicit Euler velocity update
    return [x, v]

def IP_val(x, e, x_tilde, m, l2, k, n, o, contact_area, v, mu_lambda, DBC, DBC_target, DBC_stiff, h):
    return InertiaEnergy.val(x, x_tilde, m) + h * h * (     # implicit Euler
        MassSpringEnergy.val(x, e, l2, k) + 
        GravityEnergy.val(x, m) + 
        BarrierEnergy.val(x, n, o, contact_area) + 
        FrictionEnergy.val(v, mu_lambda, h, n)
    ) + SpringEnergy.val(x, m, DBC, DBC_target, DBC_stiff)

def IP_grad(x, e, x_tilde, m, l2, k, n, o, contact_area, v, mu_lambda, DBC, DBC_target, DBC_stiff, h):
    return InertiaEnergy.grad(x, x_tilde, m) + h * h * (    # implicit Euler
        MassSpringEnergy.grad(x, e, l2, k) + 
        GravityEnergy.grad(x, m) + 
        BarrierEnergy.grad(x, n, o, contact_area) + 
        FrictionEnergy.grad(v, mu_lambda, h, n)
    ) + SpringEnergy.grad(x, m, DBC, DBC_target, DBC_stiff)

def IP_hess(x, e, x_tilde, m, l2, k, n, o, contact_area, v, mu_lambda, DBC, DBC_target, DBC_stiff, h):
    IJV_In = InertiaEnergy.hess(x, x_tilde, m)
    IJV_MS = MassSpringEnergy.hess(x, e, l2, k)
    IJV_B = BarrierEnergy.hess(x, n, o, contact_area)
    IJV_F = FrictionEnergy.hess(v, mu_lambda, h, n)
    IJV_S = SpringEnergy.hess(x, m, DBC, DBC_target, DBC_stiff)
    IJV_MS[2] *= h * h    # implicit Euler
    IJV_B[2] *= h * h     # implicit Euler
    IJV_F[2] *= h * h     # implicit Euler
    IJV_In_MS = np.append(IJV_In, IJV_MS, axis=1)
    IJV_In_MS_B = np.append(IJV_In_MS, IJV_B, axis=1)
    IJV_In_MS_B_F = np.append(IJV_In_MS_B, IJV_F, axis=1)
    IJV = np.append(IJV_In_MS_B_F, IJV_S, axis=1)
    H = sparse.coo_matrix((IJV[2], (IJV[0], IJV[1])), shape=(len(x) * 2, len(x) * 2)).tocsr()
    return H

def search_dir(x, e, x_tilde, m, l2, k, n, o, contact_area, v, mu_lambda, is_DBC, DBC, DBC_target, DBC_stiff, tol, h):
    projected_hess = IP_hess(x, e, x_tilde, m, l2, k, n, o, contact_area, v, mu_lambda, DBC, DBC_target, DBC_stiff, h)
    reshaped_grad = IP_grad(x, e, x_tilde, m, l2, k, n, o, contact_area, v, mu_lambda, DBC, DBC_target, DBC_stiff, h).reshape(len(x) * 2, 1)
    # ANCHOR: dbc_check
    # check whether each DBC is satisfied
    DBC_satisfied = [False] * len(x)
    for i in range(0, len(DBC)):
        if LA.norm(x[DBC[i]] - DBC_target[i]) / h < tol:
            DBC_satisfied[DBC[i]] = True
    # ANCHOR_END: dbc_check
    # ANCHOR: dof_elimination
    # eliminate DOF if it's a satisfied DBC by modifying gradient and Hessian for DBC:
    for i, j in zip(*projected_hess.nonzero()):
        if (is_DBC[int(i / 2)] & DBC_satisfied[int(i / 2)]) | (is_DBC[int(j / 2)] & DBC_satisfied[int(j / 2)]): 
            projected_hess[i, j] = (i == j)
    for i in range(0, len(x)):
        if is_DBC[i] & DBC_satisfied[i]:
            reshaped_grad[i * 2] = reshaped_grad[i * 2 + 1] = 0.0
    return [spsolve(projected_hess, -reshaped_grad).reshape(len(x), 2), DBC_satisfied]
    # ANCHOR_END: dof_elimination