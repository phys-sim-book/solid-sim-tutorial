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

def step_forward(x, e, v, m, l2, k, y_ground, contact_area, is_DBC, h, tol):
    x_tilde = x + v * h     # implicit Euler predictive position
    x_n = copy.deepcopy(x)

    # Newton loop
    iter = 0
    E_last = IP_val(x, e, x_tilde, m, l2, k, y_ground, contact_area, h)
    p = search_dir(x, e, x_tilde, m, l2, k, y_ground, contact_area, is_DBC, h)
    while LA.norm(p, inf) / h > tol:
        print('Iteration', iter, ':')
        print('residual =', LA.norm(p, inf) / h)

        # filter line search
        alpha = BarrierEnergy.init_step_size(x, y_ground, p)  # avoid interpenetration and tunneling
        while IP_val(x + alpha * p, e, x_tilde, m, l2, k, y_ground, contact_area, h) > E_last:
            alpha /= 2
        print('step size =', alpha)

        x += alpha * p
        E_last = IP_val(x, e, x_tilde, m, l2, k, y_ground, contact_area, h)
        p = search_dir(x, e, x_tilde, m, l2, k, y_ground, contact_area, is_DBC, h)
        iter += 1

    v = (x - x_n) / h   # implicit Euler velocity update
    return [x, v]

def IP_val(x, e, x_tilde, m, l2, k, y_ground, contact_area, h):
    return InertiaEnergy.val(x, x_tilde, m) + h * h * (MassSpringEnergy.val(x, e, l2, k) + GravityEnergy.val(x, m) + BarrierEnergy.val(x, y_ground, contact_area))     # implicit Euler

def IP_grad(x, e, x_tilde, m, l2, k, y_ground, contact_area, h):
    return InertiaEnergy.grad(x, x_tilde, m) + h * h * (MassSpringEnergy.grad(x, e, l2, k) + GravityEnergy.grad(x, m) + BarrierEnergy.grad(x, y_ground, contact_area))   # implicit Euler

def IP_hess(x, e, x_tilde, m, l2, k, y_ground, contact_area, h):
    IJV_In = InertiaEnergy.hess(x, x_tilde, m)
    IJV_MS = MassSpringEnergy.hess(x, e, l2, k)
    IJV_B = BarrierEnergy.hess(x, y_ground, contact_area)
    IJV_MS[2] *= h * h    # implicit Euler
    IJV_B[2] *= h * h     # implicit Euler
    IJV_In_MS = np.append(IJV_In, IJV_MS, axis=1)
    IJV = np.append(IJV_In_MS, IJV_B, axis=1)
    H = sparse.coo_matrix((IJV[2], (IJV[0], IJV[1])), shape=(len(x) * 2, len(x) * 2)).tocsr()
    return H

def search_dir(x, e, x_tilde, m, l2, k, y_ground, contact_area, is_DBC, h):
    projected_hess = IP_hess(x, e, x_tilde, m, l2, k, y_ground, contact_area, h)
    reshaped_grad = IP_grad(x, e, x_tilde, m, l2, k, y_ground, contact_area, h).reshape(len(x) * 2, 1)
    # eliminate DOF by modifying gradient and Hessian for DBC:
    for i, j in zip(*projected_hess.nonzero()):
        if is_DBC[int(i / 2)] | is_DBC[int(j / 2)]: 
            projected_hess[i, j] = (i == j)
    for i in range(0, len(x)):
        if is_DBC[i]:
            reshaped_grad[i * 2] = reshaped_grad[i * 2 + 1] = 0.0
    return spsolve(projected_hess, -reshaped_grad).reshape(len(x), 2)