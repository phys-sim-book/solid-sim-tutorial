import copy
from cmath import inf

import numpy as np
import numpy.linalg as LA
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

import InertiaEnergy
import MassSpringEnergy

def step_forward(x, e, v, m, l2, k, h, tol):
    x_tilde = x + v * h     # implicit Euler predictive position
    x_n = copy.deepcopy(x)

    # Newton loop
    iter = 0
    E_last = IP_val(x, e, x_tilde, m, l2, k, h)
    p = search_dir(x, e, x_tilde, m, l2, k, h)
    while LA.norm(p, inf) / h > tol:
        print('Iteration', iter, ':')
        print('residual =', LA.norm(p, inf) / h)

        # line search
        alpha = 1
        while IP_val(x + alpha * p, e, x_tilde, m, l2, k, h) > E_last:
            alpha /= 2
        print('step size =', alpha)

        x += alpha * p
        E_last = IP_val(x, e, x_tilde, m, l2, k, h)
        p = search_dir(x, e, x_tilde, m, l2, k, h)
        iter += 1

    v = (x - x_n) / h   # implicit Euler velocity update
    return [x, v]

def IP_val(x, e, x_tilde, m, l2, k, h):
    return InertiaEnergy.val(x, x_tilde, m) + h * h * MassSpringEnergy.val(x, e, l2, k)     # implicit Euler

def IP_grad(x, e, x_tilde, m, l2, k, h):
    return InertiaEnergy.grad(x, x_tilde, m) + h * h * MassSpringEnergy.grad(x, e, l2, k)   # implicit Euler

def IP_hess(x, e, x_tilde, m, l2, k, h):
    IJV_In = InertiaEnergy.hess(x, x_tilde, m)
    IJV_MS = MassSpringEnergy.hess(x, e, l2, k)
    IJV_MS[2] *= h * h    # implicit Euler
    IJV = np.append(IJV_In, IJV_MS, axis=1)
    H = sparse.coo_matrix((IJV[2], (IJV[0], IJV[1])), shape=(len(x) * 2, len(x) * 2)).tocsr()
    return H

def search_dir(x, e, x_tilde, m, l2, k, h):
    projected_hess = IP_hess(x, e, x_tilde, m, l2, k, h)
    reshaped_grad = IP_grad(x, e, x_tilde, m, l2, k, h).reshape(len(x) * 2, 1)
    return spsolve(projected_hess, -reshaped_grad).reshape(len(x), 2)