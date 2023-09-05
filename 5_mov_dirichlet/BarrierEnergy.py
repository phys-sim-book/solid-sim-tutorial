import math
import numpy as np

dhat = 0.01
kappa = 1e5

def val(x, n, o, contact_area):
    sum = 0.0
    # floor:
    for i in range(0, len(x)):
        d = n.dot(x[i] - o)
        if d < dhat:
            s = d / dhat
            sum += contact_area[i] * dhat * kappa / 2 * (s - 1) * math.log(s)
    # ceil:
    n = np.array([0.0, -1.0])
    for i in range(0, len(x) - 1):
        d = n.dot(x[i] - x[-1])
        if d < dhat:
            s = d / dhat
            sum += contact_area[i] * dhat * kappa / 2 * (s - 1) * math.log(s)
    return sum

def grad(x, n, o, contact_area):
    g = np.array([[0.0, 0.0]] * len(x))
    # floor:
    for i in range(0, len(x)):
        d = n.dot(x[i] - o)
        if d < dhat:
            s = d / dhat
            g[i] = contact_area[i] * dhat * (kappa / 2 * (math.log(s) / dhat + (s - 1) / d)) * n
    # ceil:
    n = np.array([0.0, -1.0])
    for i in range(0, len(x) - 1):
        d = n.dot(x[i] - x[-1])
        if d < dhat:
            s = d / dhat
            local_grad = contact_area[i] * dhat * (kappa / 2 * (math.log(s) / dhat + (s - 1) / d)) * n
            g[i] += local_grad
            g[-1] -= local_grad
    return g

def hess(x, n, o, contact_area):
    IJV = [[0] * 0, [0] * 0, np.array([0.0] * 0)]
    # floor:
    for i in range(0, len(x)):
        d = n.dot(x[i] - o)
        if d < dhat:
            local_hess = contact_area[i] * dhat * kappa / (2 * d * d * dhat) * (d + dhat) * np.outer(n, n)
            for c in range(0, 2):
                for r in range(0, 2):
                    IJV[0].append(i * 2 + r)
                    IJV[1].append(i * 2 + c)
                    IJV[2] = np.append(IJV[2], local_hess[r, c])
    # ceil:
    n = np.array([0.0, -1.0])
    for i in range(0, len(x) - 1):
        d = n.dot(x[i] - x[-1])
        if d < dhat:
            local_hess = contact_area[i] * dhat * kappa / (2 * d * d * dhat) * (d + dhat) * np.outer(n, n)
            index = [i, len(x) - 1]
            for nI in range(0, 2):
                for nJ in range(0, 2):
                    for c in range(0, 2):
                        for r in range(0, 2):
                            IJV[0].append(index[nI] * 2 + r)
                            IJV[1].append(index[nJ] * 2 + c)
                            IJV[2] = np.append(IJV[2], ((-1) ** (nI != nJ)) * local_hess[r, c])
    return IJV

def init_step_size(x, n, o, p):
    alpha = 1
    # floor:
    for i in range(0, len(x)):
        p_n = p[i].dot(n)
        if p_n < 0:
            alpha = min(alpha, 0.9 * n.dot(x[i] - o) / -p_n)
    # ceil:
    n = np.array([0.0, -1.0])
    for i in range(0, len(x) - 1):
        p_n = (p[i] - p[-1]).dot(n)
        if p_n < 0:
            alpha = min(alpha, 0.9 * n.dot(x[i] - x[-1]) / -p_n)
    return alpha

def compute_mu_lambda(x, n, o, contact_area, mu):
    mu_lambda = np.array([0.0] * len(x))
    for i in range(0, len(x)):
        d = n.dot(x[i] - o)
        if d < dhat:
            s = d / dhat
            mu_lambda[i] = mu * -contact_area[i] * dhat * (kappa / 2 * (math.log(s) / dhat + (s - 1) / d))
    return mu_lambda