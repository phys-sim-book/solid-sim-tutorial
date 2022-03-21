import numpy as np
import utils

epsv = 1e-3

def f0(vhatnorm, epsv, hhat):
    if vhatnorm >= epsv:
        return vhatnorm * hhat
    else:
        vhatnormhhat = vhatnorm * hhat
        epsvhhat = epsv * hhat
        return vhatnormhhat * vhatnormhhat * (-vhatnormhhat / 3.0 + epsvhhat) / (epsvhhat * epsvhhat) + epsvhhat / 3.0

def f1_div_vhatnorm(vhatnorm, epsv):
    if vhatnorm >= epsv:
        return 1.0 / vhatnorm
    else:
        return (-vhatnorm + 2.0 * epsv) / (epsv * epsv)

def f_hess_term(vhatnorm, epsv):
    if vhatnorm >= epsv:
        return -1.0 / (vhatnorm * vhatnorm)
    else:
        return -1.0 / (epsv * epsv)

def val(v, mu_lambda, hhat, n):
    sum = 0.0
    T = np.identity(2) - np.outer(n, n) # tangent of slope is constant
    for i in range(0, len(v)):
        if mu_lambda[i] > 0:
            vhat = np.transpose(T).dot(v[i])
            sum += mu_lambda[i] * f0(np.linalg.norm(vhat), epsv, hhat)
    return sum

def grad(v, mu_lambda, hhat, n):
    g = np.array([[0.0, 0.0]] * len(v))
    T = np.identity(2) - np.outer(n, n) # tangent of slope is constant
    for i in range(0, len(v)):
        if mu_lambda[i] > 0:
            vhat = np.transpose(T).dot(v[i])
            g[i] = mu_lambda[i] * f1_div_vhatnorm(np.linalg.norm(vhat), epsv) * T.dot(vhat)
    return g

def hess(v, mu_lambda, hhat, n):
    IJV = [[0] * 0, [0] * 0, np.array([0.0] * 0)]
    T = np.identity(2) - np.outer(n, n) # tangent of slope is constant
    for i in range(0, len(v)):
        if mu_lambda[i] > 0:
            vhat = np.transpose(T).dot(v[i])
            vhatnorm = np.linalg.norm(vhat)
            inner_term = f1_div_vhatnorm(vhatnorm, epsv) * np.identity(2)
            if vhatnorm != 0:
                inner_term += f_hess_term(vhatnorm, epsv) / vhatnorm * np.outer(vhat, vhat)
            local_hess = mu_lambda[i] * T.dot(utils.make_PD(inner_term)).dot(np.transpose(T)) / hhat
            for c in range(0, 2):
                for r in range(0, 2):
                    IJV[0].append(i * 2 + r)
                    IJV[1].append(i * 2 + c)
                    IJV[2] = np.append(IJV[2], local_hess[r, c])
    return IJV