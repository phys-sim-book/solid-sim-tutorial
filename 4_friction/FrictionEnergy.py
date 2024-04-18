# ANCHOR: f_terms
import numpy as np
import utils

epsv = 1e-3

def f0(vbarnorm, epsv, hhat):
    if vbarnorm >= epsv:
        return vbarnorm * hhat
    else:
        vbarnormhhat = vbarnorm * hhat
        epsvhhat = epsv * hhat
        return vbarnormhhat * vbarnormhhat * (-vbarnormhhat / 3.0 + epsvhhat) / (epsvhhat * epsvhhat) + epsvhhat / 3.0

def f1_div_vbarnorm(vbarnorm, epsv):
    if vbarnorm >= epsv:
        return 1.0 / vbarnorm
    else:
        return (-vbarnorm + 2.0 * epsv) / (epsv * epsv)

def f_hess_term(vbarnorm, epsv):
    if vbarnorm >= epsv:
        return -1.0 / (vbarnorm * vbarnorm)
    else:
        return -1.0 / (epsv * epsv)
# ANCHOR_END: f_terms

# ANCHOR: val_grad_hess
def val(v, mu_lambda, hhat, n):
    sum = 0.0
    T = np.identity(2) - np.outer(n, n) # tangent of slope is constant
    for i in range(0, len(v)):
        if mu_lambda[i] > 0:
            vbar = np.transpose(T).dot(v[i])
            sum += mu_lambda[i] * f0(np.linalg.norm(vbar), epsv, hhat)
    return sum

def grad(v, mu_lambda, hhat, n):
    g = np.array([[0.0, 0.0]] * len(v))
    T = np.identity(2) - np.outer(n, n) # tangent of slope is constant
    for i in range(0, len(v)):
        if mu_lambda[i] > 0:
            vbar = np.transpose(T).dot(v[i])
            g[i] = mu_lambda[i] * f1_div_vbarnorm(np.linalg.norm(vbar), epsv) * T.dot(vbar)
    return g

def hess(v, mu_lambda, hhat, n):
    IJV = [[0] * 0, [0] * 0, np.array([0.0] * 0)]
    T = np.identity(2) - np.outer(n, n) # tangent of slope is constant
    for i in range(0, len(v)):
        if mu_lambda[i] > 0:
            vbar = np.transpose(T).dot(v[i])
            vbarnorm = np.linalg.norm(vbar)
            inner_term = f1_div_vbarnorm(vbarnorm, epsv) * np.identity(2)
            if vbarnorm != 0:
                inner_term += f_hess_term(vbarnorm, epsv) / vbarnorm * np.outer(vbar, vbar)
            local_hess = mu_lambda[i] * T.dot(utils.make_PSD(inner_term)).dot(np.transpose(T)) / hhat
            for c in range(0, 2):
                for r in range(0, 2):
                    IJV[0].append(i * 2 + r)
                    IJV[1].append(i * 2 + c)
                    IJV[2] = np.append(IJV[2], local_hess[r, c])
    return IJV
# ANCHOR_END: val_grad_hess