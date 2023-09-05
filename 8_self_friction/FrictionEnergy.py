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

def val(v, mu_lambda, mu_lambda_self, hhat, n):
    sum = 0.0
    # floor:
    T = np.identity(2) - np.outer(n, n) # tangent of slope is constant
    for i in range(0, len(v)):
        if mu_lambda[i] > 0:
            vbar = np.transpose(T).dot(v[i])
            sum += mu_lambda[i] * f0(np.linalg.norm(vbar), epsv, hhat)
    # self-contact:
    for i in range(0, len(mu_lambda_self)):
        [xI, eI0, eI1, mu_lam, n, r] = mu_lambda_self[i]
        T = np.identity(2) - np.outer(n, n)
        rel_v = v[xI] - ((1 - r) * v[eI0] + r * v[eI1])
        vbar = np.transpose(T).dot(rel_v)
        sum += mu_lam * f0(np.linalg.norm(vbar), epsv, hhat)
    return sum

def grad(v, mu_lambda, mu_lambda_self, hhat, n):
    g = np.array([[0.0, 0.0]] * len(v))
    # floor:
    T = np.identity(2) - np.outer(n, n) # tangent of slope is constant
    for i in range(0, len(v)):
        if mu_lambda[i] > 0:
            vbar = np.transpose(T).dot(v[i])
            g[i] = mu_lambda[i] * f1_div_vbarnorm(np.linalg.norm(vbar), epsv) * T.dot(vbar)
    # self-contact:
    for i in range(0, len(mu_lambda_self)):
        [xI, eI0, eI1, mu_lam, n, r] = mu_lambda_self[i]
        T = np.identity(2) - np.outer(n, n)
        rel_v = v[xI] - ((1 - r) * v[eI0] + r * v[eI1])
        vbar = np.transpose(T).dot(rel_v)
        g_rel_v = mu_lam * f1_div_vbarnorm(np.linalg.norm(vbar), epsv) * T.dot(vbar)
        g[xI] += g_rel_v
        g[eI0] += g_rel_v * -(1 - r)
        g[eI1] += g_rel_v * -r
    return g

def hess(v, mu_lambda, mu_lambda_self, hhat, n):
    IJV = [[0] * 0, [0] * 0, np.array([0.0] * 0)]
    # floor:
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
    # self-contact:
    for i in range(0, len(mu_lambda_self)):
        [xI, eI0, eI1, mu_lam, n, r] = mu_lambda_self[i]
        T = np.identity(2) - np.outer(n, n)
        rel_v = v[xI] - ((1 - r) * v[eI0] + r * v[eI1])
        vbar = np.transpose(T).dot(rel_v)
        vbarnorm = np.linalg.norm(vbar)
        inner_term = f1_div_vbarnorm(vbarnorm, epsv) * np.identity(2)
        if vbarnorm != 0:
            inner_term += f_hess_term(vbarnorm, epsv) / vbarnorm * np.outer(vbar, vbar)
        hess_rel_v = mu_lam * T.dot(utils.make_PSD(inner_term)).dot(np.transpose(T)) / hhat
        index = [xI, eI0, eI1]
        d_rel_v_dv = [1, -(1 - r), -r]
        for nI in range(0, 3):
            for nJ in range(0, 3):
                for c in range(0, 2):
                    for r in range(0, 2):
                        IJV[0].append(index[nI] * 2 + r)
                        IJV[1].append(index[nJ] * 2 + c)
                        IJV[2] = np.append(IJV[2], d_rel_v_dv[nI] * d_rel_v_dv[nJ] * hess_rel_v[r, c])
    return IJV