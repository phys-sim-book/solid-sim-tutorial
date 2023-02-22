import math
import numpy as np

import distance.PointEdgeDistance as PE
import distance.CCD as CCD

import utils

dhat = 0.01
kappa = 1e5

def val(x, n, o, bp, be, contact_area):
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
    # self-contact
    dhat_sqr = dhat * dhat
    for xI in bp:
        for eI in be:
            if xI != eI[0] and xI != eI[1]: # do not consider a point and its incident edge
                d_sqr = PE.val(x[xI], x[eI[0]], x[eI[1]])
                if d_sqr < dhat_sqr:
                    s = d_sqr / dhat_sqr
                    # since d_sqr is used, need to divide by 8 not 2 here for consistency to linear elasticity:
                    sum += contact_area[xI] * dhat * kappa / 8 * (s - 1) * math.log(s)
    return sum

def grad(x, n, o, bp, be, contact_area):
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
    # self-contact
    dhat_sqr = dhat * dhat
    for xI in bp:
        for eI in be:
            if xI != eI[0] and xI != eI[1]: # do not consider a point and its incident edge
                d_sqr = PE.val(x[xI], x[eI[0]], x[eI[1]])
                if d_sqr < dhat_sqr:
                    s = d_sqr / dhat_sqr
                    # since d_sqr is used, need to divide by 8 not 2 here for consistency to linear elasticity:
                    local_grad = contact_area[xI] * dhat * (kappa / 8 * (math.log(s) / dhat_sqr + (s - 1) / d_sqr)) * PE.grad(x[xI], x[eI[0]], x[eI[1]])
                    g[xI] += local_grad[0:2]
                    g[eI[0]] += local_grad[2:4]
                    g[eI[1]] += local_grad[4:6]
    return g

def hess(x, n, o, bp, be, contact_area):
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
    # self-contact
    dhat_sqr = dhat * dhat
    for xI in bp:
        for eI in be:
            if xI != eI[0] and xI != eI[1]: # do not consider a point and its incident edge
                d_sqr = PE.val(x[xI], x[eI[0]], x[eI[1]])
                if d_sqr < dhat_sqr:
                    d_sqr_grad = PE.grad(x[xI], x[eI[0]], x[eI[1]])
                    s = d_sqr / dhat_sqr
                    # since d_sqr is used, need to divide by 8 not 2 here for consistency to linear elasticity:
                    local_hess = contact_area[xI] * dhat * utils.make_PD(kappa / (8 * d_sqr * d_sqr * dhat_sqr) * (d_sqr + dhat_sqr) * np.outer(d_sqr_grad, d_sqr_grad) \
                        + (kappa / 8 * (math.log(s) / dhat_sqr + (s - 1) / d_sqr)) * PE.hess(x[xI], x[eI[0]], x[eI[1]]))
                    index = [xI, eI[0], eI[1]]
                    for nI in range(0, 3):
                        for nJ in range(0, 3):
                            for c in range(0, 2):
                                for r in range(0, 2):
                                    IJV[0].append(index[nI] * 2 + r)
                                    IJV[1].append(index[nJ] * 2 + c)
                                    IJV[2] = np.append(IJV[2], local_hess[nI * 2 + r, nJ * 2 + c])
    return IJV

def init_step_size(x, n, o, bp, be, p):
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
    # self-contact
    for xI in bp:
        for eI in be:
            if xI != eI[0] and xI != eI[1]: # do not consider a point and its incident edge
                if CCD.bbox_overlap(x[xI], x[eI[0]], x[eI[1]], p[xI], p[eI[0]], p[eI[1]], alpha):
                    toc = CCD.narrow_phase_CCD(x[xI], x[eI[0]], x[eI[1]], p[xI], p[eI[0]], p[eI[1]], alpha)
                    if alpha > toc:
                        alpha = toc
    return alpha

def compute_mu_lambda(x, n, o, bp, be, contact_area, mu):
    # floor:
    mu_lambda = np.array([0.0] * len(x))
    for i in range(0, len(x)):
        d = n.dot(x[i] - o)
        if d < dhat:
            s = d / dhat
            mu_lambda[i] = mu * -contact_area[i] * dhat * (kappa / 2 * (math.log(s) / dhat + (s - 1) / d))
    # self-contact
    mu_lambda_self = []
    dhat_sqr = dhat * dhat
    for xI in bp:
        for eI in be:
            if xI != eI[0] and xI != eI[1]: # do not consider a point and its incident edge
                d_sqr = PE.val(x[xI], x[eI[0]], x[eI[1]])
                if d_sqr < dhat_sqr:
                    s = d_sqr / dhat_sqr
                    # since d_sqr is used, need to divide by 8 not 2 here for consistency to linear elasticity
                    # also, lambda = -\partial b / \partial d = -(\partial b / \partial d^2) * (\partial d^2 / \partial d)
                    mu_lam = mu * -contact_area[xI] * dhat * (kappa / 8 * (math.log(s) / dhat_sqr + (s - 1) / d_sqr)) * 2 * math.sqrt(d_sqr)
                    [n, r] = PE.tangent(x[xI], x[eI[0]], x[eI[1]]) # normal and closest point parameterization on the edge
                    mu_lambda_self.append([xI, eI[0], eI[1], mu_lam, n, r])
    return [mu_lambda, mu_lambda_self]