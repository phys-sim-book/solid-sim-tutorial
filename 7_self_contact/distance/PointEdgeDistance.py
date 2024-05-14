# ANCHOR: PE_val_grad
import numpy as np

import distance.PointPointDistance as PP
import distance.PointLineDistance as PL

def val(p, e0, e1):
    e = e1 - e0
    ratio = np.dot(e, p - e0) / np.dot(e, e)
    if ratio < 0:    # point(p)-point(e0) expression
        return PP.val(p, e0)
    elif ratio > 1:  # point(p)-point(e1) expression
        return PP.val(p, e1)
    else:            # point(p)-line(e0e1) expression
        return PL.val(p, e0, e1)

def grad(p, e0, e1):
    e = e1 - e0
    ratio = np.dot(e, p - e0) / np.dot(e, e)
    if ratio < 0:    # point(p)-point(e0) expression
        g_PP = PP.grad(p, e0)
        return np.reshape([g_PP[0:2], g_PP[2:4], np.array([0.0, 0.0])], (1, 6))[0]
    elif ratio > 1:  # point(p)-point(e1) expression
        g_PP = PP.grad(p, e1)
        return np.reshape([g_PP[0:2], np.array([0.0, 0.0]), g_PP[2:4]], (1, 6))[0]
    else:            # point(p)-line(e0e1) expression
        return PL.grad(p, e0, e1)
# ANCHOR_END: PE_val_grad

def hess(p, e0, e1):
    e = e1 - e0
    ratio = np.dot(e, p - e0) / np.dot(e, e)
    if ratio < 0:    # point(p)-point(e0) expression
        H_PP =  PP.hess(p, e0)
        return np.array([np.reshape([H_PP[0, 0:2], H_PP[0, 2:4], np.array([0.0, 0.0])], (1, 6))[0], \
            np.reshape([H_PP[1, 0:2], H_PP[1, 2:4], np.array([0.0, 0.0])], (1, 6))[0], \
            np.reshape([H_PP[2, 0:2], H_PP[2, 2:4], np.array([0.0, 0.0])], (1, 6))[0], \
            np.reshape([H_PP[3, 0:2], H_PP[3, 2:4], np.array([0.0, 0.0])], (1, 6))[0], \
            np.array([0.0] * 6), \
            np.array([0.0] * 6)])
    elif ratio > 1:  # point(p)-point(e1) expression
        H_PP = PP.hess(p, e1)
        return np.array([np.reshape([H_PP[0, 0:2], np.array([0.0, 0.0]), H_PP[0, 2:4]], (1, 6))[0], \
            np.reshape([H_PP[1, 0:2], np.array([0.0, 0.0]), H_PP[1, 2:4]], (1, 6))[0], \
            np.array([0.0] * 6), \
            np.array([0.0] * 6), \
            np.reshape([H_PP[2, 0:2], np.array([0.0, 0.0]), H_PP[2, 2:4]], (1, 6))[0], \
            np.reshape([H_PP[3, 0:2], np.array([0.0, 0.0]), H_PP[3, 2:4]], (1, 6))[0]])
    else:            # point(p)-line(e0e1) expression
        return PL.hess(p, e0, e1)