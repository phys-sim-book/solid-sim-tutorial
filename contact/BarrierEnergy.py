import math
import numpy as np

y_ground = -1
dhat = 0.01
kappa = 1e5

def val(x, contact_area):
    sum = 0.0
    for i in range(0, len(x)):
        d = x[i][1] - y_ground
        if d < dhat:
            s = d / dhat
            sum += contact_area[i] * dhat * kappa / 2 * (s - 1) * math.log(s)
    return sum

def grad(x, contact_area):
    g = np.array([[0.0, 0.0]] * len(x))
    for i in range(0, len(x)):
        d = x[i][1] - y_ground
        if d < dhat:
            s = d / dhat
            g[i][1] = contact_area[i] * dhat * (kappa / 2 * (math.log(s) / dhat + (s - 1) / d))
    return g

def hess(x, contact_area):
    IJV = [[0] * len(x), [0] * len(x), np.array([0.0] * len(x))]
    for i in range(0, len(x)):
        IJV[0][i] = i * 2 + 1
        IJV[1][i] = i * 2 + 1
        d = x[i][1] - y_ground
        if d < dhat:
            IJV[2][i] = contact_area[i] * dhat * kappa / (2 * d * d * dhat) * (d + dhat)
        else:
            IJV[2][i] = 0.0
    return IJV

def init_step_size(x, p):
    alpha = 1
    for i in range(0, len(x)):
        if p[i][1] < 0:
            alpha = min(alpha, 0.9 * (y_ground - x[i][1]) / p[i][1])
    return alpha