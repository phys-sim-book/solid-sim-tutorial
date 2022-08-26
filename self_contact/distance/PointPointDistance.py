import numpy as np

def val(p0, p1):
    e = p0 - p1
    return np.dot(e, e)

def grad(p0, p1):
    e = p0 - p1
    return np.reshape([2 * e, -2 * e], (1, 4))[0]

def hess(p0, p1):
    H = np.array([[0.0] * 4] * 4)
    H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 2
    H[0, 2] = H[1, 3] = H[2, 0] = H[3, 1] = -2
    return H