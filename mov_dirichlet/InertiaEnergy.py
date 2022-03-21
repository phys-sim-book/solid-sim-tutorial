import numpy as np

def val(x, x_tilde, m):
    sum = 0.0
    for i in range(0, len(x)):
        diff = x[i] - x_tilde[i]
        sum += 0.5 * m[i] * diff.dot(diff)
    return sum

def grad(x, x_tilde, m):
    g = np.array([[0.0, 0.0]] * len(x))
    for i in range(0, len(x)):
        g[i] = m[i] * (x[i] - x_tilde[i])
    return g

def hess(x, x_tilde, m):
    IJV = [[0] * (len(x) * 2), [0] * (len(x) * 2), np.array([0.0] * (len(x) * 2))]
    for i in range(0, len(x)):
        for d in range(0, 2):
            IJV[0][i * 2 + d] = i * 2 + d
            IJV[1][i * 2 + d] = i * 2 + d
            IJV[2][i * 2 + d] = m[i]
    return IJV