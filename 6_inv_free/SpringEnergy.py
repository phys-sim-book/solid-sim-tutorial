import numpy as np

def val(x, m, DBC, DBC_target, k):
    sum = 0.0
    for i in range(0, len(DBC)):
        diff = x[DBC[i]] - DBC_target[i]
        sum += 0.5 * k * m[DBC[i]] * diff.dot(diff)
    return sum

def grad(x, m, DBC, DBC_target, k):
    g = np.array([[0.0, 0.0]] * len(x))
    for i in range(0, len(DBC)):
        g[DBC[i]] = k * m[DBC[i]] * (x[DBC[i]] - DBC_target[i])
    return g

def hess(x, m, DBC, DBC_target, k):
    IJV = [[0] * 0, [0] * 0, np.array([0.0] * 0)]
    for i in range(0, len(DBC)):
        for d in range(0, 2):
            IJV[0].append(DBC[i] * 2 + d)
            IJV[1].append(DBC[i] * 2 + d)
            IJV[2] = np.append(IJV[2], k * m[DBC[i]])
    return IJV