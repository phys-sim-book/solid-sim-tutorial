import numpy as np
import utils

def val(x, e, l2, k):
    sum = 0.0
    for i in range(0, len(e)):
        diff = x[e[i][0]] - x[e[i][1]]
        sum += l2[i] * 0.5 * k[i] * (diff.dot(diff) / l2[i] - 1) ** 2
    return sum

def grad(x, e, l2, k):
    g = np.array([[0.0, 0.0]] * len(x))
    for i in range(0, len(e)):
        diff = x[e[i][0]] - x[e[i][1]]
        g_diff = 2 * k[i] * (diff.dot(diff) / l2[i] - 1) * diff
        g[e[i][0]] += g_diff
        g[e[i][1]] -= g_diff
    return g

def hess(x, e, l2, k):
    IJV = [[0] * (len(e) * 16), [0] * (len(e) * 16), np.array([0.0] * (len(e) * 16))]
    for i in range(0, len(e)):
        diff = x[e[i][0]] - x[e[i][1]]
        H_diff = 2 * k[i] / l2[i] * (2 * np.outer(diff, diff) + (diff.dot(diff) - l2[i]) * np.identity(2))
        H_local = utils.make_PD(np.block([[H_diff, -H_diff], [-H_diff, H_diff]]))
        # add to global matrix
        for nI in range(0, 2):
            for nJ in range(0, 2):
                indStart = i * 16 + (nI * 2 + nJ) * 4
                for r in range(0, 2):
                    for c in range(0, 2):
                        IJV[0][indStart + r * 2 + c] = e[i][nI] * 2 + r
                        IJV[1][indStart + r * 2 + c] = e[i][nJ] * 2 + c
                        IJV[2][indStart + r * 2 + c] = H_local[nI * 2 + r, nJ * 2 + c]
    return IJV