import numpy as np
import numpy.linalg as LA

def make_PD(hess):
    [lam, V] = LA.eigh(hess)    # Eigen decomposition on sparse matrix hess
    # set all negative Eigenvalues to 0
    for i in range(0, len(lam)):
        lam[i] = max(0, lam[i])
    return np.matmul(np.matmul(V, np.diag(lam)), np.transpose(V))