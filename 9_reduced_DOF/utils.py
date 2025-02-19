import numpy as np
import numpy.linalg as LA

import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh

import MassSpringEnergy

def make_PSD(hess):
    [lam, V] = LA.eigh(hess)    # Eigen decomposition on symmetric matrix
    # set all negative Eigenvalues to 0
    for i in range(0, len(lam)):
        lam[i] = max(0, lam[i])
    return np.matmul(np.matmul(V, np.diag(lam)), np.transpose(V))

def compute_reduced_basis(x, e, l2, k, is_DBC, method, order):
    if method == 0: # polynomial basis
        if order == 1: # linear basis, or affine basis
            basis = np.zeros((len(x) * 2, 6)) # 1, x, y for both x- and y-displacements
            for i in range(len(x)):
                for d in range(2):
                    if not is_DBC[i]: # ignore the floor DOF
                        basis[i * 2 + d][d * 3] = 1
                        basis[i * 2 + d][d * 3 + 1] = x[i][0]
                        basis[i * 2 + d][d * 3 + 2] = x[i][1]
        elif order == 2: # quadratic polynomial basis 
            basis = np.zeros((len(x) * 2, 12)) # 1, x, y, x^2, xy, y^2 for both x- and y-displacements
            for i in range(len(x)):
                for d in range(2):
                    if not is_DBC[i]: # ignore the floor DOF
                        basis[i * 2 + d][d * 6] = 1
                        basis[i * 2 + d][d * 6 + 1] = x[i][0]
                        basis[i * 2 + d][d * 6 + 2] = x[i][1]
                        basis[i * 2 + d][d * 6 + 3] = x[i][0] * x[i][0]
                        basis[i * 2 + d][d * 6 + 4] = x[i][0] * x[i][1]
                        basis[i * 2 + d][d * 6 + 5] = x[i][1] * x[i][1]
        elif order == 3: # cubic polynomial basis
            basis = np.zeros((len(x) * 2, 20)) # 1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3 for both x- and y-displacements
            for i in range(len(x)):
                for d in range(2):
                    if not is_DBC[i]: # ignore the floor DOF
                        basis[i * 2 + d][d * 10] = 1
                        basis[i * 2 + d][d * 10 + 1] = x[i][0]
                        basis[i * 2 + d][d * 10 + 2] = x[i][1]
                        basis[i * 2 + d][d * 10 + 3] = x[i][0] * x[i][0]
                        basis[i * 2 + d][d * 10 + 4] = x[i][0] * x[i][1]
                        basis[i * 2 + d][d * 10 + 5] = x[i][1] * x[i][1]
                        basis[i * 2 + d][d * 10 + 6] = x[i][0] * x[i][0] * x[i][0]
                        basis[i * 2 + d][d * 10 + 7] = x[i][0] * x[i][0] * x[i][1]
                        basis[i * 2 + d][d * 10 + 8] = x[i][0] * x[i][1] * x[i][1]
                        basis[i * 2 + d][d * 10 + 9] = x[i][1] * x[i][1] * x[i][1]
        else:
            print("unsupported order of polynomial basis for reduced DOF")
            exit()
        return basis
    else: # modal-order reduction
        IJV = MassSpringEnergy.hess(x, e, l2, k) #TODO: no SPD projection, switch to NH
        H = sparse.coo_matrix((IJV[2], (IJV[0], IJV[1])), shape=(len(x) * 2, len(x) * 2)).tocsr()
        for i, j in zip(*H.nonzero()):
            if is_DBC[int(i / 2)] | is_DBC[int(j / 2)]: 
                H[i, j] = (i == j) # ignore the floor DOF
        eigenvalues, eigenvectors = eigsh(H, k=order, which='SM') # get 'order' eigenvectors with smallest eigenvalues 
        return eigenvectors