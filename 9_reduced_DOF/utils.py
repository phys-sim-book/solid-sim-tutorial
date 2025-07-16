import math
import numpy as np
import numpy.linalg as LA

import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh

import NeoHookeanEnergy

def make_PSD(hess):
    [lam, V] = LA.eigh(hess)    # Eigen decomposition on symmetric matrix
    # set all negative Eigenvalues to 0
    for i in range(0, len(lam)):
        lam[i] = max(0, lam[i])
    return np.matmul(np.matmul(V, np.diag(lam)), np.transpose(V))

# ANCHOR: find_positive_real_root
def smallest_positive_real_root_quad(a, b, c, tol = 1e-6):
    # return negative value if no positive real root is found
    t = 0
    if abs(a) <= tol:
        if abs(b) <= tol: # f(x) = c > 0 for all x
            t = -1
        else:
            t = -c / b
    else:
        desc = b * b - 4 * a * c
        if desc > 0:
            t = (-b - math.sqrt(desc)) / (2 * a)
            if t < 0:
                t = (-b + math.sqrt(desc)) / (2 * a)
        else: # desv<0 ==> imag, f(x) > 0 for all x > 0
            t = -1
    return t
# ANCHOR_END: find_positive_real_root

def compute_abd_anchor_basis(x):
    c = x.mean(axis=0)
    diag = np.linalg.norm(x.max(axis=0) - x.min(axis=0))
    scale = diag / math.sqrt(2)
    c -= 1/3 * scale
    anchors = np.stack([c, c + np.asarray([scale, 0]), c + np.asarray([0, scale])], axis=0)

    basis = np.zeros((len(anchors) * 2, 6)) # 1, x, y for both x- and y-displacements
    for i in range(len(anchors)):
        for d in range(2):
            basis[i * 2 + d][d * 3] = 1
            basis[i * 2 + d][d * 3 + 1] = anchors[i][0]
            basis[i * 2 + d][d * 3 + 2] = anchors[i][1]
    return basis

# ANCHOR: compute_reduced_basis
def compute_reduced_basis(x, e, vol, IB, mu_lame, lam, method, order):
    if method == 0: # full basis, no reduction
        basis = np.zeros((len(x) * 2, len(x) * 2))
        for i in range(len(x) * 2):
            basis[i][i] = 1
        return basis
    elif method == 1: # polynomial basis
        if order == 1: # linear basis, or affine basis
            basis = np.zeros((len(x) * 2, 6)) # 1, x, y for both x- and y-displacements
            for i in range(len(x)):
                for d in range(2):
                    basis[i * 2 + d][d * 3] = 1
                    basis[i * 2 + d][d * 3 + 1] = x[i][0]
                    basis[i * 2 + d][d * 3 + 2] = x[i][1]
        elif order == 2: # quadratic polynomial basis 
            basis = np.zeros((len(x) * 2, 12)) # 1, x, y, x^2, xy, y^2 for both x- and y-displacements
            for i in range(len(x)):
                for d in range(2):
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
        if order <= 0 or order >= len(x) * 2:
            print("invalid number of target basis for modal reduction")
            exit()
        IJV = NeoHookeanEnergy.hess(x, e, vol, IB, mu_lame, lam, project_PSD=False)
        H = sparse.coo_matrix((IJV[2], (IJV[0], IJV[1])), shape=(len(x) * 2, len(x) * 2)).tocsr()
        eigenvalues, eigenvectors = eigsh(H, k=order, which='SM') # get 'order' eigenvectors with smallest eigenvalues 
        return eigenvectors
# ANCHOR_END: compute_reduced_basis