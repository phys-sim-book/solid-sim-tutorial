import numpy as np
import numpy.linalg as LA
import math

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
            t = (-b - math.sqrt(desc)) / (2 * a) # if a > 0, this is either the smaller positive root, or both roots are negative; 
            # if a < 0, there are 1 negative and 1 positive real roots, and we just need the positive one.
            if t < 0:
                t = (-b + math.sqrt(desc)) / (2 * a)
        else: # desv<0 ==> imag, f(x) > 0 for all x > 0
            t = -1
    return t
# ANCHOR_END: find_positive_real_root