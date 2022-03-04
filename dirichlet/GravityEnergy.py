import numpy as np

gravity = [0.0, -9.81]

def val(x, m):
    sum = 0.0
    for i in range(0, len(x)):
        sum += -m[i] * x[i].dot(gravity)
    return sum

def grad(x, m):
    g = np.array([gravity] * len(x))
    for i in range(0, len(x)):
        g[i] *= -m[i]
    return g

# Hessian is 0