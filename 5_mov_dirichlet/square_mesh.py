import numpy as np

def generate(side_length, n_seg):
    # sample nodes uniformly on a square
    x = np.array([[0.0, 0.0]] * ((n_seg + 1) ** 2))
    step = side_length / n_seg
    for i in range(0, n_seg + 1):
        for j in range(0, n_seg + 1):
            x[i * (n_seg + 1) + j] = [-side_length / 2 + i * step, -side_length / 2 + j * step]
    
    # connect the nodes with edges
    e = []
    # horizontal edges
    for i in range(0, n_seg):
        for j in range(0, n_seg + 1):
            e.append([i * (n_seg + 1) + j, (i + 1) * (n_seg + 1) + j])
    # vertical edges
    for i in range(0, n_seg + 1):
        for j in range(0, n_seg):
            e.append([i * (n_seg + 1) + j, i * (n_seg + 1) + j + 1])
    # diagonals
    for i in range(0, n_seg):
        for j in range(0, n_seg):
            e.append([i * (n_seg + 1) + j, (i + 1) * (n_seg + 1) + j + 1])
            e.append([(i + 1) * (n_seg + 1) + j, i * (n_seg + 1) + j + 1])

    return [x, e]