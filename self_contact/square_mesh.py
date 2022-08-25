import numpy as np

def generate(side_length, n_seg):
    # sample nodes uniformly on a square
    x = np.array([[0.0, 0.0]] * ((n_seg + 1) ** 2))
    step = side_length / n_seg
    for i in range(0, n_seg + 1):
        for j in range(0, n_seg + 1):
            x[i * (n_seg + 1) + j] = [-side_length / 2 + i * step, -side_length / 2 + j * step]
    
    # connect the nodes with triangle elements
    e = []
    for i in range(0, n_seg):
        for j in range(0, n_seg):
            # triangulate each cell following a symmetric pattern:
            if (i % 2)^(j % 2):
                e.append([i * (n_seg + 1) + j, (i + 1) * (n_seg + 1) + j, i * (n_seg + 1) + j + 1])
                e.append([(i + 1) * (n_seg + 1) + j, (i + 1) * (n_seg + 1) + j + 1, i * (n_seg + 1) + j + 1])
            else:
                e.append([i * (n_seg + 1) + j, (i + 1) * (n_seg + 1) + j, (i + 1) * (n_seg + 1) + j + 1])
                e.append([i * (n_seg + 1) + j, (i + 1) * (n_seg + 1) + j + 1, i * (n_seg + 1) + j + 1])

    return [x, e]

def find_boundary(e):
    # index all half-edges for fast query
    edge_set = set()
    for i in range(0, len(e)):
        for j in range(0, 3):
            edge_set.add((e[i][j], e[i][(j + 1) % 3]))

    # find boundary points and edges
    bp_set = set()
    be = []
    for eI in edge_set:
        if (eI[1], eI[0]) not in edge_set:
            # if the inverse edge of a half-edge does not exist,
            # then it is a boundary edge
            be.append([eI[0], eI[1]])
            bp_set.add(eI[0])
            bp_set.add(eI[1])
    return [list(bp_set), be]