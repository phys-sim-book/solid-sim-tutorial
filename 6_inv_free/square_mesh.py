import numpy as np
import os

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

def write_to_file(frameNum, x, e):
    # Check if 'output' directory exists; if not, create it
    if not os.path.exists('output'):
        os.makedirs('output')

    # create obj file
    filename = f"output/{frameNum}.obj"
    with open(filename, 'w') as f:
        # write vertex coordinates
        for row in x:
            f.write(f"v {float(row[0]):.6f} {float(row[1]):.6f} 0.0\n") 
        # write vertex indices for each triangle
        for row in e:
            #NOTE: vertex indices start from 1 in obj file format
            f.write(f"f {row[0] + 1} {row[1] + 1} {row[2] + 1}\n")