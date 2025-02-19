# Mass-Spring Solids Simulation

import numpy as np  # numpy for linear algebra
import pygame       # pygame for visualization
pygame.init()

import utils
import square_mesh   # square mesh
import time_integrator

# simulation setup
side_len = 1
rho = 2000      # density of square
E = 1e4         # Young's modulus
nu = 0.4        # Poisson's ratio
n_seg = 10       # num of segments per side of the square
h = 0.04        # time step size in s
DBC = []        # no nodes need to be fixed
y_ground = -1   # height of the planar ground

# initialize simulation
[x, e] = square_mesh.generate(side_len, n_seg)  # node positions and edge node indices
v = np.array([[0.0, 0.0]] * len(x))             # velocity
m = [rho * side_len * side_len / ((n_seg + 1) * (n_seg + 1))] * len(x)  # calculate node mass evenly
# ANCHOR: elem_precomp
# rest shape basis, volume, and lame parameters
vol = [0.0] * len(e)
IB = [np.array([[0.0, 0.0]] * 2)] * len(e)
for i in range(0, len(e)):
    TB = [x[e[i][1]] - x[e[i][0]], x[e[i][2]] - x[e[i][0]]]
    vol[i] = np.linalg.det(np.transpose(TB)) / 2
    IB[i] = np.linalg.inv(np.transpose(TB))
mu_lame = [0.5 * E / (1 + nu)] * len(e)
lam = [E * nu / ((1 + nu) * (1 - 2 * nu))] * len(e)
# ANCHOR_END: elem_precomp
# identify whether a node is Dirichlet
is_DBC = [False] * len(x)
for i in DBC:
    is_DBC[i] = True
# ANCHOR: contact_area
contact_area = [side_len / n_seg] * len(x)     # perimeter split to each node
# ANCHOR_END: contact_area
# compute reduced basis using 0: no reduction; 1: polynomial functions; 2: modal reduction
reduced_basis = utils.compute_reduced_basis(x, e, vol, IB, mu_lame, lam, method=1, order=2)

# simulation with visualization
resolution = np.array([900, 900])
offset = resolution / 2
scale = 200
def screen_projection(x):
    return [offset[0] + scale * x[0], resolution[1] - (offset[1] + scale * x[1])]

time_step = 0
square_mesh.write_to_file(time_step, x, e)
screen = pygame.display.set_mode(resolution)
running = True
while running:
    # run until the user asks to quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    print('### Time step', time_step, '###')

    # fill the background and draw the square
    screen.fill((255, 255, 255))
    pygame.draw.aaline(screen, (0, 0, 255), screen_projection([-2, y_ground]), screen_projection([2, y_ground]))   # ground
    for eI in e:
        # ANCHOR: draw_tri
        pygame.draw.aaline(screen, (0, 0, 255), screen_projection(x[eI[0]]), screen_projection(x[eI[1]]))
        pygame.draw.aaline(screen, (0, 0, 255), screen_projection(x[eI[1]]), screen_projection(x[eI[2]]))
        pygame.draw.aaline(screen, (0, 0, 255), screen_projection(x[eI[2]]), screen_projection(x[eI[0]]))
        # ANCHOR_END: draw_tri
    for xI in x:
        pygame.draw.circle(screen, (0, 0, 255), screen_projection(xI), 0.1 * side_len / n_seg * scale)

    pygame.display.flip()   # flip the display

    # step forward simulation and wait for screen refresh
    [x, v] = time_integrator.step_forward(x, e, v, m, vol, IB, mu_lame, lam, y_ground, contact_area, is_DBC, reduced_basis, h, 1e-2)
    time_step += 1
    pygame.time.wait(int(h * 1000))
    square_mesh.write_to_file(time_step, x, e)

pygame.quit()