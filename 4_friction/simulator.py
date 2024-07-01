# Mass-Spring Solids Simulation

import numpy as np  # numpy for linear algebra
import pygame       # pygame for visualization
pygame.init()

import square_mesh   # square mesh
import time_integrator

# simulation setup
side_len = 1
rho = 1000      # density of square
k = 4e4         # spring stiffness
n_seg = 10       # num of segments per side of the square
h = 0.01        # time step size in s
DBC = []        # no nodes need to be fixed
# ANCHOR: slope_setup
ground_n = np.array([0.1, 1.0])     # normal of the slope
ground_n /= np.linalg.norm(ground_n)    # normalize ground normal vector just in case
ground_o = np.array([0.0, -1.0])    # a point on the slope  
# ANCHOR_END: slope_setup
# ANCHOR: set_mu
mu = 0.11        # friction coefficient of the slope
# ANCHOR_END: set_mu

# initialize simulation
[x, e] = square_mesh.generate(side_len, n_seg)  # node positions and edge node indices
v = np.array([[0.0, 0.0]] * len(x))             # velocity
m = [rho * side_len * side_len / ((n_seg + 1) * (n_seg + 1))] * len(x)  # calculate node mass evenly
# rest length squared
l2 = []
for i in range(0, len(e)):
    diff = x[e[i][0]] - x[e[i][1]]
    l2.append(diff.dot(diff))
k = [k] * len(e)    # spring stiffness
# identify whether a node is Dirichlet
is_DBC = [False] * len(x)
for i in DBC:
    is_DBC[i] = True
contact_area = [side_len / n_seg] * len(x)     # perimeter split to each node

# simulation with visualization
resolution = np.array([900, 900])
offset = resolution / 2
scale = 200
def screen_projection(x):
    return [offset[0] + scale * x[0], resolution[1] - (offset[1] + scale * x[1])]

time_step = 0
square_mesh.write_to_file(time_step, x, n_seg)
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
    # ANCHOR: slope_vis
    pygame.draw.aaline(screen, (0, 0, 255), screen_projection([ground_o[0] - 3.0 * ground_n[1], ground_o[1] + 3.0 * ground_n[0]]), 
        screen_projection([ground_o[0] + 3.0 * ground_n[1], ground_o[1] - 3.0 * ground_n[0]]))   # slope
    # ANCHOR_END: slope_vis
    for eI in e:
        pygame.draw.aaline(screen, (0, 0, 255), screen_projection(x[eI[0]]), screen_projection(x[eI[1]]))
    for xI in x:
        pygame.draw.circle(screen, (0, 0, 255), screen_projection(xI), 0.1 * side_len / n_seg * scale)

    pygame.display.flip()   # flip the display

    # step forward simulation and wait for screen refresh
    [x, v] = time_integrator.step_forward(x, e, v, m, l2, k, ground_n, ground_o, contact_area, mu, is_DBC, h, 1e-2)
    time_step += 1
    pygame.time.wait(int(h * 1000))
    square_mesh.write_to_file(time_step, x, n_seg)

pygame.quit()