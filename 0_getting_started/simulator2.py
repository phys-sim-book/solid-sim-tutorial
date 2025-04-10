# Mass-Spring Solid Simulation

import math
import numpy as np  # for vector data structure and computations
import pygame       # for visualization
import square_mesh  # for generating a square mesh

# simulation setup
m = 1000                    # mass of each particle
side_length = 1             # side length of the square
n_seg = 4                   # number of springs per side of the square
[x, e] = square_mesh.generate(side_length, n_seg)   # array of particle positions and springs   ###
v = np.array([[0.0, 0.0]] * len(x))     # velocity array of particles ###
g = np.array([0, -9.81])    # gravitational acceleration
spring_rest_len = []        # rest length array of the springs ###
for i in range(0, len(e)):  # calculate the rest length of each spring
    spring_vec = x[e[i][0]] - x[e[i][1]]    # the vector connecting two ends of spring i
    spring_rest_len.append(math.sqrt(spring_vec[0] * spring_vec[0] + spring_vec[1] * spring_vec[1]))
spring_stiffness = 1000     # stiffness of the spring
h = 0.1                     # time step size in seconds

# visualization/rendering setup
pygame.init()
render_FPS = 100                    # number of frames to render per second
resolution = np.array([900, 900])   # visualization window size in pixels
offset = resolution / 2             # offset between window coordinates and simulated coordinates
scale = 200                         # scale between window coordinates and simulated coordinates
def screen_projection(x):           # convert simulated coordinates to window coordinates
    return [offset[0] + scale * x[0], resolution[1] - (offset[1] + scale * x[1])]
screen = pygame.display.set_mode(resolution)    # initialize visualizer

time_step = 0   # the number of the current time step
running = True  # flag indicating whether the simulation is still running
while running:
    # run until the user asks to quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # update the frame to display according to render_FPS
    if time_step % int(math.ceil((1.0 / render_FPS) / h)) == 0:
        # fill the background with white color, display simulation time at the top,
        # draw each spring segment, and render each particle as a circle:
        screen.fill((255, 255, 255))    
        pygame.display.set_caption('Current time: ' + f'{time_step * h: .2f}s')
        for i in range(0, len(e)):  ###
            pygame.draw.aaline(screen, (0, 0, 255), screen_projection(x[e[i][0]]), screen_projection(x[e[i][1]]))
        for i in range(0, len(x)):  ###
            pygame.draw.circle(screen, (0, 0, 255), screen_projection(x[i]), 0.02 * scale)
        pygame.display.flip()   # flip the display
        pygame.time.wait(int(1000.0 / render_FPS))  # wait to render the next frame

    # step forward the simulation by updating particle velocity and position ###
    for i in range(0, len(e)):
        # calculate elasticity force of spring i:
        spring_vec = x[e[i][0]] - x[e[i][1]]
        spring_cur_len = math.sqrt(spring_vec[0] * spring_vec[0] + spring_vec[1] * spring_vec[1])
        spring_displacement = spring_cur_len - spring_rest_len[i]
        spring_force = -spring_stiffness * spring_displacement * (spring_vec / spring_cur_len)
        # update the velocity of the two ends of spring i
        v[e[i][0]] += h * (g + spring_force) / m
        v[e[i][1]] += h * (g - spring_force) / m
    # fix the top left and top right corner by setting velocity to 0:
    v[n_seg] = v[(n_seg + 1) * (n_seg + 1) - 1] = np.array([0, 0])
    # update the position of each particle:
    for i in range(0, len(x)):
        x[i] += h * v[i]

    time_step += 1  # update time step counter