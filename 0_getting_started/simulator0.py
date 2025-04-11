# Free Fall Simulation

import math
import numpy as np  # for vector data structure and computations
import pygame       # for visualization

# simulation setup
x = np.array([0.0, 0.0])    # position of particle
v = np.array([0.0, 0.0])    # velocity of particle
g = np.array([0.0, -10.0])  # gravitational acceleration
h = 0.01                    # time step size in seconds

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
        # render a floor at y=-1, and render the particle as a circle:
        screen.fill((255, 255, 255))    
        pygame.display.set_caption('Current time: ' + f'{time_step * h: .2f}s')
        pygame.draw.aaline(screen, (0, 0, 255), screen_projection([-2, -1]), screen_projection([2, -1]))
        pygame.draw.circle(screen, (0, 0, 255), screen_projection(x), 0.1 * scale)
        pygame.display.flip()   # flip the display
        pygame.time.wait(int(1000.0 / render_FPS))  # wait to render the next frame

    # pause the simulation when the particle touches on the ground 
    if x[1] <= -1:
        input()
        break

    # step forward the simulation by updating particle velocity and position
    v += h * g
    x += h * v

    time_step += 1  # update time step counter