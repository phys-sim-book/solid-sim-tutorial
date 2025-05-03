# Material Point Method Simulation
import numpy as np  # numpy for linear algebra
import taichi as ti # taichi for fast and parallelized computation 

ti.init(arch=ti.cpu)

# ANCHOR: property_def
# simulation setup
grid_size = 128 # background Eulerian grid's resolution, in 2D is [128, 128]
dx = 1.0 / grid_size # the domain size is [1m, 1m] in 2D, so dx for each cell is (1/128)m
dt = 2e-4 # time step size in second
ppc = 8 # average particle-per-cell

density = 1000 # kg / m^3
E, nu = 1e4, 0.3 # sand's Young's modulus and Poisson's ratio
mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu)) # sand's Lame parameters
# ANCHOR_END: property_def

# ANCHOR: setting
# sampling material particles with uniform sampling (Section 26.1)
def uniform_grid(x0, y0, x1, y1, dx):
    xx, yy = np.meshgrid(np.arange(x0, x1 + dx, dx), np.arange(y0, y1 + dx, dx))
    return np.column_stack((xx.ravel(), yy.ravel()))

box1_samples = uniform_grid(0.2, 0.4, 0.4, 0.6, dx / np.sqrt(ppc))
box1_velocities = np.tile(np.array([10.0, 0]), (len(box1_samples), 1))
box2_samples = uniform_grid(0.6, 0.4, 0.8, 0.6, dx / np.sqrt(ppc))
box2_velocities = np.tile(np.array([-10.0, 0]), (len(box1_samples), 1))
all_samples = np.concatenate([box1_samples, box2_samples], axis=0)
all_velocities = np.concatenate([box1_velocities, box2_velocities], axis=0)
# ANCHOR_END: setting

# ANCHOR: data_def
# material particles data (Section 26.1)
N_particles = len(all_samples)
x = ti.Vector.field(2, float, N_particles) # the position of particles
x.from_numpy(all_samples)
v = ti.Vector.field(2, float, N_particles) # the velocity of particles
v.from_numpy(all_velocities)
vol = ti.field(float, N_particles)         # the volume of particle
vol.fill(0.2 * 0.4 / N_particles) # get the volume of each particle as V_rest / N_particles
m = ti.field(float, N_particles)           # the mass of particle
m.fill(vol[0] * density)
F = ti.Matrix.field(2, 2, float, N_particles)  # the deformation gradient of particles
F.from_numpy(np.tile(np.eye(2), (N_particles, 1, 1)))

# grid data
grid_m = ti.field(float, (grid_size, grid_size))
grid_v = ti.Vector.field(2, float, (grid_size, grid_size))
# ANCHOR_END: data_def

# ANCHOR: reset_grid
def reset_grid():
    # after each transfer, the grid is reset
    grid_m.fill(0)
    grid_v.fill(0)
# ANCHOR_END: reset_grid

################################
# Stvk Hencky Elasticity
################################
# ANCHOR: stvk
@ti.func
def StVK_Hencky_PK1_2D(F):
    U, sig, V = ti.svd(F)
    inv_sig = sig.inverse()
    e = ti.Matrix([[ti.log(sig[0, 0]), 0], [0, ti.log(sig[1, 1])]])
    return U @ (2 * mu * inv_sig @ e + lam * e.trace() * inv_sig) @ V.transpose()
# ANCHOR_END: stvk

# Particle-to-Grid (P2G) Transfers (Section 26.3)
# ANCHOR: p2g
@ti.kernel
def particle_to_grid_transfer():
    for p in range(N_particles):
        base = (x[p] / dx - 0.5).cast(int)
        fx = x[p] / dx - base.cast(float)
        # quadratic B-spline interpolating functions (Section 26.2)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        # gradient of the interpolating function (Section 26.2)
        dw_dx = [fx - 1.5, 2 * (1.0 - fx), fx - 0.5]

        P = StVK_Hencky_PK1_2D(F[p])
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                weight = w[i][0] * w[j][1]
                grad_weight = ti.Vector([(1. / dx) * dw_dx[i][0] * w[j][1], 
                                          w[i][0] * (1. / dx) * dw_dx[j][1]])

                grid_m[base + offset] += weight * m[p] # mass transfer
                grid_v[base + offset] += weight * m[p] * v[p] # momentum Transfer, PIC formulation
                # internal force (stress) transfer
                fi = -vol[p] * P @ F[p].transpose() @ grad_weight
                grid_v[base + offset] += dt * fi
# ANCHOR_END: p2g

# Grid Update (Section 26.3)
# ANCHOR: grid_update
@ti.kernel
def update_grid():
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] = grid_v[i, j] / grid_m[i, j] # extract updated nodal velocity from transferred nodal momentum

            # Dirichlet BC near the bounding box
            if i <= 3 or i > grid_size - 3 or j <= 2 or j > grid_size - 3:
                grid_v[i, j] = 0
# ANCHOR_END: grid_update


# Grid-to-Particle (G2P) Transfers (Section 26.3)
# ANCHOR: g2p
@ti.kernel
def grid_to_particle_transfer():
    for p in range(N_particles):
        base = (x[p] / dx - 0.5).cast(int)
        fx = x[p] / dx - base.cast(float)
        # quadratic B-spline interpolating functions (Section 26.2)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        # gradient of the interpolating function (Section 26.2)
        dw_dx = [fx - 1.5, 2 * (1.0 - fx), fx - 0.5]

        new_v = ti.Vector.zero(float, 2)
        v_grad_wT = ti.Matrix.zero(float, 2, 2)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                weight = w[i][0] * w[j][1]
                grad_weight = ti.Vector([(1. / dx) * dw_dx[i][0] * w[j][1], 
                                          w[i][0] * (1. / dx) * dw_dx[j][1]])

                new_v += weight * grid_v[base + offset]
                v_grad_wT += grid_v[base + offset].outer_product(grad_weight)

        v[p] = new_v
        F[p] = (ti.Matrix.identity(float, 2) + dt * v_grad_wT) @ F[p]
# ANCHOR_END: g2p

# Deformation Gradient and Particle State Update (Section 26.4)
# ANCHOR: particle_update
@ti.kernel
def update_particle_state():
    for p in range(N_particles):
        x[p] += dt * v[p] # advection (Section 26.4)
# ANCHOR_END: particle_update

# ANCHOR: time_step
def step():
    # a single time step of the Material Point Method (MPM) simulation (Section 26.5)
    reset_grid()
    particle_to_grid_transfer()
    update_grid()
    grid_to_particle_transfer()
    update_particle_state()
# ANCHOR_END: time_step

################################
# Main 
################################
gui = ti.GUI("2D MPM Elasticity", res = 512, background_color = 0xFFFFFF)
while True:
    for s in range(50): step()

    gui.circles(x.to_numpy()[:len(box1_samples)], radius=3, color=0xFF0000)
    gui.circles(x.to_numpy()[len(box1_samples):], radius=3, color=0x0000FF)
    gui.show()