# Material Point Method Simulation
import numpy as np  # numpy for linear algebra
import taichi as ti # taichi for fast and parallelized computation 

ti.init(arch=ti.cpu)

# sampling material particles with poisson-disk sampling
################################
# Poisson Disk Sampling Tool
################################
def poisson_disk_sampling(radius, domain_size, k=30):
    """Bridson's algorithm for Poisson-disk sampling in 2D"""
    cell_size = radius / np.sqrt(2)
    grid_shape = (int(domain_size[0] / cell_size) + 1, int(domain_size[1] / cell_size) + 1)
    grid = -np.ones(grid_shape, dtype=int)
    samples = []
    active_list = []

    def in_domain(p):
        return 0 <= p[0] < domain_size[0] and 0 <= p[1] < domain_size[1]

    def get_cell_coords(p):
        return int(p[0] / cell_size), int(p[1] / cell_size)

    def get_nearby_samples(p):
        i, j = get_cell_coords(p)
        neighbors = []
        for di in [-2, -1, 0, 1, 2]:
            for dj in [-2, -1, 0, 1, 2]:
                ni, nj = i + di, j + dj
                if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                    idx = grid[ni, nj]
                    if idx != -1:
                        neighbors.append(samples[idx])
        return neighbors

    # Start with a random point
    first_point = np.array([np.random.uniform(0, domain_size[0]), np.random.uniform(0, domain_size[1])])
    samples.append(first_point)
    active_list.append(0)
    grid[get_cell_coords(first_point)] = 0

    while active_list:
        idx = np.random.choice(active_list)
        base_point = samples[idx]
        found = False
        for _ in range(k):
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(radius, 2 * radius)
            new_point = base_point + r * np.array([np.cos(angle), np.sin(angle)])
            if in_domain(new_point):
                neighbors = get_nearby_samples(new_point)
                if all(np.linalg.norm(new_point - n) >= radius for n in neighbors):
                    samples.append(new_point)
                    active_list.append(len(samples) - 1)
                    grid[get_cell_coords(new_point)] = len(samples) - 1
                    found = True
        if not found:
            active_list.remove(idx)
    return np.array(samples)

# ANCHOR: data_def
# simulation setup
grid_size = 128 # background Eulerian grid's resolution, in 2D is [128, 128]
dx = 1.0 / grid_size # the domain size is [1m, 1m] in 2D, so dx for each cell is (1/128)m
dt = 2e-4 # time step size in second
ppc = 8 # average particle-per-cell

density = 400 # kg / m^3
E, nu = 3.537e5, 0.3 # sand's Young's modulus and Poisson's ratio
mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu)) # sand's Lame parameters
sdf_friction = 0.5 # frictional coefficient of SDF boundary condition
friction_angle_in_degrees = 25.0 # Drucker Prager friction angle
# ANCHOR: D_def
D = (1./4.) * dx * dx # constant D for Quadratic B-spline used for APIC
# ANCHOR_END: D_def

# sampling material particles with poisson-disk sampling
poisson_samples = poisson_disk_sampling(dx / np.sqrt(ppc), [0.2, 0.4]) # simulating a [30cm, 50cm] sand block

# material particles data
N_particles = len(poisson_samples)
x = ti.Vector.field(2, float, N_particles) # the position of particles
x.from_numpy(np.array(poisson_samples) + [0.4, 0.55])
v = ti.Vector.field(2, float, N_particles) # the velocity of particles
vol = ti.field(float, N_particles)         # the volume of particle
vol.fill(0.2 * 0.4 / N_particles) # get the volume of each particle as V_rest / N_particles
m = ti.field(float, N_particles)           # the mass of particle
m.fill(vol[0] * density)
F = ti.Matrix.field(2, 2, float, N_particles)  # the deformation gradient of particles
F.from_numpy(np.tile(np.eye(2), (N_particles, 1, 1)))
C = ti.Matrix.field(2, 2, float, N_particles)  # the affine-matrix of particles

diff_log_J = ti.field(float, N_particles) # tracks changes in the log of the volume gained during extension

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

################################
# Drucker Prager plasticity
################################
# ANCHOR: drucker_prager
@ti.func
def Drucker_Prager_return_mapping(F, diff_log_J):
    dim = ti.static(F.n)
    sin_phi = ti.sin(friction_angle_in_degrees/ 180.0 * ti.math.pi)
    friction_alpha = ti.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)
    U, sig_diag, V = ti.svd(F)
    sig = ti.Vector([ti.max(sig_diag[i,i], 0.05) for i in ti.static(range(dim))])
    epsilon = ti.log(sig)
    epsilon += diff_log_J / dim # volume correction treatment
    trace_epsilon = epsilon.sum()
    shifted_trace = trace_epsilon
    if shifted_trace >= 0:
        for d in ti.static(range(dim)):
            epsilon[d] = 0.
    else:
        epsilon_hat = epsilon - (trace_epsilon / dim)
        epsilon_hat_norm = ti.sqrt(epsilon_hat.dot(epsilon_hat)+1e-8)
        delta_gamma = epsilon_hat_norm + (dim * lam + 2. * mu) / (2. * mu) * (shifted_trace) * friction_alpha
        epsilon -= (ti.max(delta_gamma, 0) / epsilon_hat_norm) * epsilon_hat
    sig_out = ti.exp(epsilon)
    for d in ti.static(range(dim)):
        sig_diag[d,d] = sig_out[d]
    return U @ sig_diag @ V.transpose()
# ANCHOR_END: drucker_prager

# Particle-to-Grid (P2G) Transfers
# ANCHOR: p2g
@ti.kernel
def particle_to_grid_transfer():
    for p in range(N_particles):
        base = (x[p] / dx - 0.5).cast(int)
        fx = x[p] / dx - base.cast(float)
        # quadratic B-spline interpolating functions
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        # gradient of the interpolating function
        dw_dx = [fx - 1.5, 2 * (1.0 - fx), fx - 0.5]

        P = StVK_Hencky_PK1_2D(F[p])
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[i][0] * w[j][1]
                grad_weight = ti.Vector([(1. / dx) * dw_dx[i][0] * w[j][1], 
                                          w[i][0] * (1. / dx) * dw_dx[j][1]])

                grid_m[base + offset] += weight * m[p] # mass transfer
                # ANCHOR: apic_p2g
                grid_v[base + offset] += weight * m[p] * (v[p] + C[p] @ dpos) # momentum Transfer, APIC formulation
                # ANCHOR_END: apic_p2g
                # internal force (stress) transfer
                fi = -vol[p] * P @ F[p].transpose() @ grad_weight
                grid_v[base + offset] += dt * fi
# ANCHOR_END: p2g

# Grid Update
@ti.kernel
def update_grid():
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] = grid_v[i, j] / grid_m[i, j] # extract updated nodal velocity from transferred nodal momentum
            grid_v[i, j] += ti.Vector([0, -9.81]) * dt # gravity

            # Dirichlet BC near the bounding box
            if i <= 3 or i > grid_size - 3 or j <= 2 or j > grid_size - 3:
                grid_v[i, j] = 0
            
            x_i = (ti.Vector([i, j])) * dx # position of the grid-node
            
            # ANCHOR: sphere_sdf
            # a sphere SDF as boundary condition
            sphere_center = ti.Vector([0.5, 0.5])
            sphere_radius = 0.05 + dx # add a dx-gap to avoid penetration
            if (x_i - sphere_center).norm() < sphere_radius:
                normal = (x_i - sphere_center).normalized()
                diff_vel = -grid_v[i, j]
                dotnv = normal.dot(diff_vel)
                dotnv_frac = dotnv * (1.0 - sdf_friction)
                grid_v[i, j] += diff_vel * sdf_friction + normal * dotnv_frac
            # ANCHOR_END: sphere_sdf


# Grid-to-Particle (G2P) Transfers
# ANCHOR: g2p
@ti.kernel
def grid_to_particle_transfer():
    for p in range(N_particles):
        base = (x[p] / dx - 0.5).cast(int)
        fx = x[p] / dx - base.cast(float)
        # quadratic B-spline interpolating functions 
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        # gradient of the interpolating function
        dw_dx = [fx - 1.5, 2 * (1.0 - fx), fx - 0.5]

        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        v_grad_wT = ti.Matrix.zero(float, 2, 2)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[i][0] * w[j][1]
                grad_weight = ti.Vector([(1. / dx) * dw_dx[i][0] * w[j][1], 
                                          w[i][0] * (1. / dx) * dw_dx[j][1]])

                new_v += weight * grid_v[base + offset]
                # ANCHOR: apic_g2p
                new_C += weight * grid_v[base + offset].outer_product(dpos) / D
                # ANCHOR_END: apic_g2p
                v_grad_wT += grid_v[base + offset].outer_product(grad_weight)

        v[p] = new_v
        C[p] = new_C
        # note the updated F here is the trial elastic deformation gradient
        F[p] = (ti.Matrix.identity(float, 2) + dt * v_grad_wT) @ F[p]
# ANCHOR_END: g2p

# Deformation Gradient and Particle State Update
# ANCHOR: particle_update
@ti.kernel
def update_particle_state():
    for p in range(N_particles):
        # trial elastic deformation gradient
        F_tr = F[p]
        # apply return mapping to correct the trial elastic state, projecting the stress induced by F_tr
        # back onto the yield surface, following the direction specified by the plastic flow rule.
        new_F = Drucker_Prager_return_mapping(F_tr, diff_log_J[p])
        # track the volume change incurred by return mapping to correct volume, following https://dl.acm.org/doi/10.1145/3072959.3073651 sec 4.3.4
        diff_log_J[p] += -ti.log(new_F.determinant()) + ti.log(F_tr.determinant()) 
        F[p] = new_F
        # advection
        x[p] += dt * v[p]
# ANCHOR_END: particle_update

# ANCHOR: time_step
def step():
    # a single time step of the Material Point Method (MPM) simulation
    reset_grid()
    particle_to_grid_transfer()
    update_grid()
    grid_to_particle_transfer()
    update_particle_state()
# ANCHOR_END: time_step

################################
# Main 
################################
gui = ti.GUI("2D MPM Sand", res = 512, background_color = 0xFFFFFF)
frame = 0
while True:
    for s in range(50):
        step()

    gui.circles(np.array([[0.5, 0.5]]), radius = 0.05 * gui.res[0], color = 0xFF0000)
    gui.circles(x.to_numpy(), radius = 1.5, color = 0xD6B588)
    gui.show()