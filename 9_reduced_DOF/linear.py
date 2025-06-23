
"""
2D Modal Analysis with Combined Animation of Mode Shapes

This program performs a dynamic modal analysis of a 2D vertical cantilever beam
using 4-node quadrilateral finite elements under a plane stress assumption.
The first 6 eigenmodes are computed and visualized side-by-side in a single
animated GIF, showing the harmonic motion of each mode over time.

Author: Žiga Kovačič
Date: 2025-06-22
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation
import time

# Problem parameters
L = 15.0  # Length (now vertical)
H = 1.0   # Height (now horizontal width)
t = 0.1   # Thickness

# Material properties
E = 5e7   # Young's modulus
nu = 0.3  # Poisson's ratio
rho = 500 # Density

# Mesh discretization
Nel_L = 30 # Elements along the length
Nel_H = 5  # Elements along the height

print(f"Solving with a 2D mesh: {Nel_H}x{Nel_L} elements for a vertical beam.")

def generate_2d_mesh(L_mesh, H_mesh, Nx_mesh, Ny_mesh):
    """Generate structured grid of nodes and quadrilateral elements."""
    x_coords = np.linspace(0, L_mesh, Nx_mesh + 1)
    y_coords = np.linspace(0, H_mesh, Ny_mesh + 1)
    nodes_x, nodes_y = np.meshgrid(x_coords, y_coords)
    nodes = np.vstack([nodes_x.ravel(), nodes_y.ravel()]).T

    elements = []
    for j in range(Ny_mesh):
        for i in range(Nx_mesh):
            n0, n1 = j * (Nx_mesh + 1) + i, j * (Nx_mesh + 1) + (i + 1)
            n2, n3 = (j + 1) * (Nx_mesh + 1) + i, (j + 1) * (Nx_mesh + 1) + (i + 1)
            elements.append([n0, n1, n3, n2])
    return nodes, np.array(elements)

def get_element_matrices(element_nodes, D, rho, t):
    r"""
    Compute stiffness (K) and mass (M) matrices for a 4-node quad element.
    This function computes the building blocks for the global M and K matrices
    required by the PDF's Equation (2): $M \ddot{u} + D \dot{u} + K u = f$.
    """
    
    # The integrals for k_e and m_e are computed numerically using Gaussian Quadrature.
    # The method approximates an integral over the standard interval [-1, 1] as:
    # $\int_{-1}^{1} g(x) dx \approx \sum_{i=1}^{n} w_i g(x_i)$
    # We use the 2-point Gauss-Legendre rule (n=2), which is exact for polynomials up to degree 2n-1=3.

    # Gauss point coordinate $\xi_i, \eta_i = \pm 1/\sqrt{3}$ for 2x2 numerical integration.
    gp = 1.0 / np.sqrt(3) 

    gauss_points = np.array([[-gp, -gp], [gp, -gp], [gp, gp], [-gp, gp]])

    # Initialize element stiffness $k_e$ and mass $m_e$ matrices
    k_e, m_e = np.zeros((8, 8)), np.zeros((8, 8))

    # loop over the 4 Gauss points to perform numerical integration.
    for i in range(4):
        xi, eta = gauss_points[i]

        # derivatives of shape functions w.r.t. natural coords $(\xi,\eta)$:
        # dN_dxi_eta corresponds to the matrix $[\partial N / \partial \xi, \partial N / \partial \eta]^T$
        dN_dxi_eta = 0.25 * np.array([
            [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
            [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]
        ])

        # Jacobian matrix: $J = \frac{\partial(x,y)}{\partial(\xi,\eta)}$
        J = dN_dxi_eta @ element_nodes
        # $dA = \det(J) d\xi d\eta$
        detJ = np.linalg.det(J)
        # derivatives of shape functions w.r.t. real coords $(x,y)$: $[\partial N / \partial x, \partial N / \partial y]^T = J^{-1} [\partial N / \partial \xi, \partial N / \partial \eta]^T$
        dN_dxy = np.linalg.inv(J) @ dN_dxi_eta

        # Form the strain-displacement matrix $B$, which defines strain: $\varepsilon = B d$
        B = np.zeros((3, 8))
        for j in range(4):
            B[0, 2*j], B[1, 2*j + 1] = dN_dxy[0, j], dN_dxy[1, j]
            B[2, 2*j], B[2, 2*j + 1] = dN_dxy[1, j], dN_dxy[0, j]

        # Approximate the stiffness matrix integral: $k_e = \int_A (B^T D B) t \, dA$
        # by calculating one term in the Gauss quadrature sum: $(B^T D B) \cdot (t \cdot \det(J)) \cdot w_i$, where again $w_i=1$.
        k_e += B.T @ D @ B * detJ * t

        # Shape function matrix $N$, for interpolating displacement: $u(x,y) = N d$
        N_shape = 0.25 * (1 + np.array([-1, 1, 1, -1])*xi) * (1 + np.array([-1, -1, 1, 1])*eta)
        N_mat = np.zeros((2, 8))
        for j in range(4):
            N_mat[0, 2*j] = N_mat[1, 2*j+1] = N_shape[j]
        
        # Approximate the mass matrix integral: $m_e = \int_A (\rho N^T N) t \, dA$
        # by calculating one term in the sum: $(\rho N^T N) \cdot (t \cdot \det(J)) \cdot w_i$, where again $w_i=1$.
        m_e += rho * t * N_mat.T @ N_mat * detJ

    return k_e, m_e

start_time = time.time()

# Generate mesh for a vertical beam (width H, length L)
nodes, elements = generate_2d_mesh(H, L, Nel_H, Nel_L)
n_nodes, n_dof = nodes.shape[0], 2 * nodes.shape[0]

# Plane stress constitutive matrix, defines stress-strain relationship
D = (E / (1 - nu**2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])

K_global = lil_matrix((n_dof, n_dof))
M_global = lil_matrix((n_dof, n_dof))

print("Assembling global matrices...")
for el_nodes_indices in elements:
    k_e, m_e = get_element_matrices(nodes[el_nodes_indices], D, rho, t)
    dof_indices = np.ravel([[2*n, 2*n+1] for n in el_nodes_indices])
    K_global[np.ix_(dof_indices, dof_indices)] += k_e
    M_global[np.ix_(dof_indices, dof_indices)] += m_e
 
K_global, M_global = K_global.tocsr(), M_global.tocsr() # Convert to CSR format for efficient operations

# Apply boundary conditions 
# Vertical beam fixed at the bottom (y=0)
clamped_nodes = np.where(nodes[:, 1] < 1e-9)[0]
clamped_dofs = np.ravel([[2*n, 2*n+1] for n in clamped_nodes])
free_dofs = np.setdiff1d(np.arange(n_dof), clamped_dofs)

# Compute reduced matrices
K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
M_reduced = M_global[np.ix_(free_dofs, free_dofs)]

# Solve eigenvalue problem
# Equation: K_reduced * psi = lambda * M_reduced * psi
N_eig = 8
print(f"Solving eigenvalue problem for {N_eig} modes...")

eigenvalues, eigenvectors_reduced = eigsh(K_reduced, k=N_eig, M=M_reduced, sigma=0, which='LM')
angular_freq = np.sqrt(np.abs(eigenvalues))
frequencies = angular_freq / (2 * np.pi) 

# Reconstruct full eigenvectors, zero out fixed DOFs
eigenvectors = np.zeros((n_dof, N_eig)) 
eigenvectors[free_dofs, :] = eigenvectors_reduced

assembly_time = time.time() - start_time
print(f"Assembly and solution took {assembly_time:.2f} seconds.")

print("\nComputed Frequencies (Hz):")
for i in range(N_eig):
    print(f"Mode {i+1}: {frequencies[i]:.2f} Hz")


# ---------------------
# --- Visualization ---
# ---------------------
print("\nGenerating combined mode animation...")

num_modes_to_animate = 6
animation_duration = 5.0
fps = 30
num_frames = int(animation_duration * fps)
displacement_scale = L * 0.15 # Scale displacements for better visibility

fig, axes = plt.subplots(1, num_modes_to_animate, figsize=(15, 8))

triangles = []
for el in elements:
    triangles.append([el[0], el[1], el[3]])
    triangles.append([el[1], el[2], el[3]])
triang = tri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)

all_deformed_nodes = []
for i in range(num_modes_to_animate):
    if i >= N_eig: break
    psi = eigenvectors[:, i]
    max_disp_shape = (psi.reshape(-1, 2) / np.max(np.abs(psi))) * displacement_scale
    all_deformed_nodes.append(nodes + max_disp_shape)
    all_deformed_nodes.append(nodes - max_disp_shape)

all_deformed_nodes = np.vstack(all_deformed_nodes)
x_min, y_min = all_deformed_nodes.min(axis=0)
x_max, y_max = all_deformed_nodes.max(axis=0)
x_range, y_range = x_max - x_min, y_max - y_min
fixed_xlim = [x_min - 0.1 * x_range, x_max + 0.1 * x_range]
fixed_ylim = [y_min - 0.1 * y_range, y_max + 0.1 * y_range]

vmax_values = []
for i in range(num_modes_to_animate):
    psi = eigenvectors[:, i]
    psi_normalized = psi / np.max(np.abs(psi))
    disp_mag_static = np.sqrt(psi_normalized.reshape(-1, 2)[:, 0]**2 + psi_normalized.reshape(-1, 2)[:, 1]**2)
    vmax_values.append(np.max(disp_mag_static) * displacement_scale)

def animate(frame):
    t_current = frame / fps
    for i, ax in enumerate(axes):
        if i >= N_eig: 
            ax.axis('off')
            continue

        ax.clear()
        ax.set_xlim(fixed_xlim)
        ax.set_ylim(fixed_ylim)
        ax.set_aspect('equal')
        
        ax.triplot(triang, '--', color='gray', linewidth=0.5)
        
        psi = eigenvectors[:, i]
        omega = angular_freq[i]
        
        # Normalize eigenvector so max displacement is displacement_scale
        psi_normalized = psi / np.max(np.abs(psi))
        
        scale_factor = np.sin(omega * t_current)
        current_disp = psi_normalized.reshape(-1, 2) * scale_factor * displacement_scale
        deformed_nodes = nodes + current_disp
        
        disp_mag_instant = np.sqrt(current_disp[:, 0]**2 + current_disp[:, 1]**2)
        
        tpc = ax.tripcolor(deformed_nodes[:, 0], deformed_nodes[:, 1], triang.triangles,
                          facecolors=disp_mag_instant[triang.triangles].mean(axis=1),
                          edgecolors='k', cmap='viridis', linewidth=0.5,
                          vmin=0, vmax=vmax_values[i])
                          
        ax.set_title(f'Mode {i+1}\n{frequencies[i]:.2f} Hz', fontsize=15)
        ax.set_xticks([])
        ax.set_yticks([])

    if axes[0]:
        axes[0].set_ylabel('Y-Position (m)')
    fig.supxlabel('Mode Shapes', y=0.05)
    
    return [artist for ax in axes for artist in ax.get_children() if isinstance(artist, plt.Artist)]

ani = animation.FuncAnimation(fig, animate, frames=num_frames, interval=1000/fps, blit=False)
fig.tight_layout(rect=[0, 0.05, 1, 1]) 

filename = f"modes_1-{num_modes_to_animate}_combined_animation.gif"
print(f"Saving combined animation for Modes 1-{num_modes_to_animate} to '{filename}'...")
ani.save(filename, writer='pillow', fps=fps)
plt.close(fig)

print("All animations generated.")