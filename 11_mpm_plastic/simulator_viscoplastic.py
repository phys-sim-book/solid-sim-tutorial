import taichi as ti
import os
import shutil

ti.init(arch=ti.cuda)

quality = 6
n_particles, n_grid = 9000 * quality**2, 256 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol, p_rho = (dx * 0.5) ** 2, 1
p_mass = p_vol * p_rho

nu = 0.2
E_toothpaste = 350.0
mu_toothpaste = E_toothpaste / (2 * (1 + nu))
lam_toothpaste = E_toothpaste * nu / ((1 + nu) * (1 - 2 * nu))

yield_stress_bar1 = 1.2
plastic_viscosity_bar1 = 0.7
yield_stress_bar2 = 1.0
plastic_viscosity_bar2 = 0.3

ground_friction = 0.3

x = ti.Vector.field(2, dtype=float, shape=n_particles)
v = ti.Vector.field(2, dtype=float, shape=n_particles)
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)
color_stripe = ti.field(dtype=int, shape=n_particles)
yield_stress = ti.field(dtype=float, shape=n_particles)
plastic_viscosity = ti.field(dtype=float, shape=n_particles)
grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))
gravity = ti.Vector.field(2, dtype=float, shape=())


@ti.func
def viscoplastic_return_mapping_stvk_2d(F_trial, mu_p, lam_p, yield_stress_p, plastic_viscosity_p, dt_val):
    """Viscoplastic return mapping for toothpaste-like material (2D version)"""
    U, sig, V = ti.svd(F_trial)
    
    # Clamp singular values to prevent numerical issues
    sig0 = ti.max(sig[0, 0], 0.01)
    sig1 = ti.max(sig[1, 1], 0.01)
    
    # Compute logarithmic strain (2D)
    epsilon = ti.Vector([ti.log(sig0), ti.log(sig1)])
    trace_epsilon = epsilon[0] + epsilon[1]
    
    # Deviatoric strain
    epsilon_hat = epsilon - ti.Vector([trace_epsilon / 2.0, trace_epsilon / 2.0])
    
    # Deviatoric stress in logarithmic strain space
    s_trial = 2.0 * mu_p * epsilon_hat
    s_trial_norm = s_trial.norm()
    
    # Yield condition - using similar factor to 3D version (sqrt(2/3) ≈ 0.816)
    # For 2D, we use sqrt(2/3) to match the 3D implementation's scaling
    y = s_trial_norm - ti.sqrt(2.0 / 3.0) * yield_stress_p
    
    F_elastic_res = F_trial
    if y > 0:
        # Compute b_trial for mu_hat (2D adaptation)
        b_trial = sig0 * sig0 + sig1 * sig1
        mu_hat = mu_p * b_trial / 2.0
        
        # Rate-dependent plastic flow
        denom = 1.0 + plastic_viscosity_p / (2.0 * mu_hat * dt_val)
        s_new_norm = s_trial_norm - y / denom
        s_scale = s_new_norm / s_trial_norm if s_trial_norm > 1e-10 else 1.0
        s_new = s_scale * s_trial
        
        # Reconstruct strain
        epsilon_new = 1.0 / (2.0 * mu_p) * s_new + ti.Vector([trace_epsilon / 2.0, trace_epsilon / 2.0])
        
        # Reconstruct deformation gradient
        sig_elastic = ti.Matrix([[ti.exp(epsilon_new[0]), 0.0], [0.0, ti.exp(epsilon_new[1])]])
        F_elastic_res = U @ sig_elastic @ V.transpose()
    
    return F_elastic_res


@ti.func
def stvk_stress_2d(F_elastic, U, V, sig, mu_p, lam_p):
    """Saint Venant-Kirchhoff stress model (2D version)"""
    sig0 = ti.max(sig[0, 0], 0.01)
    sig1 = ti.max(sig[1, 1], 0.01)
    
    epsilon = ti.Vector([ti.log(sig0), ti.log(sig1)])
    log_sig_sum = ti.log(sig0) + ti.log(sig1)
    
    ONE = ti.Vector([1.0, 1.0])
    tau = 2.0 * mu_p * epsilon + lam_p * log_sig_sum * ONE
    
    tau_mat = ti.Matrix([[tau[0], 0.0], [0.0, tau[1]]])
    return U @ tau_mat @ V.transpose() @ F_elastic.transpose()


@ti.kernel
def p2g2p():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:
        if x[p][0] < 0.0 or x[p][1] < 0.0 or x[p][0] > 1.0 or x[p][1] > 1.0:
            continue
        
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        F_trial = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]
        
        F[p] = viscoplastic_return_mapping_stvk_2d(
            F_trial, mu_toothpaste, lam_toothpaste, yield_stress[p], plastic_viscosity[p], dt
        )
        
        U, sig, V = ti.svd(F[p])
        stress = stvk_stress_2d(F[p], U, V, sig, mu_toothpaste, lam_toothpaste)
        
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j]
            grid_v[i, j] += dt * gravity[None] * 30
            
            if i < 3 and grid_v[i, j][0] < 0:
                grid_v[i, j][0] = 0
            if i > n_grid - 3 and grid_v[i, j][0] > 0:
                grid_v[i, j][0] = 0
            
            if j < 3 and grid_v[i, j][1] < 0:
                v_y = grid_v[i, j][1]
                v_x = grid_v[i, j][0]
                normal_component = ti.abs(v_y)
                tangential_velocity = v_x
                
                if ti.abs(tangential_velocity) > 1e-20:
                    friction_reduction = ground_friction * normal_component
                    v_x_new = tangential_velocity
                    if v_x_new > 0:
                        v_x_new = ti.max(0.0, v_x_new - friction_reduction)
                    else:
                        v_x_new = ti.min(0.0, v_x_new + friction_reduction)
                    grid_v[i, j] = ti.Vector([v_x_new, 0.0])
                else:
                    grid_v[i, j] = ti.Vector([0.0, 0.0])
            
            if j > n_grid - 3 and grid_v[i, j][1] > 0:
                grid_v[i, j][1] = 0
    for p in x:
        if x[p][0] < 0.0 or x[p][1] < 0.0 or x[p][0] > 1.0 or x[p][1] > 1.0:
            continue
        
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]


@ti.kernel
def reset():
    bar_width = 0.015
    bar_height = 0.85
    bar_bottom_y = 0.1
    bar_spacing = 0.2
    bar1_center_x = 0.5 - bar_spacing / 2.0
    bar2_center_x = 0.5 + bar_spacing / 2.0
    
    particles_per_bar = n_particles // 2
    
    for i in range(n_particles):
        bar_center_x = 0.5
        bar_yield_stress = 1.2
        bar_plastic_viscosity = 0.7
        
        if i < particles_per_bar:
            bar_center_x = bar1_center_x
            bar_yield_stress = 1.2
            bar_plastic_viscosity = 0.7
        else:
            bar_center_x = bar2_center_x
            bar_yield_stress = 1.0
            bar_plastic_viscosity = 0.3
        
        x_pos = ti.random() * bar_width + (bar_center_x - bar_width / 2.0)
        y_pos = ti.random() * bar_height + bar_bottom_y
        
        x[i] = [x_pos, y_pos]
        
        stripe_width = bar_width / 3.0
        relative_x = x_pos - (bar_center_x - bar_width / 2.0)
        if relative_x < stripe_width:
            color_stripe[i] = 0
        elif relative_x < 2.0 * stripe_width:
            color_stripe[i] = 1
        else:
            color_stripe[i] = 2
        
        v[i] = [0, 0]
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        C[i] = ti.Matrix.zero(float, 2, 2)
        yield_stress[i] = bar_yield_stress
        plastic_viscosity[i] = bar_plastic_viscosity


gui = ti.GUI("Taichi MLS-MPM-128", res=1024, background_color=0xEEEEF0, show_gui=False)
reset()
gravity[None] = [0, -1]

save_duration = 10.0
output_fps = 30
frame_time = 3e-2
total_frames_to_run = int(save_duration / frame_time)
frames_per_save = int(1.0 / (output_fps * frame_time))
total_saves = int(save_duration * output_fps)
save_counter = 0

output_dir = os.path.join(os.path.dirname(__file__), "frames")
os.makedirs(output_dir, exist_ok=True)
video_output = os.path.join(os.path.dirname(__file__), "viscoplastic_animation.mp4")


for frame in range(total_frames_to_run):
    substeps_per_frame = int(2e-3 // dt)
    
    for s in range(substeps_per_frame):
        p2g2p()
    
    if frame % frames_per_save == 0 and save_counter < total_saves:
        gui.circles(
            x.to_numpy(),
            radius=1.5,
            palette=[0x3366FF, 0xFFFF00, 0xFF3333],
            palette_indices=color_stripe,
        )
        filename = os.path.join(output_dir, f'{save_counter:06d}.png')
        gui.show(filename)
        save_counter += 1
        if save_counter % 30 == 0:
            print(f"Saved {save_counter}/{total_saves} frames...")

print(f"Frame saving complete! Saved {save_counter} frames.")

ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path is not None and save_counter > 0:
    pattern = os.path.join(output_dir, "%06d.png")
    cmd = (
        f'"{ffmpeg_path}" -y -framerate {output_fps} -i "{pattern}" '
        f'-c:v libx264 -pix_fmt yuv420p -crf 18 "{video_output}"'
    )
    print("Running:", cmd)
    os.system(cmd)
    print(f"Video written to: {video_output}")
else:
    print("ffmpeg not found or no frames saved. To build video manually, run:")
    print(
        f"ffmpeg -y -framerate {output_fps} -i '{os.path.join(output_dir, '%06d.png')}' "
        f"-c:v libx264 -pix_fmt yuv420p -crf 18 '{video_output}'"
    )