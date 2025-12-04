import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import cg
import csv

# --- Simulation Parameters ---
NX = 50  # Number of grid cells in x
NY = 50  # Number of grid cells in y
X_MAX = 1.0
Y_MAX = 1.0
DX = X_MAX / NX
DY = Y_MAX / NY
DT = 0.001 # Time step
T_FINAL = 0.2 # Final time (adjust for shorter/longer simulation)
N_STEPS = int(T_FINAL / DT)
PLOT_EVERY = N_STEPS // 2 # Plot a few intermediate steps + initial/final

# Courant-Friedrichs-Lewy (CFL) condition parameter for advection
CFL_SAFETY_FACTOR = 0.5

# --- Domain Definition (Irregular Hexagon) ---
# Centered hexagon, slightly irregular by adjusting radius for some points
center_x, center_y = X_MAX / 2, Y_MAX / 2
base_radius = min(X_MAX, Y_MAX) * 0.4

hexagon_vertices = []
radii_factors = [1.0, 0.9, 1.1, 1.0, 0.95, 1.05] # Make it irregular
for i in range(6):
    angle = np.pi / 3 * i + np.pi/6 # Rotate slightly for better visual
    current_radius = base_radius * radii_factors[i]
    hexagon_vertices.append([center_x + current_radius * np.cos(angle),
                             center_y + current_radius * np.sin(angle)])
hexagon_path = mpltPath.Path(np.array(hexagon_vertices))

# --- Create Mesh ---
# Pressure points (cell centers)
xp = np.linspace(DX / 2, X_MAX - DX / 2, NX)
yp = np.linspace(DY / 2, Y_MAX - DY / 2, NY)
Xp, Yp = np.meshgrid(xp, yp)

# U-velocity points (centers of vertical cell faces)
xu = np.linspace(0, X_MAX, NX + 1) # NX+1 points for NX cells
yu = np.linspace(DY / 2, Y_MAX - DY / 2, NY)
Xu, Yu = np.meshgrid(xu, yu)

# V-velocity points (centers of horizontal cell faces)
xv = np.linspace(DX / 2, X_MAX - DX / 2, NX)
yv = np.linspace(0, Y_MAX, NY + 1) # NY+1 points for NY cells
Xv, Yv = np.meshgrid(xv, yv)

# --- Identify cells inside the polygon (Masks) ---
# True if point is inside polygon
pressure_points = np.vstack((Xp.ravel(), Yp.ravel())).T
p_mask = hexagon_path.contains_points(pressure_points).reshape(NY, NX)

u_points = np.vstack((Xu.ravel(), Yu.ravel())).T
u_mask = hexagon_path.contains_points(u_points).reshape(NY, NX + 1)

v_points = np.vstack((Xv.ravel(), Yv.ravel())).T
v_mask = hexagon_path.contains_points(v_points).reshape(NY + 1, NX)

# --- Initial Conditions: Multiple Vortices ---
u = np.zeros((NY, NX + 1)) # x-component of velocity
v = np.zeros((NY + 1, NX)) # y-component of velocity
p = np.zeros((NY, NX))     # pressure

def add_rankine_vortex(xc, yc, gamma, core_radius, u_arr, v_arr, Xu_grid, Yu_grid, Xv_grid, Yv_grid):
    # For u-velocity points (x-component)
    dx_u, dy_u = Xu_grid - xc, Yu_grid - yc
    r_u = np.sqrt(dx_u**2 + dy_u**2)
    r_u[r_u < 1e-9] = 1e-9 # Avoid division by zero
    vel_theta_u = np.where(r_u <= core_radius,
                           gamma * r_u / (2 * np.pi * core_radius**2),
                           gamma / (2 * np.pi * r_u))
    u_arr += -vel_theta_u * dy_u / r_u # u_x = -vel_theta * sin(theta)

    # For v-velocity points (y-component)
    dx_v, dy_v = Xv_grid - xc, Yv_grid - yc
    r_v = np.sqrt(dx_v**2 + dy_v**2)
    r_v[r_v < 1e-9] = 1e-9
    vel_theta_v = np.where(r_v <= core_radius,
                           gamma * r_v / (2 * np.pi * core_radius**2),
                           gamma / (2 * np.pi * r_v))
    v_arr += vel_theta_v * dx_v / r_v # u_y = vel_theta * cos(theta)

# Vortex parameters: (center_x, center_y, strength Gamma, core_radius R)
vortices_params = [
    (center_x - base_radius*0.45, center_y + base_radius*0.45, 1.5, 0.06),
    (0.3,0.3, 2.0, 0.06),
    (center_x + base_radius*0.45, center_y - base_radius*0.25, -1.5, 0.06),
    (center_x + base_radius*0.0, center_y + base_radius*0.5, -1.0, 0.05),
    (center_x + base_radius*0.1, center_y + base_radius*0.1, 1.0, 0.05),
]
for xc, yc, gamma, core_r in vortices_params:
    add_rankine_vortex(xc, yc, gamma, core_r, u, v, Xu, Yu, Xv, Yv)

u *= u_mask # Ensure initial velocities are zero outside polygon
v *= v_mask

u_initial = u.copy()
v_initial = v.copy()

# --- Helper functions ---
def calculate_vorticity_at_centers(u_arr, v_arr, dx, dy, p_mask_local):
    """
    Calculates vorticity omega_z = dv/dx - du/dy at pressure cell centers.
    This version uses centered differences of face-averaged velocities.
    A simpler, but common, definition for staggered grids is also possible (see notes).
    The formula used here: omega[j,i] = (v[j+1,i] - v[j,i])/DX - (u[j,i+1] - u[j,i])/DY
    This is actually (dv_y/dy - du_x/dx) if (j,i) is bottom-left.
    Let's use the circulation-based definition for P_j,i (prog indices):
    omega_z = (u_left - u_right)/DY + (v_top - v_bottom)/DX
    u_left = u_arr[j,i], u_right = u_arr[j,i+1]
    v_bottom = v_arr[j,i], v_top = v_arr[j+1,i]
    """
    omega = np.zeros((NY, NX))
    for j in range(NY):
        for i in range(NX):
            if p_mask_local[j,i]:
                # omega[j,i] = (v[j+1,i] - v[j,i])/DY - (u[j,i+1] - u[j,i])/DX # This is dv/dy - du/dx
                # For omega_z = dv/dx - du/dy.
                # dv/dx at P(j,i): use v at interfaces to the right and left of P(j,i)'s center.
                #   v_interface_R = (v[j,i] + v[j+1,i])/2 (v at x=xp[i], avg over y)
                #   v_interface_L = (v[j,i-1] + v[j+1,i-1])/2 (v at x=xp[i-1], avg over y) (for i>0)
                # du/dy at P(j,i): use u at interfaces above and below P(j,i)'s center.
                #   u_interface_T = (u[j,i] + u[j,i+1])/2 (u at y=yp[j], avg over x)
                #   u_interface_B = (u[j-1,i] + u[j-1,i+1])/2 (u at y=yp[j-1], avg over x) (for j>0)

                # Using simpler definition often cited for staggered grids (may be off by sign or convention):
                # (u_left - u_right)/DY + (v_top - v_bottom)/DX
                term_u = (u_arr[j,i] - u_arr[j,i+1]) / dy # (u_L - u_R)/dy
                term_v = (v_arr[j+1,i] - v_arr[j,i]) / dx # (v_T - v_B)/dx
                omega[j,i] = term_v + term_u
    return omega * p_mask_local
def calculate_divergence_at_centers(u_arr, v_arr, dx, dy, p_mask_local):
    """
    Calculates divergence at pressure cell centers using finite differences.
    divergence[j,i] = (u_arr[j, i+1] - u_arr[j, i]) / dx + (v_arr[j+1, i] - v_arr[j, i]) / dy
    """
    div = np.zeros((NY, NX))
    for j in range(NY):
        for i in range(NX):
            if p_mask_local[j, i]:
                term_u = (u_arr[j, i+1] - u_arr[j, i]) / dx
                term_v = (v_arr[j+1, i] - v_arr[j, i]) / dy
                div[j, i] = term_u + term_v
    return div * p_mask_local

def plot_state(fig_num, time, u_vel, v_vel, p_scal, title_prefix, p_mask_plot, Xp_grid, Yp_grid):
    plt.figure(fig_num, figsize=(14, 6))
    plt.clf() # Clear figure
    axs = [plt.subplot(121), plt.subplot(122)]

    u_c = np.zeros((NY, NX)) # u at cell centers
    v_c = np.zeros((NY, NX)) # v at cell centers
    for j in range(NY):
        for i in range(NX):
            if p_mask_plot[j,i]:
                u_c[j,i] = 0.5 * (u_vel[j,i] + u_vel[j,i+1])
                v_c[j,i] = 0.5 * (v_vel[j,i] + v_vel[j+1,i])

    u_c[~p_mask_plot] = np.nan # Mask for plotting
    v_c[~p_mask_plot] = np.nan

    skip = 2 # Plot every 'skip' arrow
    axs[0].quiver(Xp_grid[::skip, ::skip], Yp_grid[::skip, ::skip],
                  u_c[::skip, ::skip], v_c[::skip, ::skip],
                  color='blue', scale=20, width=0.003, angles='xy', scale_units='xy')
    axs[0].plot(np.array(hexagon_vertices)[:,0], np.array(hexagon_vertices)[:,1], 'k-', lw=1.5)
    axs[0].plot(np.append(hexagon_vertices, [hexagon_vertices[0]], axis=0)[:,0],
                np.append(hexagon_vertices, [hexagon_vertices[0]], axis=0)[:,1], 'k-', lw=1.5) # Close polygon

    axs[0].set_title(f'{title_prefix} Velocity (t={time:.3f}s)')
    axs[0].set_xlabel('x'); axs[0].set_ylabel('y')
    axs[0].set_xlim([0, X_MAX]); axs[0].set_ylim([0, Y_MAX])
    axs[0].set_aspect('equal', adjustable='box')

    vort = calculate_vorticity_at_centers(u_vel, v_vel, DX, DY, p_mask_plot)
    vort[~p_mask_plot] = np.nan # Mask for plotting

    vort_abs_max = np.nanmax(np.abs(vort)) if np.any(~np.isnan(vort)) else 1.0
    if vort_abs_max < 1e-6: vort_abs_max = 1.0

    im = axs[1].contourf(Xp_grid, Yp_grid, vort, levels=np.linspace(-vort_abs_max, vort_abs_max, 31),
                         cmap='coolwarm', extend='both')
    axs[1].plot(np.append(hexagon_vertices, [hexagon_vertices[0]], axis=0)[:,0],
                np.append(hexagon_vertices, [hexagon_vertices[0]], axis=0)[:,1], 'k-', lw=1.5)
    axs[1].set_title(f'{title_prefix} Vorticity (t={time:.3f}s)')
    axs[1].set_xlabel('x'); axs[1].set_ylabel('y')
    axs[1].set_aspect('equal', adjustable='box')
    plt.colorbar(im, ax=axs[1], orientation='vertical', label='Vorticity')

    plt.tight_layout()
    plt.pause(0.01) # Pause to update plot

# --- Boundary Condition Application & Masking ---
def apply_boundary_conditions_and_masks(u_arr, v_arr, u_mask_domain, v_mask_domain, p_mask_domain):
    # Strictly enforce zero velocity outside the polygon for all components
    u_arr[~u_mask_domain] = 0.0
    v_arr[~v_mask_domain] = 0.0

    # Box boundaries (if polygon doesn't touch them, these act as free-slip/no-pen depending on setup)
    # For simplicity, let's enforce no-penetration on the overall simulation box edges
    # These might be overridden by the polygon mask if the polygon edge is there.
    u_arr[:, 0] = 0  # Left wall
    u_arr[:, NX] = 0 # Right wall
    v_arr[0, :] = 0  # Bottom wall
    v_arr[NY, :] = 0 # Top wall

    # Re-apply polygon mask after box BCs, as box BCs might affect points that should be masked.
    u_arr[~u_mask_domain] = 0.0
    v_arr[~v_mask_domain] = 0.0

    # Approximate No-Penetration for polygon:
    # If a u-face is on the boundary (one p-cell in, one out), set u=0.
    # If a v-face is on the boundary, set v=0.
    # This is a simplification. True no-penetration requires projecting velocity normal to the actual boundary segment.
    for j in range(NY):
        for i in range(1, NX): # Internal u-faces
            if u_mask_domain[j,i]: # If u-node is in fluid domain
                is_left_p_fluid = p_mask_domain[j,i-1]
                is_right_p_fluid = p_mask_domain[j,i]
                if (is_left_p_fluid and not is_right_p_fluid) or \
                   (not is_left_p_fluid and is_right_p_fluid):
                    u_arr[j,i] = 0.0 # u is normal to this effective boundary face

    for i in range(NX):
        for j in range(1, NY): # Internal v-faces
            if v_mask_domain[j,i]: # If v-node is in fluid domain
                is_bottom_p_fluid = p_mask_domain[j-1,i]
                is_top_p_fluid = p_mask_domain[j,i]
                if (is_bottom_p_fluid and not is_top_p_fluid) or \
                   (not is_bottom_p_fluid and is_top_p_fluid):
                    v_arr[j,i] = 0.0 # v is normal to this effective boundary face
    return u_arr, v_arr

u, v = apply_boundary_conditions_and_masks(u, v, u_mask, v_mask, p_mask)
u_initial, v_initial = u.copy(), v.copy()

# Plot initial state
print("Plotting initial state...")
plot_state(1, 0.0, u_initial, v_initial, p, "Initial", p_mask, Xp, Yp)
plt.savefig("fluid_sim_initial_state.png")


# --- Time Stepping Loop (Projection Method) ---
print(f"Starting simulation for {N_STEPS} steps, DT={DT:.2e}...")
for n_step in range(N_STEPS):
    u_old = u.copy()
    v_old = v.copy()

    # --- CFL Check (Informational) ---
    max_abs_u = np.max(np.abs(u_old[u_mask])) if np.any(u_old[u_mask]) else 1e-9
    max_abs_v = np.max(np.abs(v_old[v_mask])) if np.any(v_old[v_mask]) else 1e-9
    dt_cfl_x = DX / max_abs_u * CFL_SAFETY_FACTOR
    dt_cfl_y = DY / max_abs_v * CFL_SAFETY_FACTOR

    if DT > min(dt_cfl_x, dt_cfl_y):
        print(f"Warning: Step {n_step}, DT={DT:.2e} may be too large for CFL. Max U={max_abs_u:.2e}, Max V={max_abs_v:.2e}")
        print(f"Recommended DT_x={dt_cfl_x:.2e}, DT_y={dt_cfl_y:.2e}")
        # In a real sim, you might reduce DT or stop. For this example, we continue.

    # 1. Advection Term (Intermediate velocity u_star, v_star)
    # Using first-order upwind (simple, but diffusive). Python loops are slow.
    u_star = u_old.copy()
    v_star = v_old.copy()

    # Advect u (x-velocity)
    for j in range(NY):      # Iterates over u's y-indices (yu)
        for i in range(1, NX): # Iterates over u's x-indices (xu), internal faces
            if u_mask[j,i]: # Only compute for u-points within the fluid domain
                # Interpolate v to u[j,i]'s location: (i*DX, (j+0.5)*DY)
                # v_adj are v[j, i-1], v[j+1, i-1], v[j, i], v[j+1, i] (corners around u)
                v_interp = 0.25 * (v_old[j, i-1] + v_old[j+1, i-1] + v_old[j, i] + v_old[j+1, i])

                # u * du/dx (upwind)
                u_curr = u_old[j,i]
                if u_curr >= 0:
                    grad_u_x = u_curr * (u_old[j,i] - u_old[j,i-1]) / DX
                else:
                    grad_u_x = u_curr * (u_old[j,i+1] - u_old[j,i]) / DX

                # v * du/dy (upwind)
                # u_up = u_old[j+1,i], u_down = u_old[j-1,i]
                grad_u_y = 0
                if v_interp >= 0:
                    if j > 0: grad_u_y = v_interp * (u_old[j,i] - u_old[j-1,i]) / DY
                    else: grad_u_y = v_interp * (u_old[j,i] - 0) / (DY/2) # Approx at boundary
                else:
                    if j < NY-1: grad_u_y = v_interp * (u_old[j+1,i] - u_old[j,i]) / DY
                    else: grad_u_y = v_interp * (0 - u_old[j,i]) / (DY/2) # Approx at boundary

                u_star[j,i] -= DT * (grad_u_x + grad_u_y)

    # Advect v (y-velocity)
    for j in range(1, NY): # Iterates over v's y-indices (yv), internal faces
        for i in range(NX):  # Iterates over v's x-indices (xv)
            if v_mask[j,i]: # Only compute for v-points within the fluid domain
                # Interpolate u to v[j,i]'s location: ((i+0.5)*DX, j*DY)
                # u_adj are u[j-1,i], u[j-1,i+1], u[j,i], u[j,i+1]
                u_interp = 0.25 * (u_old[j-1, i] + u_old[j-1, i+1] + u_old[j, i] + u_old[j, i+1])

                # u * dv/dx (upwind)
                grad_v_x = 0
                if u_interp >= 0:
                    if i > 0: grad_v_x = u_interp * (v_old[j,i] - v_old[j,i-1]) / DX
                    else: grad_v_x = u_interp * (v_old[j,i] - 0) / (DX/2)
                else:
                    if i < NX-1: grad_v_x = u_interp * (v_old[j,i+1] - v_old[j,i]) / DX
                    else: grad_v_x = u_interp * (0-v_old[j,i]) / (DX/2)

                # v * dv/dy (upwind)
                v_curr = v_old[j,i]
                if v_curr >= 0:
                    grad_v_y = v_curr * (v_old[j,i] - v_old[j-1,i]) / DY
                else:
                    grad_v_y = v_curr * (v_old[j+1,i] - v_old[j,i]) / DY

                v_star[j,i] -= DT * (grad_v_x + grad_v_y)

    u_star, v_star = apply_boundary_conditions_and_masks(u_star, v_star, u_mask, v_mask, p_mask)

    # 2. Pressure Correction Step
    # Solve Poisson eq: Del^2 P = (1/DT) * Div(u_star)
    rhs_p = np.zeros((NY, NX))
    for j in range(NY):
        for i in range(NX):
            if p_mask[j,i]: # Only for fluid pressure cells
                div_u_star = (u_star[j,i+1] - u_star[j,i])/DX + \
                             (v_star[j+1,i] - v_star[j,i])/DY
                rhs_p[j,i] = div_u_star / DT

    # Assemble sparse matrix A for Lap(P) = rhs_p for fluid cells
    # Map (j,i) of fluid p_cells to a linear index k
    k_map = -np.ones_like(p_mask, dtype=int)
    fluid_cell_count = 0
    for j_idx in range(NY):
        for i_idx in range(NX):
            if p_mask[j_idx, i_idx]:
                k_map[j_idx, i_idx] = fluid_cell_count
                fluid_cell_count += 1

    if fluid_cell_count == 0:
        p.fill(0) # No fluid cells
    else:
        A_p = lil_matrix((fluid_cell_count, fluid_cell_count))
        b_p_flat = np.zeros(fluid_cell_count)

        for j in range(NY):
            for i in range(NX):
                if p_mask[j,i]:
                    k = k_map[j,i]
                    b_p_flat[k] = rhs_p[j,i]

                    # Center coefficient for P[j,i]
                    A_p[k,k] = -2.0/(DX**2) - 2.0/(DY**2)

                    # Neighbors (Neumann BC: dP/dn=0 at polygon boundary)
                    # If neighbor is solid, it modifies the diagonal (P_ghost = P_center)
                    # X-direction
                    if i+1 < NX and p_mask[j,i+1]: A_p[k, k_map[j,i+1]] = 1.0/(DX**2) # East fluid
                    else: A_p[k,k] += 1.0/(DX**2) # East solid/boundary

                    if i-1 >= 0 and p_mask[j,i-1]: A_p[k, k_map[j,i-1]] = 1.0/(DX**2) # West fluid
                    else: A_p[k,k] += 1.0/(DX**2) # West solid/boundary

                    # Y-direction
                    if j+1 < NY and p_mask[j+1,i]: A_p[k, k_map[j+1,i]] = 1.0/(DY**2) # North fluid
                    else: A_p[k,k] += 1.0/(DY**2) # North solid/boundary

                    if j-1 >= 0 and p_mask[j-1,i]: A_p[k, k_map[j-1,i]] = 1.0/(DY**2) # South fluid
                    else: A_p[k,k] += 1.0/(DY**2) # South solid/boundary

        A_p_csr = A_p.tocsr()
        p_prev_flat = p[p_mask].ravel() # Use previous pressure as initial guess

        # Check if p_prev_flat has correct size (can happen if fluid_cell_count changes, though unlikely here)
        if len(p_prev_flat) != fluid_cell_count:
            p_prev_flat = np.zeros(fluid_cell_count)

        p_flat_sol, info = cg(A_p_csr, b_p_flat, x0=p_prev_flat, rtol=1e-6, maxiter=1000)
        if info != 0: print(f"Pressure solver convergence issue (info={info}) at step {n_step}.")

        # Scatter solution back to 2D pressure array
        p_new = np.zeros_like(p)
        p_new[p_mask] = p_flat_sol
        p = p_new

    # 3. Velocity Update
    u_next = u_star.copy()
    v_next = v_star.copy()

    for j in range(NY):
        for i in range(1, NX): # u-velocities are at (j, i) where i is face index
            if u_mask[j,i]: # Update only fluid u-velocities
                # Pressure gradient dP/dx at u[j,i] uses P[j,i] (right) and P[j,i-1] (left)
                if p_mask[j,i] and p_mask[j,i-1]: # Both adjacent p-cells are fluid
                    grad_p_x = (p[j,i] - p[j,i-1]) / DX
                elif p_mask[j,i]: # Fluid right, solid left (approx. dP/dx = 0 into solid)
                    grad_p_x = (p[j,i] - p[j,i]) / DX # effectively P_left_ghost = P_right
                elif p_mask[j,i-1]: # Fluid left, solid right
                    grad_p_x = (p[j,i-1] - p[j,i-1]) / DX # effectively P_right_ghost = P_left
                else: # Both solid (u should be 0 by mask anyway)
                    grad_p_x = 0
                u_next[j,i] -= DT * grad_p_x

    for j in range(1, NY): # v-velocities are at (j, i) where j is face index
        for i in range(NX):
            if v_mask[j,i]: # Update only fluid v-velocities
                # Pressure gradient dP/dy at v[j,i] uses P[j,i] (top) and P[j-1,i] (bottom)
                if p_mask[j,i] and p_mask[j-1,i]: # Both adjacent p-cells are fluid
                    grad_p_y = (p[j,i] - p[j-1,i]) / DY
                elif p_mask[j,i]: # Fluid top, solid bottom
                    grad_p_y = (p[j,i] - p[j,i]) / DY
                elif p_mask[j-1,i]: # Fluid bottom, solid top
                    grad_p_y = (p[j-1,i] - p[j-1,i]) / DY
                else: # Both solid
                    grad_p_y = 0
                v_next[j,i] -= DT * grad_p_y

    u = u_next
    v = v_next
    u, v = apply_boundary_conditions_and_masks(u, v, u_mask, v_mask, p_mask)

    if (n_step + 1) % PLOT_EVERY == 0 or n_step == N_STEPS -1 :
        print(f"Step {n_step+1}/{N_STEPS}, Time: {(n_step+1)*DT:.3f} s. Plotting...")
        plot_state(1, (n_step+1)*DT, u, v, p, f"State at t={(n_step+1)*DT:.2f}", p_mask, Xp, Yp)
        if n_step == N_STEPS -1:
             plt.savefig(f"fluid_sim_final_state_t{T_FINAL:.2f}.png")
# --- Plot divergence of final state ---
divergence = calculate_divergence_at_centers(u, v, DX, DY, p_mask)
plt.figure(2, figsize=(6,5))
contf = plt.contourf(Xp, Yp, divergence, levels=31, cmap='viridis')
plt.colorbar(contf, label='Divergence')
plt.title(f'Final Divergence (t={T_FINAL:.3f}s)')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(f"fluid_sim_final_divergence_t{T_FINAL:.2f}.png")

# --- Output final state to CSV ---
with open("fluid_sim_final_state.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["x", "y", "vx", "vy", "vorticity", "divergence"])
    # Compute vorticity and divergence for final state
    vort = calculate_vorticity_at_centers(u, v, DX, DY, p_mask)
    div_val = calculate_divergence_at_centers(u, v, DX, DY, p_mask)
    for j in range(NY):
        for i in range(NX):
            if p_mask[j, i]:
                x_val = Xp[j, i]
                y_val = Yp[j, i]
                vx_val = 0.5 * (u[j, i] + u[j, i+1])
                vy_val = 0.5 * (v[j, i] + v[j+1, i])
                writer.writerow([x_val, y_val, vx_val, vy_val, vort[j, i], div_val[j, i]])


# Plot final state (if not plotted by loop condition)
# if N_STEPS % PLOT_EVERY != 0 :
#    plot_state(1, T_FINAL, u, v, p, "Final", p_mask, Xp, Yp)
#    plt.savefig("fluid_sim_final_state.png")
