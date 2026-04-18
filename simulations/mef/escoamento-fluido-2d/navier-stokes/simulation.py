import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from pyscript import when, document
import asyncio
from common import generate_mesh_generic, assemble_matrices, plot_to_base64
import geometry

# Global control flag
STOP_SIMULATION = False

@when("click", "#stop-btn")
def stop_simulation_handler(event=None):
    global STOP_SIMULATION
    STOP_SIMULATION = True
    document.getElementById("stop-btn").classList.add("hidden")
    document.getElementById("stop-btn").classList.remove("flex")
    document.getElementById("run-btn").classList.remove("hidden")
    document.getElementById("run-btn").classList.add("flex")

# ==============================
# NS Helpers
# ==============================
def assemble_convection(X, Y, triang, u_el, v_el):
    n_nodes = len(X)
    C = lil_matrix((n_nodes, n_nodes))
    elements = triang.triangles
    mask = triang.mask if triang.mask is not None else np.zeros(len(elements), dtype=bool)
    
    for i, el in enumerate(elements):
        if mask[i]: continue
        x = X[el]
        y = Y[el]
        area = 0.5 * np.abs(x[0]*(y[1]-y[2]) + x[1]*(y[2]-y[0]) + x[2]*(y[0]-y[1]))
        b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
        c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
        ue = u_el[i]
        ve = v_el[i]
        K_dot_grad = (ue * b + ve * c) / (2 * area)
        ce = np.outer(np.ones(3) * (area / 3.0), K_dot_grad)
        for row in range(3):
            for col in range(3):
                C[el[row], el[col]] += ce[row, col]
    return C.tocsr()

def get_boundary_normals(X, Y, boundary_nodes, triang):
    adj = [set() for _ in range(len(X))]
    mask = triang.mask if triang.mask is not None else np.zeros(len(triang.triangles), dtype=bool)
    for i, el in enumerate(triang.triangles):
        if mask[i]: continue
        for j in range(3):
            n1, n2 = el[j], el[(j+1)%3]
            adj[n1].add(n2)
            adj[n2].add(n1)
            
    boundary_set = set(boundary_nodes)
    valid_neighbors = np.zeros(len(boundary_nodes), dtype=int)
    h_sq = np.zeros(len(boundary_nodes))
    
    for i, node in enumerate(boundary_nodes):
        node_neighbors = list(adj[node])
        best_nb = -1
        min_dist = 1e9
        for nb in node_neighbors:
            if nb not in boundary_set:
                d = (X[node]-X[nb])**2 + (Y[node]-Y[nb])**2
                if d < min_dist:
                    min_dist = d
                    best_nb = nb
        if best_nb == -1:
            for nb in node_neighbors:
                d = (X[node]-X[nb])**2 + (Y[node]-Y[nb])**2
                if d < min_dist:
                    min_dist = d
                    best_nb = nb
        valid_neighbors[i] = best_nb
        h_sq[i] = min_dist
    return valid_neighbors, h_sq

# ==============================
# Solver NS
# ==============================
async def solve_navier_stokes(params, plot_div):
    # 1. Malha
    cx, cy = params["cx"], params["cy"]
    
    # Helper for inlet profile
    def get_psi_inlet(y_val):
        # Default: u=1, psi=y
        return y_val

    scenario = params.get("scenario", "channel")
    
    if scenario == "cavity":
        # Force no geometry/obstruction for cavity
        boundary_pts = np.empty((0, 2))
        def mask_func(x, y): return np.ones_like(x, dtype=bool)
    else:
        boundary_pts, mask_func = geometry.get_geometry(params["geo_type"], params)

    X, Y, triang = generate_mesh_generic(
        params["L"], params["H"], boundary_pts, mask_func, 
        cx, cy, params["nx"], params["ny"]
    )
    n_nodes = len(X)
    
    # 2. Matrizes
    K, M = assemble_matrices(X, Y, triang)
    
    # 3. Fronteiras
    eps = 1e-5
    inlet_nodes = np.where(X < eps)[0]
    wall_bottom = np.where(Y < eps)[0]
    wall_top = np.where(Y > params["H"] - eps)[0]
    
    # Obstacle nodes (last N points)
    n_bound = len(boundary_pts)
    obstacle_nodes = np.arange(n_nodes - n_bound, n_nodes)
    
    solid_walls = np.concatenate([wall_bottom, wall_top, obstacle_nodes])
    solid_walls = np.unique(solid_walls)
    
    wall_neighbors, wall_h_sq = get_boundary_normals(X, Y, solid_walls, triang)
    
    # 4. Inicialização
    # 4. Inicialização e Condições de Contorno
    psi = np.zeros(n_nodes)
    omega = np.zeros(n_nodes)
    
    scenario = params.get("scenario", "channel")
    
    if scenario == "cavity":
        # Lid-Driven Cavity
        # Walls: Bottom, Left, Right, Top
        # Psi = 0 on all walls (closed box)
        # Omega BC drives the flow.
        
        # Identify walls
        eps = 1e-5
        wall_bottom = np.where(Y < eps)[0]
        wall_top = np.where(Y > params["H"] - eps)[0]
        wall_left = np.where(X < eps)[0]
        wall_right = np.where(X > params["L"] - eps)[0]
        
        # All boundary nodes
        all_walls = np.concatenate([wall_bottom, wall_top, wall_left, wall_right])
        all_walls = np.unique(all_walls)
        
        # Psi BC: 0 everywhere on boundary
        psi[all_walls] = 0.0
        
        dirichlet_psi = all_walls
        
        # Omega BC nodes (same as Psi for now, updated in loop)
        dirichlet_omega = all_walls
        
        # For Thom's formula, we need neighbors for ALL walls
        wall_neighbors, wall_h_sq = get_boundary_normals(X, Y, all_walls, triang)
        
        # Store wall types for specific BC application
        # Top wall moves: u = 1 -> dPsi/dy = 1
        # Thom's formula with moving wall:
        # omega_w = -2(psi_nb - psi_w - h*u_wall) / h^2
        # Here u_wall = 1 for top, 0 for others.
        
        # We need to map each wall node to its velocity (u_tan)
        wall_u_tan = np.zeros(len(all_walls))
        
        # Find indices of top wall nodes within all_walls
        # This is O(N^2) worst case but N_bound is small.
        # Better: Create a map.
        
        # Let's just iterate
        for i, node_idx in enumerate(all_walls):
            if Y[node_idx] > params["H"] - eps:
                wall_u_tan[i] = 1.0 # Moving lid
            else:
                wall_u_tan[i] = 0.0
                
        solid_walls = all_walls # For consistency with loop
        
    else:
        # Channel Flow (Original Logic)
        psi[inlet_nodes] = get_psi_inlet(Y[inlet_nodes])
        psi[wall_bottom] = 0.0
        
        # Top Wall BC
        psi[wall_top] = params["H"]
        
        # Obstacle BC
        if params.get("geo_type") == "step":
             psi[obstacle_nodes] = 0.0
        else:
            # Cylinder/Rectangle/etc
            psi[obstacle_nodes] = get_psi_inlet(cy)

        dirichlet_psi = np.concatenate([inlet_nodes, wall_bottom, wall_top, obstacle_nodes])
        dirichlet_psi = np.unique(dirichlet_psi)
        
        solid_walls = np.concatenate([wall_bottom, wall_top, obstacle_nodes])
        solid_walls = np.unique(solid_walls)
        
        dirichlet_omega = np.concatenate([inlet_nodes, solid_walls])
        dirichlet_omega = np.unique(dirichlet_omega)
        
        wall_neighbors, wall_h_sq = get_boundary_normals(X, Y, solid_walls, triang)
        
        # Stationary walls
        wall_u_tan = np.zeros(len(solid_walls))

    free_psi = np.setdiff1d(np.arange(n_nodes), dirichlet_psi)
    free_omega = np.setdiff1d(np.arange(n_nodes), dirichlet_omega)
    
    K_free_psi = K[free_psi, :][:, free_psi]
    
    dt = params["dt"]
    Re = params["Re"]
    steps_per_frame = params["steps_per_frame"]
    total_steps = params["total_steps"]
    
    current_step = 0
    
    while current_step < total_steps and not STOP_SIMULATION:
        for _ in range(steps_per_frame):
            # A. Poisson Psi
            rhs_psi = M @ omega
            b_psi = rhs_psi[free_psi] - K[free_psi, :][:, dirichlet_psi] @ psi[dirichlet_psi]
            psi[free_psi] = spsolve(K_free_psi, b_psi)
            
            # B. Thom's Formula (Generalized for moving walls)
            # Canonical form: omega_w = -2 (psi_nb - psi_w)/h^2 - 2 u_tan/h
            # Derivation (top lid, n inward = -y): Taylor at the wall with step -h gives
            #   psi_nb = psi_w - h (dpsi/dy)_w + (h^2/2) (d2psi/dy2)_w
            # Since u = dpsi/dy, (dpsi/dy)_w = u_tan. With psi constant along the wall,
            # d2psi/dx2 = 0 there, so d2psi/dy2 = -omega_w. Solving:
            #   omega_w = -2 (psi_nb - psi_w + h u_tan) / h^2
            # Validated against Ghia et al. (1982) at Re=100 (RMS 0.065 vs. 0.93 with the
            # wrong sign). See .agents/review/thom_validation/ for the reproducible check.
            psi_nb = psi[wall_neighbors]
            psi_w = psi[solid_walls]
            h_vals = np.sqrt(wall_h_sq)
            omega[solid_walls] = -2.0 * (psi_nb - psi_w + h_vals * wall_u_tan) / wall_h_sq
            
            # C. Velocities
            elements = triang.triangles
            mask = triang.mask if triang.mask is not None else np.zeros(len(elements), dtype=bool)
            el_nodes = elements[~mask]
            x_el = X[el_nodes]
            y_el = Y[el_nodes]
            p_el = psi[el_nodes]
            
            b_coeff = np.column_stack([y_el[:,1]-y_el[:,2], y_el[:,2]-y_el[:,0], y_el[:,0]-y_el[:,1]])
            c_coeff = np.column_stack([x_el[:,2]-x_el[:,1], x_el[:,0]-x_el[:,2], x_el[:,1]-x_el[:,0]])
            area_el = 0.5 * np.abs(x_el[:,0]*(y_el[:,1]-y_el[:,2]) + x_el[:,1]*(y_el[:,2]-y_el[:,0]) + x_el[:,2]*(y_el[:,0]-y_el[:,1]))
            
            u_vals = np.sum(p_el * c_coeff, axis=1) / (2 * area_el)
            v_vals = -np.sum(p_el * b_coeff, axis=1) / (2 * area_el)
            
            u_el_full = np.zeros(len(elements))
            v_el_full = np.zeros(len(elements))
            u_el_full[~mask] = u_vals
            v_el_full[~mask] = v_vals
            
            # D. Convection
            C = assemble_convection(X, Y, triang, u_el_full, v_el_full)
            
            # E. Vorticity Transport
            A = M + (dt / Re) * K
            rhs = M @ omega - dt * (C @ omega)
            
            A_free = A[free_omega, :][:, free_omega]
            b_omega = rhs[free_omega] - A[free_omega, :][:, dirichlet_omega] @ omega[dirichlet_omega]
            omega[free_omega] = spsolve(A_free, b_omega)
            
            current_step += 1
            
            # F. Calculate Forces (Pressure + Viscous)
            # Removed asymmetry/lift proxy calculation as requested.
            
            current_step += 1
            
        # Plot
        tci = tri.LinearTriInterpolator(triang, psi)
        (dpsi_dx, dpsi_dy) = tci.gradient(X, Y)
        u = dpsi_dy
        v = -dpsi_dx
        
        fig = plt.figure(figsize=(10, 12))
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1])
        
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        
        # Smooth gradient for Psi using tripcolor with Gouraud shading
        # User requested "Blue to Red" (Jet)
        contour_psi = ax1.tripcolor(triang, psi, shading='gouraud', cmap='jet')
        # Removed discrete contour lines for cleaner look
        ax1.triplot(triang, color='k', alpha=0.2, linewidth=0.5) # Visible mesh
        ax1.set_title(f"Navier-Stokes (Step {current_step}) - $\\psi$")
        ax1.set_aspect('equal')
        ax1.set_xlabel("x [m]")
        ax1.set_ylabel("y [m]")
        fig.colorbar(contour_psi, ax=ax1)
        
        # Smooth gradient for Omega
        contour_omega = ax2.tripcolor(triang, omega, shading='gouraud', cmap='jet')
        ax2.triplot(triang, color='k', alpha=0.2, linewidth=0.5)
        ax2.set_title(f"Navier-Stokes (Step {current_step}) - $\\omega$")
        ax2.set_aspect('equal')
        ax2.set_xlabel("x [m]")
        ax2.set_ylabel("y [m]")
        fig.colorbar(contour_omega, ax=ax2)
        
        xi = np.linspace(0, params["L"], 100)
        yi = np.linspace(0, params["H"], 100)
        Xi, Yi = np.meshgrid(xi, yi)
        (dpsi_dx_grid, dpsi_dy_grid) = tci.gradient(Xi, Yi)
        u_grid = dpsi_dy_grid
        v_grid = -dpsi_dx_grid
        vel_mag_grid = np.sqrt(u_grid**2 + v_grid**2)
        
        # Streamlines with 'jet'
        strm = ax3.streamplot(Xi, Yi, u_grid, v_grid, color=vel_mag_grid, cmap='jet', density=1.5, linewidth=1, arrowsize=1.5)
        ax3.set_title("Linhas de Corrente")
        ax3.set_aspect('equal')
        ax3.set_xlim(0, params["L"])
        ax3.set_ylim(0, params["H"])
        ax3.set_xlabel("x [m]")
        ax3.set_ylabel("y [m]")
        fig.colorbar(strm.lines, ax=ax3)
        

        
        plt.tight_layout()
        
        img_base64 = plot_to_base64(fig)
        plot_div.innerHTML = f'<img src="data:image/png;base64,{img_base64}" class="max-w-full h-auto rounded shadow-lg" />'
        await asyncio.sleep(0.01)
        
    if not STOP_SIMULATION:
        plot_div.innerHTML += '<p class="text-center text-green-600 font-bold mt-2">Simulação Concluída!</p>'
    else:
        plot_div.innerHTML += '<p class="text-center text-red-600 font-bold mt-2">Simulação Parada pelo Usuário.</p>'

@when("click", "#run-btn")
async def run_handler(event=None):
    global STOP_SIMULATION
    STOP_SIMULATION = False
    
    plot_div = document.getElementById("plot-output")
    plot_div.innerHTML = "Iniciando..."
    await asyncio.sleep(0.1)
    
    # Show stop button
    document.getElementById("run-btn").classList.add("hidden")
    document.getElementById("run-btn").classList.remove("flex")
    document.getElementById("stop-btn").classList.remove("hidden")
    document.getElementById("stop-btn").classList.add("flex")
    
    try:
        params = {
            "L": float(document.getElementById("L").value),
            "H": float(document.getElementById("H").value),
            "cx": float(document.getElementById("cx").value),
            "cy": float(document.getElementById("cy").value),
            "scenario": document.getElementById("scenario").value,
            "geo_type": document.getElementById("geo_type").value,
            "r": float(document.getElementById("r").value),
            "w": float(document.getElementById("w").value),
            "h": float(document.getElementById("h").value),
            "step_h": float(document.getElementById("step_h").value),
            "step_l": float(document.getElementById("step_l").value),
            "nx": int(document.getElementById("nx").value),
            "ny": int(document.getElementById("ny").value),
            "Re": float(document.getElementById("Re").value),
            "dt": float(document.getElementById("dt").value),
            "steps_per_frame": int(document.getElementById("steps_per_frame").value),
            "total_steps": int(document.getElementById("total_steps").value)
        }
        
        await solve_navier_stokes(params, plot_div)
        
    except Exception as e:
        plot_div.innerHTML = f"Erro: {e}"
        print(e)
    finally:
        document.getElementById("stop-btn").classList.add("hidden")
        document.getElementById("stop-btn").classList.remove("flex")
        document.getElementById("run-btn").classList.remove("hidden")
        document.getElementById("run-btn").classList.add("flex")
