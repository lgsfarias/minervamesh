import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.sparse.linalg import spsolve
from pyscript import when, document
import asyncio
from common import generate_mesh_generic, assemble_matrices, plot_to_base64

# ==============================
# 1. Geometrias
# ==============================
def get_geometry(geo_type, params):
    cx, cy = params["cx"], params["cy"]
    
    if geo_type == "cylinder":
        r = params["r"]
        n_pts = 100
        theta = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
        x_bound = cx + r * np.cos(theta)
        y_bound = cy + r * np.sin(theta)
        
        def mask_func(x, y):
            return (x - cx)**2 + (y - cy)**2 > (r * 1.01)**2
            
        return np.column_stack([x_bound, y_bound]), mask_func
        
    elif geo_type == "square":
        side = params["side"]
        angle = np.radians(params["angle"]) # Rotação
        
        # Pontos do quadrado (4 cantos + subdivisões)
        n_side = 25
        pts = []
        # Local coords: [-s/2, s/2]
        s2 = side / 2
        # Bottom
    elif geo_type == "airfoil" or geo_type == "naca4412":
        # General NACA 4-Digit Series Generator
        # M: Max camber (1st digit)
        # P: Max camber pos (2nd digit)
        # T: Max thickness (3rd, 4th digits)
        
        if geo_type == "naca4412":
            M = 0.04
            P = 0.40
            T = 0.12
        else: # Default 0012
            M = 0.00
            P = 0.00
            T = 0.12
            
        chord = params["chord"]
        angle = np.radians(params["angle"])
        
        n_pts = 100
        beta = np.linspace(0, np.pi, n_pts)
        xc = 0.5 * (1 - np.cos(beta)) # Cosine spacing [0, 1]
        
        # Thickness distribution (yt)
        yt = 5 * T * (0.2969*np.sqrt(xc) - 0.1260*xc - 0.3516*xc**2 + 0.2843*xc**3 - 0.1015*xc**4)
        
        # Camber line (yc) and gradient (dyc_dx)
        yc = np.zeros_like(xc)
        dyc_dx = np.zeros_like(xc)
        
        if M > 0:
            # Forward of max camber (0 <= x <= P)
            idx1 = xc <= P
            yc[idx1] = (M / P**2) * (2*P*xc[idx1] - xc[idx1]**2)
            dyc_dx[idx1] = (2*M / P**2) * (P - xc[idx1])
            
            # Aft of max camber (P < x <= 1)
            idx2 = xc > P
            yc[idx2] = (M / (1-P)**2) * ((1-2*P) + 2*P*xc[idx2] - xc[idx2]**2)
            dyc_dx[idx2] = (2*M / (1-P)**2) * (P - xc[idx2])
            
        theta_c = np.arctan(dyc_dx)
        
        # Upper and Lower surface coordinates
        x_upper = xc - yt * np.sin(theta_c)
        y_upper = yc + yt * np.cos(theta_c)
        x_lower = xc + yt * np.sin(theta_c)
        y_lower = yc - yt * np.cos(theta_c)
        
        # Scale by chord
        x_upper *= chord
        y_upper *= chord
        x_lower *= chord
        y_lower *= chord
        
        # Combine (trailing edge to leading edge to trailing edge)
        x_local = np.concatenate([x_upper[::-1], x_lower[1:]])
        y_local = np.concatenate([y_upper[::-1], y_lower[1:]])
        
        # Center at cx, cy (approximate center of chord)
        x_local -= chord / 2
        
        # Rotate and Translate
        c, s = np.cos(angle), np.sin(angle)
        x_rot = c * x_local - s * y_local + cx
        y_rot = s * x_local + c * y_local + cy
        
        from matplotlib.path import Path
        poly_path = Path(np.column_stack([x_rot, y_rot]))
        
        def mask_func(x, y):
            pts = np.column_stack([x, y])
            return ~poly_path.contains_points(pts, radius=-1e-3)
            
        return np.column_stack([x_rot, y_rot]), mask_func

    return None, None

# ==============================
# 2. Solver Potencial
# ==============================
async def solve_potential(params, plot_div):
    # 1. Geometria e Malha
    boundary_pts, mask_func = get_geometry(params["geo_type"], params)
    
    X, Y, triang = generate_mesh_generic(
        params["L"], params["H"], boundary_pts, mask_func, 
        params["cx"], params["cy"], params["nx"], params["ny"]
    )
    n_nodes = len(X)
    
    # 2. Matrizes
    K, M = assemble_matrices(X, Y, triang)
    
    # 3. Fronteiras
    eps = 1e-5
    inlet_nodes = np.where(X < eps)[0]
    outlet_nodes = np.where(X > params["L"] - eps)[0]
    wall_bottom = np.where(Y < eps)[0]
    wall_top = np.where(Y > params["H"] - eps)[0]
    
    # Obstacle nodes: são os últimos N pontos em X, Y (pois concatenamos)
    n_bound = len(boundary_pts)
    obstacle_nodes = np.arange(n_nodes - n_bound, n_nodes)
    
    # 4. Condições de Contorno
    psi = np.zeros(n_nodes)
    
    # Normalizado para Psi(H) = 1
    def get_psi_inlet_normalized(y_val):
        return y_val / params["H"]
    
    psi[inlet_nodes] = get_psi_inlet_normalized(Y[inlet_nodes])
    psi[wall_bottom] = 0.0
    psi[wall_top] = 1.0
    
    # Obstacle: Psi constante + Gamma
    # Qual valor base? Psi no centro do obstáculo (aproximado)
    psi_base = get_psi_inlet_normalized(params["cy"])
    # Hardcode Gamma = 0.0 (Pure Potential Flow)
    params["Gamma"] = 0.0
    psi[obstacle_nodes] = psi_base + params["Gamma"]
    
    # Dirichlet
    dirichlet_psi = np.concatenate([inlet_nodes, wall_bottom, wall_top, obstacle_nodes])
    dirichlet_psi = np.unique(dirichlet_psi)
    free_psi = np.setdiff1d(np.arange(n_nodes), dirichlet_psi)
    
    # Solver
    K_free = K[free_psi, :][:, free_psi]
    b_free = -K[free_psi, :][:, dirichlet_psi] @ psi[dirichlet_psi]
    psi[free_psi] = spsolve(K_free, b_free)
    
    # --- CÁLCULO AERODINÂMICO ---
    # 1. Velocidade nos nós
    # Interpolador linear para gradientes (velocidade) nos elementos
    tci = tri.LinearTriInterpolator(triang, psi)
    (dpsi_dx, dpsi_dy) = tci.gradient(X, Y)
    u_nodes = dpsi_dy
    v_nodes = -dpsi_dx
    
    # Velocidade de referência (Inlet)
    U_inf = 1.0 / params["H"]
    V_sq = u_nodes**2 + v_nodes**2
    Cp = 1.0 - V_sq / (U_inf**2)
    
    # 2. Extrair dados na superfície do obstáculo
    # obstacle_nodes já temos. Precisamos ordenar para plotar e integrar.
    # Ordenar pelo ângulo em relação ao centro (cx, cy) funciona para todos (convexos)
    obs_idx = obstacle_nodes
    x_obs = X[obs_idx]
    y_obs = Y[obs_idx]
    cp_obs = Cp[obs_idx]
    
    angles = np.arctan2(y_obs - params["cy"], x_obs - params["cx"])
    sorted_indices = np.argsort(angles)
    
    x_sorted = x_obs[sorted_indices]
    y_sorted = y_obs[sorted_indices]
    cp_sorted = cp_obs[sorted_indices]
    angles_sorted = angles[sorted_indices]
    
    # 3. Integração Numérica para CL e CD
    # F = Integral(-Cp * n * ds)
    # Aproximação por segmentos lineares entre pontos ordenados
    CL_num = 0.0
    CD_num = 0.0
    
    # Fechar o loop (último conecta ao primeiro)
    x_loop = np.append(x_sorted, x_sorted[0])
    y_loop = np.append(y_sorted, y_sorted[0])
    cp_loop = np.append(cp_sorted, cp_sorted[0])
    
    for i in range(len(x_sorted)):
        dx = x_loop[i+1] - x_loop[i]
        dy = y_loop[i+1] - y_loop[i]
        ds = np.sqrt(dx*dx + dy*dy)
        
        # Normal (rotacionar tangente 90 graus para fora)
        # Tangente: (dx, dy). Normal: (dy, -dx) (Verificar sentido horário/anti-horário)
        # Pontos ordenados por ângulo (-pi a pi) -> Sentido Anti-Horário
        # Tangente aponta no sentido anti-horário.
        # Normal para fora deve ser (dy, -dx).
        nx = dy / ds
        ny = -dx / ds
        
        cp_avg = 0.5 * (cp_loop[i] + cp_loop[i+1])
        
        # Força = -Cp * n * ds
        CD_num += -cp_avg * nx * ds
        CL_num += -cp_avg * ny * ds
        
    # Normalizar pela corda/diâmetro?
    # Cp já é adimensional. A integral dá Força/ (0.5 rho U^2).
    # Para obter CL, dividimos pela corda (ou diâmetro).
    if params["geo_type"] == "cylinder":
        ref_len = 2 * params["r"]
    elif params["geo_type"] == "square":
        ref_len = params["side"] # Aproximado
    elif params["geo_type"] == "airfoil" or params["geo_type"] == "naca4412":
        ref_len = params["chord"]
    else:
        ref_len = 1.0
        
    CL_num /= ref_len
    CD_num /= ref_len

    # --- VISUALIZAÇÃO ---
    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1.5, 1])
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    # 1. Função Corrente + Malha
    contour = ax1.tricontourf(triang, psi, levels=20, cmap='jet')
    ax1.tricontour(triang, psi, levels=20, colors='k', linewidths=0.5, alpha=0.3)
    ax1.triplot(triang, color='k', alpha=0.1, linewidth=0.5)
    ax1.set_title(f"Função Corrente ($\\psi$) - {params['geo_type'].capitalize()}")
    ax1.set_aspect('equal')
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    fig.colorbar(contour, ax=ax1)
    
    # 2. Linhas de Corrente
    xi = np.linspace(0, params["L"], 100)
    yi = np.linspace(0, params["H"], 100)
    Xi, Yi = np.meshgrid(xi, yi)
    (dpsi_dx_grid, dpsi_dy_grid) = tci.gradient(Xi, Yi)
    u_grid = dpsi_dy_grid
    v_grid = -dpsi_dx_grid
    vel_mag_grid = np.sqrt(u_grid**2 + v_grid**2)
    strm = ax2.streamplot(Xi, Yi, u_grid, v_grid, color=vel_mag_grid, cmap='jet', density=1.5, linewidth=1, arrowsize=1.5)
    ax2.set_title("Linhas de Corrente")
    ax2.set_aspect('equal')
    ax2.set_xlim(0, params["L"])
    ax2.set_ylim(0, params["H"])
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    fig.colorbar(strm.lines, ax=ax2, label='Velocidade')
    
    # 3. Distribuição de Cp
    if params["geo_type"] == "cylinder":
        # Plotar vs Theta (graus)
        theta_deg = np.degrees(angles_sorted)
        ax3.plot(theta_deg, cp_sorted, 'b-', label='Numérico', linewidth=2)
        ax3.set_xlabel(r'$\theta$ (graus)')
        ax3.set_xlim(-180, 180)
        
        # Analítico (Potencial sem circulação): Cp = 1 - 4sin^2(theta)
        theta_ana = np.linspace(-np.pi, np.pi, 200)
        cp_ana = 1 - 4 * np.sin(theta_ana)**2
        ax3.plot(np.degrees(theta_ana), cp_ana, 'r--', label='Analítico (Teórico)', alpha=0.7)
            
    elif params["geo_type"] == "airfoil" or params["geo_type"] == "naca4412":
        # Plotar vs x/c
        # Separar extradorso (upper) e intradorso (lower)
        # Upper: y > cy (aprox, ou usar ângulo)
        # Rotacionar de volta para achar x ao longo da corda?
        # Simplificação: usar x_sorted
        ax3.plot(x_sorted, cp_sorted, 'b.-', label='Cp Numérico')
        ax3.set_xlabel('x [m]')
        
    else:
        ax3.plot(angles_sorted, cp_sorted, 'b.-')
        ax3.set_xlabel('Ângulo (rad)')

    ax3.set_ylabel(r'$C_p$ (adimensional)')
    ax3.set_title(r'Distribuição de Coeficiente de Pressão ($C_p$)')
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis() # Convenção aerodinâmica: Cp negativo para cima
    ax3.legend()
    
    plt.tight_layout()
    
    img_base64 = plot_to_base64(fig)
    plot_div.innerHTML = f'<img src="data:image/png;base64,{img_base64}" class="max-w-full h-auto rounded shadow-lg" />'
    
    # Exibir Resultados Numéricos (Apenas Numérico)
    res_html = f"""
    <div class="grid grid-cols-1 gap-4 text-sm">
        <div class="bg-green-50 p-2 rounded text-center">
            <p class="font-bold">Numérico (Integração Cp)</p>
            <p>CL: {CL_num:.3f} (adimensional)</p>
            <p>CD: {CD_num:.3f} (adimensional)</p>
        </div>
    </div>
    """
    document.getElementById("lift-results").innerHTML = res_html

# ==============================
# 3. Interface Handler
# ==============================
@when("click", "#run-btn")
async def run_handler(event=None):
    plot_div = document.getElementById("plot-output")
    plot_div.innerHTML = "Calculando..."
    await asyncio.sleep(0.1)
    
    try:
        params = {
            "L": float(document.getElementById("L").value),
            "H": float(document.getElementById("H").value),
            "cx": float(document.getElementById("cx").value),
            "cy": float(document.getElementById("cy").value),
            "nx": int(document.getElementById("nx").value),
            "ny": int(document.getElementById("ny").value),
            # "Gamma": float(document.getElementById("Gamma").value), # Removed
            "geo_type": document.getElementById("geo_type").value
        }
        
        if params["geo_type"] == "cylinder":
            params["r"] = float(document.getElementById("r").value)
        elif params["geo_type"] == "airfoil" or params["geo_type"] == "naca4412":
            params["chord"] = float(document.getElementById("chord").value)
            params["angle"] = float(document.getElementById("angle_airfoil").value)
            
        await solve_potential(params, plot_div)
        
    except Exception as e:
        plot_div.innerHTML = f"Erro: {e}"
        print(e)
