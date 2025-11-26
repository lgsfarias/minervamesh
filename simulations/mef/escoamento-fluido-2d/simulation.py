#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulação de Escoamento de Fluido 2D (Função Corrente-Vorticidade)
Regime Permanente
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.sparse import lil_matrix, csr_matrix, eye
from scipy.sparse.linalg import spsolve
from pyscript import when, document
import io
import base64
import asyncio

# ==============================
# 1. Geração da Malha
# ==============================
def generate_mesh(L, H, cx, cy, r, nx, ny):
    # 1. Grid de fundo
    x = np.linspace(0, L, nx)
    y = np.linspace(0, H, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    h_mesh = min(dx, dy) # Tamanho característico do elemento
    
    Xg, Yg = np.meshgrid(x, y)
    X_flat = Xg.flatten()
    Y_flat = Yg.flatten()
    
    # 2. Nós da fronteira do obstáculo
    # Número de pontos baseado no perímetro e tamanho do elemento
    n_circ = int(2 * np.pi * r / h_mesh)
    if n_circ < 10: n_circ = 10 # Mínimo de pontos
    
    theta = np.linspace(0, 2*np.pi, n_circ, endpoint=False)
    x_circ = cx + r * np.cos(theta)
    y_circ = cy + r * np.sin(theta)
    
    # 3. Filtrar nós do grid
    # Remover nós que estão DENTRO do círculo ou MUITO PRÓXIMOS da borda
    # Buffer para evitar triângulos muito finos (slivers)
    buffer = 0.3 * h_mesh
    dist = np.sqrt((X_flat - cx)**2 + (Y_flat - cy)**2)
    mask_keep = dist > (r + buffer)
    
    X_grid_valid = X_flat[mask_keep]
    Y_grid_valid = Y_flat[mask_keep]
    
    # 4. Combinar nós
    X = np.concatenate([X_grid_valid, x_circ])
    Y = np.concatenate([Y_grid_valid, y_circ])
    
    # 5. Triangulação
    triang = tri.Triangulation(X, Y)
    
    # 6. Mascarar triângulos espúrios (dentro do obstáculo)
    # Como temos nós na borda, o centróide dos triângulos internos estará < r
    x_tri = X[triang.triangles].mean(axis=1)
    y_tri = Y[triang.triangles].mean(axis=1)
    dist_tri = np.sqrt((x_tri - cx)**2 + (y_tri - cy)**2)
    
    # Tolerância pequena para garantir que não mascaramos triângulos válidos da borda
    triang.set_mask(dist_tri < r - 1e-5)
    
    return X, Y, triang

# ==============================
# 2. Matrizes MEF (Triângulo Linear)
# ==============================
def element_matrices(coords):
    x0, x1, x2 = coords
    detJ = (x1[0] - x0[0]) * (x2[1] - x0[1]) - (x2[0] - x0[0]) * (x1[1] - x0[1])
    area = 0.5 * abs(detJ)
    
    b = np.array([x1[1] - x2[1], x2[1] - x0[1], x0[1] - x1[1]])
    c = np.array([x2[0] - x1[0], x0[0] - x2[0], x1[0] - x0[0]])
    
    ke = (1.0 / (4 * area)) * (np.outer(b, b) + np.outer(c, c))
    me = (area / 12.0) * (np.ones((3, 3)) + np.eye(3))
    
    return ke, me, area, b, c

def assemble_matrices(X, Y, triang):
    n_nodes = len(X)
    K = lil_matrix((n_nodes, n_nodes))
    M = lil_matrix((n_nodes, n_nodes))
    
    elements = triang.triangles
    mask = triang.mask if triang.mask is not None else np.zeros(len(elements), dtype=bool)
    
    for i, el in enumerate(elements):
        if mask[i]: continue
        
        coords = np.column_stack((X[el], Y[el]))
        ke, me, area, b, c = element_matrices(coords)
        
        for row in range(3):
            for col in range(3):
                K[el[row], el[col]] += ke[row, col]
                M[el[row], el[col]] += me[row, col]
                
    return K.tocsr(), M.tocsr()

# ==============================
# 3. Solver
# ==============================
# ==============================
# 3. Solver Logic
# ==============================

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

def get_params():
    try:
        mode_els = document.getElementsByName("sim_mode")
        mode = "potential"
        for el in mode_els:
            if el.checked:
                mode = el.value
                break
                
        params = {
            "mode": mode,
            "L": float(document.getElementById("L").value),
            "H": float(document.getElementById("H").value),
            "cx": float(document.getElementById("cx").value),
            "cy": float(document.getElementById("cy").value),
            "r": float(document.getElementById("r").value),
            "nx": int(document.getElementById("nx").value),
            "ny": int(document.getElementById("ny").value),
        }
        
        if mode == "potential":
            params["Gamma"] = float(document.getElementById("Gamma").value)
            # Re is just for display in potential
            params["Re"] = float(document.getElementById("Re").value) 
        else:
            params["Re"] = float(document.getElementById("Re").value)
            params["dt"] = float(document.getElementById("dt").value)
            params["steps_per_frame"] = int(document.getElementById("steps_per_frame").value)
            params["total_steps"] = int(document.getElementById("total_steps").value)
            
        return params
    except Exception as e:
        print(f"Erro ao ler parâmetros: {e}")
        return None

async def solve_potential(params, plot_div):
    # 1. Malha
    X, Y, triang = generate_mesh(params["L"], params["H"], params["cx"], params["cy"], params["r"], params["nx"], params["ny"])
    n_nodes = len(X)
    
    # 2. Matrizes
    K, M = assemble_matrices(X, Y, triang)
    
    # 3. Identificar Fronteiras
    eps = 1e-5
    inlet_nodes = np.where(X < eps)[0]
    outlet_nodes = np.where(X > params["L"] - eps)[0]
    wall_bottom = np.where(Y < eps)[0]
    wall_top = np.where(Y > params["H"] - eps)[0]
    
    dist_obs = np.sqrt((X - params["cx"])**2 + (Y - params["cy"])**2)
    obstacle_nodes = np.where(np.abs(dist_obs - params["r"]) < 1e-4)[0] 
    
    # 4. Condições de Contorno e Inicialização
    psi = np.zeros(n_nodes)
    
    # Perfil de entrada Uniforme (u = cte => Psi = u*y)
    # Normalizado para Psi(H) = 1
    def get_psi_inlet_normalized(y_val):
        return y_val / params["H"]
    
    # Aplicar CC em Psi (Normalizado)
    psi[inlet_nodes] = get_psi_inlet_normalized(Y[inlet_nodes])
    psi[wall_bottom] = 0.0
    psi[wall_top] = 1.0
    
    # Obstacle: psi constante + Circulação (Gamma)
    psi_obs_val = get_psi_inlet_normalized(params["cy"]) + params["Gamma"]
    psi[obstacle_nodes] = psi_obs_val
    
    # Nós com Dirichlet para Psi
    dirichlet_psi = np.concatenate([inlet_nodes, wall_bottom, wall_top, obstacle_nodes])
    dirichlet_psi = np.unique(dirichlet_psi)
    free_psi = np.setdiff1d(np.arange(n_nodes), dirichlet_psi)
    
    # --- SOLVER POTENCIAL (Laplace Psi = 0) ---
    K_free = K[free_psi, :][:, free_psi]
    b_free = -K[free_psi, :][:, dirichlet_psi] @ psi[dirichlet_psi]
    psi[free_psi] = spsolve(K_free, b_free)
    
    # --- CÁLCULO DE VORTICIDADE ---
    M_lumped = np.array(M.sum(axis=1)).flatten()
    omega = (K @ psi) / M_lumped
    
    # --- VISUALIZAÇÃO ---
    plot_results(X, Y, triang, psi, omega, params["L"], params["H"], plot_div, "Escoamento Potencial")

# ==============================
# 4. Navier-Stokes Helpers
# ==============================

def assemble_convection(X, Y, triang, u_el, v_el):
    """
    Monta a matriz de convecção C tal que C * omega aproxima (u.grad)omega
    Usando formulação Galerkin padrão: C_ij = Integral(phi_i * (u . grad phi_j))
    u é assumido constante no elemento (u_el, v_el)
    """
    n_nodes = len(X)
    C = lil_matrix((n_nodes, n_nodes))
    
    elements = triang.triangles
    mask = triang.mask if triang.mask is not None else np.zeros(len(elements), dtype=bool)
    
    # Pre-compute gradients of basis functions for all elements
    # x0, x1, x2 are coordinates of vertices
    # b_i = y_j - y_k, c_i = x_k - x_j
    # grad phi_i = [b_i/(2A), c_i/(2A)]
    
    for i, el in enumerate(elements):
        if mask[i]: continue
        
        # Coordinates
        x = X[el]
        y = Y[el]
        
        # Area and gradients
        detJ = (x[1] - x[0]) * (x[2] - x[1]) - (x[2] - x[0]) * (x[1] - x[0]) # Wait, formula check
        # Correct Area formula: 0.5 * |x0(y1-y2) + x1(y2-y0) + x2(y0-y1)|
        area = 0.5 * np.abs(x[0]*(y[1]-y[2]) + x[1]*(y[2]-y[0]) + x[2]*(y[0]-y[1]))
        
        # b and c coefficients for gradients
        # N1 = (a1 + b1*x + c1*y)/(2A)
        b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
        c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
        
        # Velocity in element
        ue = u_el[i]
        ve = v_el[i]
        
        # Element convection matrix
        # C_ij^e = Integral(N_i * (ue * dNj/dx + ve * dNj/dy)) dA
        # The term (ue * dNj/dx + ve * dNj/dy) is CONSTANT in the element
        # Let K_j = (ue * b_j + ve * c_j) / (2 * Area)
        # Then C_ij^e = K_j * Integral(N_i) dA = K_j * (Area / 3)
        
        K_dot_grad = (ue * b + ve * c) / (2 * area)
        
        ce = np.outer(np.ones(3) * (area / 3.0), K_dot_grad)
        
        for row in range(3):
            for col in range(3):
                C[el[row], el[col]] += ce[row, col]
                
    return C.tocsr()

def get_boundary_normals(X, Y, boundary_nodes, triang):
    """
    Para cada nó de fronteira, encontra um nó vizinho interno para aplicar Thom's formula.
    Retorna indices dos vizinhos e distâncias quadradas.
    """
    neighbors = []
    dists_sq = []
    
    # Build adjacency list
    adj = [set() for _ in range(len(X))]
    for el in triang.triangles:
        if triang.mask is not None and triang.mask[np.where(triang.triangles == el)[0][0]]: continue # Skip masked? No, mask is by index
        # Actually triang.mask is boolean array matching triangles
        # Need correct index. enumerate is safer.
        pass

    # Better: use matplotlib.tri.TriAnalyzer or just brute force for small mesh
    # Let's iterate triangles
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
        # Find a neighbor NOT in the same boundary set (internal)
        # Or just any neighbor if strictly internal is hard (corners)
        # Prefer internal nodes.
        
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
            # Fallback: use any neighbor (corner case)
            for nb in node_neighbors:
                d = (X[node]-X[nb])**2 + (Y[node]-Y[nb])**2
                if d < min_dist:
                    min_dist = d
                    best_nb = nb
                    
        valid_neighbors[i] = best_nb
        h_sq[i] = min_dist
        
    return valid_neighbors, h_sq

async def solve_navier_stokes(params, plot_div):
    # 1. Malha
    X, Y, triang = generate_mesh(params["L"], params["H"], params["cx"], params["cy"], params["r"], params["nx"], params["ny"])
    n_nodes = len(X)
    
    # 2. Matrizes Constantes (K e M)
    K, M = assemble_matrices(X, Y, triang)
    
    # 3. Fronteiras
    eps = 1e-5
    inlet_nodes = np.where(X < eps)[0]
    outlet_nodes = np.where(X > params["L"] - eps)[0]
    wall_bottom = np.where(Y < eps)[0]
    wall_top = np.where(Y > params["H"] - eps)[0]
    
    dist_obs = np.sqrt((X - params["cx"])**2 + (Y - params["cy"])**2)
    obstacle_nodes = np.where(np.abs(dist_obs - params["r"]) < 1e-4)[0]
    
    # Paredes sólidas (Top, Bottom, Obstacle)
    solid_walls = np.concatenate([wall_bottom, wall_top, obstacle_nodes])
    solid_walls = np.unique(solid_walls)
    
    # Preparar Thom's formula para solid_walls
    wall_neighbors, wall_h_sq = get_boundary_normals(X, Y, solid_walls, triang)
    
    # 4. Inicialização
    psi = np.zeros(n_nodes)
    omega = np.zeros(n_nodes)
    
    # Condição Inicial: Escoamento Potencial (Opcional, mas ajuda a convergir)
    # Por enquanto, começar do zero ou perfil uniforme
    # Perfil de entrada
    def get_psi_inlet(y_val):
        return y_val # u=1
        
    psi[inlet_nodes] = get_psi_inlet(Y[inlet_nodes])
    psi[wall_bottom] = 0.0
    psi[wall_top] = params["H"] # u=1 * H
    psi[obstacle_nodes] = get_psi_inlet(params["cy"]) # Valor da linha de corrente no centro do obstáculo
    
    # Dirichlet para Psi
    dirichlet_psi = np.concatenate([inlet_nodes, wall_bottom, wall_top, obstacle_nodes])
    dirichlet_psi = np.unique(dirichlet_psi)
    free_psi = np.setdiff1d(np.arange(n_nodes), dirichlet_psi)
    
    # Dirichlet para Omega
    # Inlet: omega = 0 (fluxo uniforme)
    # Outlet: Neumann nulo (livre) -> não entra no Dirichlet
    # Walls: Atualizado a cada passo (Thom) -> tratado como Dirichlet no passo
    
    # Matrizes reduzidas para Psi (K é constante)
    K_free_psi = K[free_psi, :][:, free_psi]
    
    # Parâmetros de tempo
    dt = params["dt"]
    Re = params["Re"]
    steps_per_frame = params["steps_per_frame"]
    total_steps = params["total_steps"]
    
    # Loop de Tempo
    current_step = 0
    
    while current_step < total_steps and not STOP_SIMULATION:
        # Loop interno (steps per frame)
        for _ in range(steps_per_frame):
            # A. Resolver Poisson para Psi: K * psi = M * omega
            # M * omega (lado direito)
            rhs_psi = M @ omega
            
            # Aplicar BCs em Psi
            # Psi nas paredes é fixo (exceto se quisermos recalcular no obstáculo, mas vamos manter fixo 0.5H por simplicidade ou ajustar)
            # Ajuste fino: Psi no obstáculo deve ser constante, mas o valor pode flutuar se houver lift?
            # Para NS fixo, geralmente fixamos o valor se for simétrico. Vamos manter fixo.
            
            b_psi = rhs_psi[free_psi] - K[free_psi, :][:, dirichlet_psi] @ psi[dirichlet_psi]
            psi[free_psi] = spsolve(K_free_psi, b_psi)
            
            # B. Atualizar Omega nas paredes (Thom's Formula)
            # omega_wall = -2 * (psi_neighbor - psi_wall) / h^2
            # psi_neighbor está em psi[wall_neighbors]
            # psi_wall está em psi[solid_walls]
            
            psi_nb = psi[wall_neighbors]
            psi_w = psi[solid_walls]
            omega[solid_walls] = -2.0 * (psi_nb - psi_w) / wall_h_sq
            
            # C. Calcular Velocidade nos Elementos (para Convecção)
            # Precisamos de u, v em cada triângulo para montar C
            # u = dpsi/dy, v = -dpsi/dx
            # Element gradients
            # Usar a mesma lógica de element_matrices mas só gradientes
            # Simplificação: Calcular u, v nos nós e interpolar para o centro do elemento?
            # Ou calcular exato no elemento (psi é linear -> vel é constante)
            
            # Vamos calcular u_el, v_el
            u_el = []
            v_el = []
            elements = triang.triangles
            mask = triang.mask if triang.mask is not None else np.zeros(len(elements), dtype=bool)
            
            # Vectorized calculation of element velocities
            # x0, x1, x2 ...
            # psi0, psi1, psi2
            # u = sum(psi_i * dNi/dy), v = sum(psi_i * -dNi/dx)
            # dNi/dy = c_i / (2A), dNi/dx = b_i / (2A) -> Wait.
            # N_i = (a + b*x + c*y)/2A
            # dNi/dx = b_i/2A, dNi/dy = c_i/2A
            # u = dpsi/dy = sum(psi_i * c_i/2A)
            # v = -dpsi/dx = -sum(psi_i * b_i/2A)
            
            # Need efficient way. Loop is slow in Python.
            # But mesh is small (~2000 nodes).
            
            # Let's try to be semi-efficient
            # We need u_el and v_el for assemble_convection
            
            # Extract coords
            el_nodes = elements[~mask]
            x = X[el_nodes] # (N_el, 3)
            y = Y[el_nodes] # (N_el, 3)
            p = psi[el_nodes] # (N_el, 3)
            
            # b coefficients: b0 = y1-y2
            b = np.column_stack([y[:,1]-y[:,2], y[:,2]-y[:,0], y[:,0]-y[:,1]])
            # c coefficients: c0 = x2-x1
            c = np.column_stack([x[:,2]-x[:,1], x[:,0]-x[:,2], x[:,1]-x[:,0]])
            
            # Area
            area = 0.5 * np.abs(x[:,0]*(y[:,1]-y[:,2]) + x[:,1]*(y[:,2]-y[:,0]) + x[:,2]*(y[:,0]-y[:,1]))
            
            # u = (p0*c0 + p1*c1 + p2*c2) / (2A)
            u_vals = np.sum(p * c, axis=1) / (2 * area)
            # v = -(p0*b0 + p1*b1 + p2*b2) / (2A)
            v_vals = -np.sum(p * b, axis=1) / (2 * area)
            
            # Reconstruct full array including masked (zeros)
            u_el_full = np.zeros(len(elements))
            v_el_full = np.zeros(len(elements))
            u_el_full[~mask] = u_vals
            v_el_full[~mask] = v_vals
            
            # D. Montar Matriz de Convecção C
            C = assemble_convection(X, Y, triang, u_el_full, v_el_full)
            
            # E. Resolver Transporte de Vorticidade
            # (M + dt/Re * K) * omega_new = M * omega_old - dt * C * omega_old
            # BCs: omega fixo em Inlet (0) e Walls (Thom)
            # Outlet: livre (Natural BC -> não faz nada na matriz)
            
            dirichlet_omega = np.concatenate([inlet_nodes, solid_walls])
            dirichlet_omega = np.unique(dirichlet_omega)
            free_omega = np.setdiff1d(np.arange(n_nodes), dirichlet_omega)
            
            # LHS Matrix
            A = M + (dt / Re) * K
            
            # RHS Vector
            rhs = M @ omega - dt * (C @ omega)
            
            # Apply BCs to system
            # Reduce system
            A_free = A[free_omega, :][:, free_omega]
            b_omega = rhs[free_omega] - A[free_omega, :][:, dirichlet_omega] @ omega[dirichlet_omega]
            
            omega[free_omega] = spsolve(A_free, b_omega)
            
            current_step += 1
            
        # Update Plot every frame
        plot_results(X, Y, triang, psi, omega, params["L"], params["H"], plot_div, f"Navier-Stokes (Step {current_step})")
        
        # Yield to UI
        await asyncio.sleep(0.01)
        
    if not STOP_SIMULATION:
        plot_div.innerHTML += '<p class="text-center text-green-600 font-bold mt-2">Simulação Concluída!</p>'
    else:
        plot_div.innerHTML += '<p class="text-center text-red-600 font-bold mt-2">Simulação Parada pelo Usuário.</p>'



def plot_results(X, Y, triang, psi, omega, L, H, plot_div, title_prefix):
    # --- CÁLCULO DE VELOCIDADE ---
    tci = tri.LinearTriInterpolator(triang, psi)
    (dpsi_dx, dpsi_dy) = tci.gradient(X, Y)
    u = dpsi_dy
    v = -dpsi_dx
    
    # Determinar se mostra vorticidade (apenas se não for Potencial)
    is_potential = "Potencial" in title_prefix
    
    if is_potential:
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(8, 10))
        ax2 = None # Não usado
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 14))
    
    # 1. Função Corrente
    contour_psi = ax1.tricontourf(triang, psi, levels=20, cmap='jet')
    ax1.tricontour(triang, psi, levels=20, colors='k', linewidths=0.5, alpha=0.3)
    ax1.triplot(triang, color='k', alpha=0.1, linewidth=0.5)
    ax1.set_title(f"{title_prefix} - Função Corrente ($\\psi$)")
    ax1.set_aspect('equal')
    fig.colorbar(contour_psi, ax=ax1)
    
    # 2. Vorticidade (Apenas Navier-Stokes)
    if not is_potential and ax2 is not None:
        contour_omega = ax2.tricontourf(triang, omega, levels=20, cmap='jet')
        ax2.triplot(triang, color='k', alpha=0.1, linewidth=0.5)
        ax2.set_title(f"{title_prefix} - Vorticidade ($\\omega$)")
        ax2.set_aspect('equal')
        fig.colorbar(contour_omega, ax=ax2)
    
    # 3. Campo de Velocidade (Linhas de Corrente)
    xi = np.linspace(0, L, 100)
    yi = np.linspace(0, H, 100)
    Xi, Yi = np.meshgrid(xi, yi)
    
    (dpsi_dx_grid, dpsi_dy_grid) = tci.gradient(Xi, Yi)
    u_grid = dpsi_dy_grid
    v_grid = -dpsi_dx_grid
    vel_mag_grid = np.sqrt(u_grid**2 + v_grid**2)
    
    strm = ax3.streamplot(Xi, Yi, u_grid, v_grid, color=vel_mag_grid, cmap='jet', density=1.5, linewidth=1, arrowsize=1.5)
    ax3.set_title(f"{title_prefix} - Linhas de Corrente")
    ax3.set_aspect('equal')
    ax3.set_xlim(0, L)
    ax3.set_ylim(0, H)
    fig.colorbar(strm.lines, ax=ax3, label='Magnitude da Velocidade')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    
    plot_div.innerHTML = f'<img src="data:image/png;base64,{img_base64}" class="max-w-full h-auto rounded shadow-lg" />'

@when("click", "#run-btn")
async def run_simulation(event=None):
    global STOP_SIMULATION
    STOP_SIMULATION = False
    
    plot_div = document.getElementById("plot-output")
    plot_div.innerHTML = """
      <div class="flex flex-col items-center justify-center py-6">
        <div class="animate-spin h-10 w-10 border-4 border-blue-500 border-t-transparent rounded-full"></div>
        <p class="mt-4 text-blue-700 font-medium">Iniciando simulação...</p>
      </div>
    """
    await asyncio.sleep(0.1)

    try:
        params = get_params()
        if not params: return

        if params["mode"] == "potential":
            await solve_potential(params, plot_div)
        else:
            # Show stop button for NS
            document.getElementById("run-btn").classList.add("hidden")
            document.getElementById("run-btn").classList.remove("flex")
            document.getElementById("stop-btn").classList.remove("hidden")
            document.getElementById("stop-btn").classList.add("flex")
            
            await solve_navier_stokes(params, plot_div)
            
            # Reset buttons after finish
            document.getElementById("stop-btn").classList.add("hidden")
            document.getElementById("stop-btn").classList.remove("flex")
            document.getElementById("run-btn").classList.remove("hidden")
            document.getElementById("run-btn").classList.add("flex")

    except Exception as e:
        plot_div.innerHTML = f"""
        <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
            <strong class="font-bold">Erro na simulação:</strong>
            <span class="block sm:inline">{str(e)}</span>
        </div>
        """
        print(f"Erro: {e}")

