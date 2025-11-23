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
@when("click", "#run-btn")
async def run_simulation(event=None):
    # UI Feedback
    plot_div = document.getElementById("plot-output")
    plot_div.innerHTML = """
      <div class="flex flex-col items-center justify-center py-6">
        <div class="animate-spin h-10 w-10 border-4 border-blue-500 border-t-transparent rounded-full"></div>
        <p class="mt-4 text-blue-700 font-medium">Calculando escoamento...</p>
      </div>
    """
    await asyncio.sleep(0.1)

    try:
        # Parâmetros
        L = float(document.getElementById("L").value)
        H = float(document.getElementById("H").value)
        cx = float(document.getElementById("cx").value)
        cy = float(document.getElementById("cy").value)
        r = float(document.getElementById("r").value)
        Re = float(document.getElementById("Re").value)
        Gamma = float(document.getElementById("Gamma").value)
        nx = int(document.getElementById("nx").value)
        ny = int(document.getElementById("ny").value)
        max_iter = int(document.getElementById("max_iter").value)
        tol = float(document.getElementById("tolerance").value)
        
        # 1. Malha
        X, Y, triang = generate_mesh(L, H, cx, cy, r, nx, ny)
        n_nodes = len(X)
        
        # 2. Matrizes
        K, M = assemble_matrices(X, Y, triang)
        
        # 3. Identificar Fronteiras
        eps = 1e-5
        inlet_nodes = np.where(X < eps)[0]
        outlet_nodes = np.where(X > L - eps)[0]
        wall_bottom = np.where(Y < eps)[0]
        wall_top = np.where(Y > H - eps)[0]
        
        dist_obs = np.sqrt((X - cx)**2 + (Y - cy)**2)
        # Selecionar apenas nós EXATAMENTE na borda (com pequena tolerância)
        # Como usamos malha conforme, os nós da borda têm dist ~ r
        # Os nós do grid têm dist > r + buffer
        obstacle_nodes = np.where(np.abs(dist_obs - r) < 1e-4)[0] 
        
        # Mapeamento reverso
        x_grid = np.linspace(0, L, nx)
        y_grid = np.linspace(0, H, ny)
        
        # 4. Condições de Contorno e Inicialização
        psi = np.zeros(n_nodes)
        
        # Perfil de entrada Uniforme (u = cte => Psi = u*y)
        # Normalizado para Psi(H) = 1
        def get_psi_inlet_raw(y_val):
            return y_val
            
        psi_total_flux = get_psi_inlet_raw(H)
        
        def get_psi_inlet_normalized(y_val):
            return get_psi_inlet_raw(y_val) / psi_total_flux
        
        # Aplicar CC em Psi (Normalizado)
        psi[inlet_nodes] = get_psi_inlet_normalized(Y[inlet_nodes])
        psi[wall_bottom] = 0.0
        psi[wall_top] = 1.0
        
        # Obstacle: psi constante + Circulação (Gamma)
        # Gamma desloca o valor de Psi no obstáculo, alterando a estagnação
        psi_obs_val = get_psi_inlet_normalized(cy) + Gamma
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
        
        # --- CÁLCULO DE VELOCIDADE ---
        tci = tri.LinearTriInterpolator(triang, psi)
        # Avaliando nos nós
        (dpsi_dx, dpsi_dy) = tci.gradient(X, Y)
        u = dpsi_dy
        v = -dpsi_dx
        vel_mag = np.sqrt(u**2 + v**2)
        
        # Visualização
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 14))
        
        # 1. Função Corrente
        # Usar 'jet' como solicitado
        contour_psi = ax1.tricontourf(triang, psi, levels=20, cmap='jet')
        ax1.tricontour(triang, psi, levels=20, colors='k', linewidths=0.5, alpha=0.3)
        # Mostrar malha fina
        ax1.triplot(triang, color='k', alpha=0.1, linewidth=0.5)
        ax1.set_title(r"Função Corrente ($\psi$)")
        ax1.set_aspect('equal')
        fig.colorbar(contour_psi, ax=ax1)
        
        # 2. Vorticidade
        # Usar 'jet' também para consistência com o pedido, ou manter seismic se fizer sentido físico
        # O usuário pediu "esquema de cores do projeto", que parece ser jet.
        contour_omega = ax2.tricontourf(triang, omega, levels=20, cmap='jet')
        ax2.triplot(triang, color='k', alpha=0.1, linewidth=0.5)
        ax2.set_title(r"Vorticidade ($\omega$)")
        ax2.set_aspect('equal')
        fig.colorbar(contour_omega, ax=ax2)
        
        # 3. Campo de Velocidade (Linhas de Corrente)
        # Criar malha regular para streamplot
        xi = np.linspace(0, L, 100)
        yi = np.linspace(0, H, 100)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # Interpolar gradientes (velocidade) na malha regular
        (dpsi_dx_grid, dpsi_dy_grid) = tci.gradient(Xi, Yi)
        u_grid = dpsi_dy_grid
        v_grid = -dpsi_dx_grid
        vel_mag_grid = np.sqrt(u_grid**2 + v_grid**2)
        
        # Streamplot
        strm = ax3.streamplot(Xi, Yi, u_grid, v_grid, color=vel_mag_grid, cmap='jet', density=1.5, linewidth=1, arrowsize=1.5)
        ax3.set_title(r"Campo de Velocidade (Linhas de Corrente)")
        ax3.set_aspect('equal')
        ax3.set_xlim(0, L)
        ax3.set_ylim(0, H)
        fig.colorbar(strm.lines, ax=ax3, label='Magnitude da Velocidade')
        
        plt.tight_layout()
        
        # Salvar imagem
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        
        plot_div.innerHTML = f'<img src="data:image/png;base64,{img_base64}" class="max-w-full h-auto rounded shadow-lg" />'

    except Exception as e:
        plot_div.innerHTML = f"""
        <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
            <strong class="font-bold">Erro na simulação:</strong>
            <span class="block sm:inline">{str(e)}</span>
        </div>
        """
        print(f"Erro: {e}")
