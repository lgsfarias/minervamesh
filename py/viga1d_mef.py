#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Viga 1D Avançada (MEF) — Problema 5.2.5 Expandido
Análise completa de flexão de vigas usando MEF com elementos de viga de Euler-Bernoulli.
Inclui múltiplos tipos de carregamento, condições de contorno, propriedades variáveis e análises completas.
"""

import numpy as np
import matplotlib.pyplot as plt
from js import document
from pyscript import when
import asyncio
import io
import imageio.v2 as imageio
import base64


# ==============================
# 1. Funções de carregamento
# ==============================
def load_uniform(x, q0, L):
    """Carregamento uniforme"""
    return np.full_like(x, q0)

def load_linear(x, q0, qL, L):
    """Carregamento linearmente variável"""
    return q0 + (qL - q0) * x / L

def load_triangular(x, q_max, L):
    """Carregamento triangular (máximo no centro)"""
    return q_max * (1 - 2 * np.abs(x - L/2) / L)

def load_sinusoidal(x, q0, L):
    """Carregamento senoidal"""
    return q0 * np.sin(np.pi * x / L)

def load_point(x, P, x_pos, L):
    """Carregamento pontual (aproximado como distribuição localizada)"""
    # Aproximação: distribuição gaussiana muito concentrada
    sigma = L / 100  # Largura da distribuição
    return P * np.exp(-0.5 * ((x - x_pos) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))


# ==============================
# 2. Soluções analíticas
# ==============================
def analytical_solution_simply_supported(x, q_func, E, I, L):
    """Solução analítica para viga simplesmente apoiada"""
    if q_func == "uniform":
        q = 1.0  # Valor normalizado
        return (q / (24 * E * I)) * (L**3 * x - 2 * L * x**3 + x**4)
    elif q_func == "triangular":
        q_max = 1.0
        return (q_max / (120 * E * I)) * (L**4 * x - 2 * L**2 * x**3 + x**5)
    return np.zeros_like(x)

def analytical_solution_cantilever(x, q_func, E, I, L):
    """Solução analítica para viga em balanço"""
    if q_func == "uniform":
        q = 1.0
        return (q / (24 * E * I)) * (6 * L**2 * x**2 - 4 * L * x**3 + x**4)
    elif q_func == "triangular":
        q_max = 1.0
        return (q_max / (120 * E * I)) * (10 * L**3 * x**2 - 10 * L**2 * x**3 + 5 * L * x**4 - x**5)
    return np.zeros_like(x)

def analytical_solution_fixed_fixed(x, q_func, E, I, L):
    """Solução analítica para viga engastada-engastada"""
    if q_func == "uniform":
        q = 1.0
        return (q / (24 * E * I)) * (L**2 * x**2 - 2 * L * x**3 + x**4)
    return np.zeros_like(x)


# ==============================
# 3. Funções auxiliares para propriedades variáveis
# ==============================
def get_variable_properties(x, prop_type, base_value, L):
    """Calcula propriedades variáveis ao longo da viga"""
    if prop_type == "constant":
        return np.full_like(x, base_value)
    elif prop_type == "linear":
        # Propriedade varia linearmente de base_value a 0.5*base_value
        return base_value * (1 - 0.5 * x / L)
    elif prop_type == "parabolic":
        # Propriedade varia parabolicamente (máximo no centro)
        return base_value * (1 - 0.3 * (x - L/2)**2 / (L/2)**2)
    return np.full_like(x, base_value)


# ==============================
# 4. Executar simulação MEF avançada para viga 1D
# ==============================
@when("click", "#runViga1D")
async def run_viga_1d(event=None):
    # Abrir modal de execução
    sim_loading = document.getElementById("sim-loading")
    try:
        sim_loading.showModal()
    except Exception:
        pass
    await asyncio.sleep(0.05)

    # Parâmetros básicos
    L = float(document.getElementById("L").value)
    nel = int(document.getElementById("nel").value)
    E_base = float(document.getElementById("E").value) * 1e9  # Converter para Pa
    I_base = float(document.getElementById("I").value) * 1e-8  # Converter para m⁴
    q_magnitude = float(document.getElementById("q").value) * 1000  # Converter para N/m
    bc_type = document.getElementById("bc_type").value
    load_type = document.getElementById("load_type").value
    prop_type = document.getElementById("prop_type").value
    show_analytical = document.getElementById("show_analytical").checked

    # Malha 1D
    nn = nel + 1
    x = np.linspace(0.0, L, nn)
    h = L / nel

    # Propriedades variáveis
    E = get_variable_properties(x, prop_type, E_base, L)
    I = get_variable_properties(x, prop_type, I_base, L)

    # Carregamento
    if load_type == "uniform":
        q = load_uniform(x, q_magnitude, L)
    elif load_type == "linear":
        q0 = q_magnitude
        qL = 0.5 * q_magnitude
        q = load_linear(x, q0, qL, L)
    elif load_type == "triangular":
        q = load_triangular(x, q_magnitude, L)
    elif load_type == "sinusoidal":
        q = load_sinusoidal(x, q_magnitude, L)
    elif load_type == "point":
        P = q_magnitude * L  # Força pontual equivalente
        x_pos = L / 2  # Posição da força
        q = load_point(x, P, x_pos, L)
    else:
        q = load_uniform(x, q_magnitude, L)

    # Para elementos de viga, cada nó tem 2 DOFs: deslocamento vertical (w) e rotação (θ)
    ndof = 2 * nn
    K = np.zeros((ndof, ndof))
    f = np.zeros(ndof)

    # Montagem da matriz global com propriedades variáveis
    for e in range(nel):
        # Propriedades do elemento (média dos nós)
        E_elem = (E[e] + E[e+1]) / 2
        I_elem = (I[e] + I[e+1]) / 2
        EI = E_elem * I_elem
        
        # Carregamento do elemento (média dos nós)
        q_elem = (q[e] + q[e+1]) / 2
        
        # Matriz de rigidez do elemento
        ke = (EI / h**3) * np.array([
            [12,    6*h,   -12,    6*h],
            [6*h,   4*h**2, -6*h,   2*h**2],
            [-12,   -6*h,   12,    -6*h],
            [6*h,   2*h**2, -6*h,   4*h**2]
        ])

        # Vetor de forças do elemento
        fe = (q_elem * h / 12) * np.array([6, h, 6, -h])

        # DOFs do elemento
        dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
        
        # Montar matriz de rigidez
        for i in range(4):
            for j in range(4):
                K[dofs[i], dofs[j]] += ke[i, j]
        
        # Montar vetor de forças
        for i in range(4):
            f[dofs[i]] += fe[i]

    # Aplicar condições de contorno
    if bc_type == "simply_supported":
        # Simplesmente apoiada: w(0) = 0, w(L) = 0
        K[0, :] = 0; K[0, 0] = 1; f[0] = 0
        K[2*nn-2, :] = 0; K[2*nn-2, 2*nn-2] = 1; f[2*nn-2] = 0
        
    elif bc_type == "cantilever":
        # Viga em balanço: w(0) = 0, θ(0) = 0
        K[0, :] = 0; K[0, 0] = 1; f[0] = 0
        K[1, :] = 0; K[1, 1] = 1; f[1] = 0
        
    elif bc_type == "fixed_fixed":
        # Engastada-engastada: w(0) = 0, θ(0) = 0, w(L) = 0, θ(L) = 0
        K[0, :] = 0; K[0, 0] = 1; f[0] = 0
        K[1, :] = 0; K[1, 1] = 1; f[1] = 0
        K[2*nn-2, :] = 0; K[2*nn-2, 2*nn-2] = 1; f[2*nn-2] = 0
        K[2*nn-1, :] = 0; K[2*nn-1, 2*nn-1] = 1; f[2*nn-1] = 0
        
    elif bc_type == "fixed_supported":
        # Engastada-apoiada: w(0) = 0, θ(0) = 0, w(L) = 0
        K[0, :] = 0; K[0, 0] = 1; f[0] = 0
        K[1, :] = 0; K[1, 1] = 1; f[1] = 0
        K[2*nn-2, :] = 0; K[2*nn-2, 2*nn-2] = 1; f[2*nn-2] = 0

    # Resolver sistema
    u = np.linalg.solve(K, f)

    # Extrair deslocamentos verticais e rotações
    w = u[::2]  # deslocamentos verticais
    theta = u[1::2]  # rotações

    # Calcular esforços internos
    M = np.zeros(nn)  # Momentos fletores
    V = np.zeros(nn)  # Esforços cortantes
    
    for i in range(1, nn-1):
        # Momento fletor: M = -EI * d²w/dx²
        EI_avg = (E[i] * I[i] + E[i-1] * I[i-1] + E[i+1] * I[i+1]) / 3
        M[i] = -EI_avg * (w[i+1] - 2*w[i] + w[i-1]) / h**2
        
        # Esforço cortante: V = -EI * d³w/dx³ ≈ -EI * dM/dx
        if i > 1 and i < nn-2:
            V[i] = -(M[i+1] - M[i-1]) / (2*h)

    # Solução analítica para comparação (se disponível)
    x_dense = np.linspace(0.0, L, max(600, min(2400, 8 * nn)))
    w_ana_dense = np.zeros_like(x_dense)
    
    if show_analytical and prop_type == "constant":
        if bc_type == "simply_supported" and load_type == "uniform":
            # Solução analítica para viga simplesmente apoiada com carregamento uniforme
            q_ana = q_magnitude  # Usar a magnitude real do carregamento
            w_ana_dense = (q_ana / (24 * E_base * I_base)) * (L**3 * x_dense - 2 * L * x_dense**3 + x_dense**4)
        elif bc_type == "cantilever" and load_type == "uniform":
            # Solução analítica para viga em balanço com carregamento uniforme
            q_ana = q_magnitude
            w_ana_dense = (q_ana / (24 * E_base * I_base)) * (6 * L**2 * x_dense**2 - 4 * L * x_dense**3 + x_dense**4)
        elif bc_type == "fixed_fixed" and load_type == "uniform":
            # Solução analítica para viga engastada-engastada com carregamento uniforme
            q_ana = q_magnitude
            w_ana_dense = (q_ana / (24 * E_base * I_base)) * (L**2 * x_dense**2 - 2 * L * x_dense**3 + x_dense**4)

    # Determinar título da condição de contorno
    bc_titles = {
        "simply_supported": "Simplesmente Apoiada",
        "cantilever": "Em Balanço", 
        "fixed_fixed": "Engastada-Engastada",
        "fixed_supported": "Engastada-Apoiada"
    }
    bc_title = bc_titles.get(bc_type, "Desconhecida")

    # Determinar título do carregamento
    load_titles = {
        "uniform": "Uniforme",
        "linear": "Linear",
        "triangular": "Triangular",
        "sinusoidal": "Senoidal",
        "point": "Pontual"
    }
    load_title = load_titles.get(load_type, "Desconhecido")

    # Plot avançado com múltiplos gráficos
    fig = plt.figure(figsize=(14, 12))
    
    # Layout: 3x2 grid
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])
    
    # 1. Deslocamentos
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(x, w*1000, 'o-', label='MEF', color='#3B82F6', markersize=4, linewidth=2)
    if show_analytical and np.any(w_ana_dense):
        ax1.plot(x_dense, w_ana_dense*1000, '--', label='Analítica', color='#EF4444', linewidth=2)
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('Deslocamento [mm]')
    ax1.set_title(f'Deslocamentos Verticais - {bc_title} ({load_title})')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    ax1.invert_yaxis()

    # 2. Carregamento
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(x, q/1000, 'g-', linewidth=2, label='Carregamento')
    ax2.fill_between(x, 0, q/1000, alpha=0.3, color='green')
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('Carregamento [kN/m]')
    ax2.set_title('Carregamento Distribuído')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    # 3. Propriedades
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Plotar linha vermelha primeiro (E) com estilo mais visível
    line1 = ax3.plot(x, E/1e9, 'r-', linewidth=4, label='E [GPa]', alpha=0.9, zorder=3)
    
    # Criar eixo secundário para linha azul (I) - tracejada e mais fina
    ax3_twin = ax3.twinx()
    line2 = ax3_twin.plot(x, I*1e8, 'b--', linewidth=2, label='I [cm⁴]', alpha=0.8, zorder=2)
    
    # Configurar eixos e labels
    ax3.set_xlabel('x [m]', fontweight='bold')
    ax3.set_ylabel('Módulo de Elasticidade [GPa]', color='r', fontweight='bold', fontsize=11)
    ax3_twin.set_ylabel('Momento de Inércia [cm⁴]', color='b', fontweight='bold', fontsize=11)
    ax3.set_title('Propriedades Materiais', fontweight='bold', fontsize=12)
    
    # Configurar cores dos ticks
    ax3.tick_params(axis='y', labelcolor='r', labelsize=10)
    ax3_twin.tick_params(axis='y', labelcolor='b', labelsize=10)
    ax3.tick_params(axis='x', labelsize=10)
    
    # Grid mais sutil
    ax3.grid(True, linestyle='--', alpha=0.3)
    
    # Ajustar limites dos eixos para melhor visualização
    E_min, E_max = np.min(E/1e9), np.max(E/1e9)
    I_min, I_max = np.min(I*1e8), np.max(I*1e8)
    
    ax3.set_ylim(0.9 * E_min, 1.1 * E_max)
    ax3_twin.set_ylim(0.9 * I_min, 1.1 * I_max)
    
    # Adicionar legendas com cores corretas e mais visíveis
    ax3.legend(loc='upper left', fontsize=11, framealpha=0.9, edgecolor='r')
    ax3_twin.legend(loc='upper right', fontsize=11, framealpha=0.9, edgecolor='b')

    # 4. Momentos fletores
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(x, M/1000, 'o-', color='#10B981', markersize=4, linewidth=2)
    ax4.set_xlabel('x [m]')
    ax4.set_ylabel('Momento Fletor [kN·m]')
    ax4.set_title('Momento Fletor')
    ax4.grid(True, linestyle='--', alpha=0.5)

    # 5. Esforços cortantes
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(x, V/1000, 'o-', color='#F59E0B', markersize=4, linewidth=2)
    ax5.set_xlabel('x [m]')
    ax5.set_ylabel('Esforço Cortante [kN]')
    ax5.set_title('Esforço Cortante')
    ax5.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Converter para GIF
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img = imageio.imread(buf)
    gif_buffer = io.BytesIO()
    imageio.mimsave(gif_buffer, [img], format='GIF', duration=1.0)
    gif_buffer.seek(0)
    gif_base64 = base64.b64encode(gif_buffer.read()).decode('utf-8')

    container = document.getElementById("viga1d-output")
    container.innerHTML = f"<img src='data:image/gif;base64,{gif_base64}' class='rounded shadow w-full'/>"

    try:
        sim_loading.close()
    except Exception:
        pass


# ==============================
# 5. Função para casos pré-definidos
# ==============================
@when("click", "#loadCase")
async def load_preset_case(event=None):
    case = document.getElementById("preset_cases").value
    
    if case == "concrete_beam":
        document.getElementById("L").value = "6.0"
        document.getElementById("E").value = "30.0"
        document.getElementById("I").value = "20000.0"
        document.getElementById("q").value = "15.0"
        document.getElementById("bc_type").value = "simply_supported"
        document.getElementById("load_type").value = "uniform"
        document.getElementById("prop_type").value = "constant"
        
    elif case == "steel_beam":
        document.getElementById("L").value = "8.0"
        document.getElementById("E").value = "200.0"
        document.getElementById("I").value = "50000.0"
        document.getElementById("q").value = "25.0"
        document.getElementById("bc_type").value = "simply_supported"
        document.getElementById("load_type").value = "uniform"
        document.getElementById("prop_type").value = "constant"
        
    elif case == "cantilever_roof":
        document.getElementById("L").value = "4.0"
        document.getElementById("E").value = "200.0"
        document.getElementById("I").value = "15000.0"
        document.getElementById("q").value = "8.0"
        document.getElementById("bc_type").value = "cantilever"
        document.getElementById("load_type").value = "triangular"
        document.getElementById("prop_type").value = "linear"
        
    elif case == "bridge_beam":
        document.getElementById("L").value = "20.0"
        document.getElementById("E").value = "200.0"
        document.getElementById("I").value = "100000.0"
        document.getElementById("q").value = "50.0"
        document.getElementById("bc_type").value = "simply_supported"
        document.getElementById("load_type").value = "point"
        document.getElementById("prop_type").value = "constant"
