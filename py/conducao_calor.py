import numpy as np
import matplotlib.pyplot as plt
from js import document
from pyodide.ffi import create_proxy

def analytical_solution(x, Q, alpha, L):
    C1 = 1 + (Q / (2 * alpha)) * L
    return - (Q / (2 * alpha)) * x**2 + C1 * x

def runSimulation(event):
    L = 1.0
    nx = int(document.getElementById("nx").value)
    alpha = float(document.getElementById("alpha").value)
    Q = float(document.getElementById("Q").value)
    dx = L / (nx + 1)

    K = np.zeros((nx, nx))
    F = np.zeros(nx)

    for i in range(nx):
        K[i, i] = 2
        if i > 0:
            K[i, i - 1] = -1
        if i < nx - 1:
            K[i, i + 1] = -1

    F.fill(Q / alpha * dx**2)
    F[-1] -= -1

    T_internal = np.linalg.solve(K, F)
    T = np.concatenate(([0], T_internal, [1]))

    x_analytical = np.linspace(0, L, 100)
    T_analytical = analytical_solution(x_analytical, Q, alpha, L)

    plt.figure(figsize=(10, 6))
    x = np.linspace(0, L, nx + 2)
    plt.plot(x, T, 'o-', label="Solução Numérica", color='#3B82F6', linewidth=2, markersize=8)
    plt.plot(x_analytical, T_analytical, '--', label="Solução Analítica", color='#EF4444', linewidth=2)
    # Adiciona os parâmetros na legenda
    param_label = f"Parâmetros:\nNº Nós (nx): {nx} | Alpha: {alpha} | Q: {Q}"
    plt.plot([], [], ' ', label=param_label)
    plt.title("Comparação: Solução Numérica vs Analítica", fontsize=14, pad=15)
    plt.xlabel("Posição (x)", fontsize=12)
    plt.ylabel("Temperatura (T)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.show()

# Proxy para o evento de clique
run_btn = document.getElementById("runBtn")
run_btn.addEventListener("click", create_proxy(runSimulation))

