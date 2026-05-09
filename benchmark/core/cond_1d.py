"""
Cópia adaptada do solver MEF de Condução Permanente 1D do MinervaMesh,
para uso nos benchmarks. O algoritmo numérico é idêntico ao de
``simulations/mef/conducao-permanente-barra1d-mef/simulation.py`` --- apenas
removidos os imports JS/DOM e o pós-processamento (Matplotlib, imageio,
escrita no DOM). Os parâmetros entram como argumentos de função.
"""

import time
import numpy as np


def analytical_solution(x, Q, k, L, T0, TL):
    C1 = (TL - T0 + (Q * L * L) / (2 * k)) / L
    C2 = T0
    return -(Q / (2 * k)) * x * x + C1 * x + C2


def solve_cond_1d(L=1.0, nel=10, T0=0.0, TL=0.0, k=1.0, Q=1.0):
    """
    Resolve a condução permanente 1D pelo MEF (elementos lineares).

    Retorna um dict com:
        x       — coordenadas nodais
        T       — temperaturas nodais
        T_ana   — solução analítica nos mesmos nós
        t_assembly — tempo de montagem da matriz global (s)
        t_solve    — tempo de np.linalg.solve (s)
        t_total    — soma dos dois acima
        nn         — número de nós (= nel + 1)
    """
    nn = nel + 1
    x = np.linspace(0.0, L, nn)
    h = L / nel

    # ---- Montagem ----
    t0 = time.perf_counter()
    K = np.zeros((nn, nn))
    b = np.zeros(nn)
    ke = (k / h) * np.array([[1.0, -1.0], [-1.0, 1.0]])
    be = (Q * h / 2.0) * np.array([1.0, 1.0])
    for e in range(nel):
        n1, n2 = e, e + 1
        K[n1:n2 + 1, n1:n2 + 1] += ke
        b[n1:n2 + 1] += be

    # Dirichlet
    K[0, :] = 0.0
    K[0, 0] = 1.0
    b[0] = T0
    K[-1, :] = 0.0
    K[-1, -1] = 1.0
    b[-1] = TL
    t_assembly = time.perf_counter() - t0

    # ---- Solve ----
    t0 = time.perf_counter()
    T = np.linalg.solve(K, b)
    t_solve = time.perf_counter() - t0

    T_ana = analytical_solution(x, Q, k, L, T0, TL)

    return {
        "x": x,
        "T": T,
        "T_ana": T_ana,
        "t_assembly": t_assembly,
        "t_solve": t_solve,
        "t_total": t_assembly + t_solve,
        "nn": nn,
    }
