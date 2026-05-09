"""
Cópia adaptada do solver MEF de Transferência de Calor 2D do MinervaMesh,
para uso nos benchmarks. O algoritmo numérico é idêntico ao de
``simulations/mef/transcalor/simulation.py`` --- apenas removidos os imports
JS/DOM e a geração de frames (Matplotlib, imageio). A malha e os parâmetros
entram como argumentos de função em vez de globais de módulo.
"""

import time
import numpy as np
import matplotlib.tri as tri
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def element_matrices(coords, k, rho, cv):
    x0, x1, x2 = coords
    area = 0.5 * abs(
        (x1[0] - x0[0]) * (x2[1] - x0[1]) - (x2[0] - x0[0]) * (x1[1] - x0[1])
    )
    b = np.array([x1[1] - x2[1], x2[1] - x0[1], x0[1] - x1[1]])
    c = np.array([x2[0] - x1[0], x0[0] - x2[0], x1[0] - x0[0]])
    ke = (k / (4 * area)) * (np.outer(b, b) + np.outer(c, c))
    me = (rho * cv * area / 12.0) * (np.ones((3, 3)) + np.eye(3))
    return ke, me


def solve_transcalor_2d(
    nx=40, ny=40, dt=0.01, n_steps=50,
    Tbottom=0.0, Ttop=100.0, k=1.0, rho=1.0, cv=1.0,
):
    """
    Resolve condução de calor 2D transiente em placa unitária pelo MEF
    (triângulos lineares + Crank-Nicolson). Mesma lógica que o módulo
    ``transcalor`` da plataforma, sem geração de frames.

    Retorna um dict com:
        u           — temperaturas nodais finais
        X, Y        — coordenadas dos nós
        n_nodes     — número de nós
        n_elements  — número de elementos
        t_mesh      — tempo de geração de malha
        t_assembly  — tempo de montagem (M, K) + aplicação de CCs em A
        t_loop      — tempo total do loop temporal (n_steps × spsolve)
        t_total     — soma de assembly + loop (núcleo numérico)
    """
    # ---- Malha ----
    t0 = time.perf_counter()
    x_lin = np.linspace(0.0, 1.0, nx)
    y_lin = np.linspace(0.0, 1.0, ny)
    Xg, Yg = np.meshgrid(x_lin, y_lin)
    X = Xg.flatten()
    Y = Yg.flatten()
    points = np.column_stack((X, Y))
    triang = tri.Triangulation(X, Y)
    elements = triang.triangles
    n_nodes = len(points)
    t_mesh = time.perf_counter() - t0

    # ---- Montagem ----
    t0 = time.perf_counter()
    M = lil_matrix((n_nodes, n_nodes))
    K = lil_matrix((n_nodes, n_nodes))
    for el in elements:
        coords = points[el]
        ke, me = element_matrices(coords, k, rho, cv)
        for i in range(3):
            for j in range(3):
                K[el[i], el[j]] += ke[i, j]
                M[el[i], el[j]] += me[i, j]
    M = M.tocsr()
    K = K.tocsr()

    u = np.zeros(n_nodes)
    for i in range(n_nodes):
        xi, yi = X[i], Y[i]
        if np.isclose(xi, 0) or np.isclose(xi, 1):
            u[i] = Tbottom + (Ttop - Tbottom) * yi
        elif np.isclose(yi, 0):
            u[i] = Tbottom
        elif np.isclose(yi, 1):
            u[i] = Ttop

    u_init = u.copy()
    boundary_nodes = [
        i for i in range(n_nodes)
        if np.isclose(X[i], 0) or np.isclose(X[i], 1)
        or np.isclose(Y[i], 0) or np.isclose(Y[i], 1)
    ]

    A = M + dt / 2 * K
    B = M - dt / 2 * K
    A = A.tolil()
    for i in boundary_nodes:
        A[i, :] = 0
        A[i, i] = 1
    A = A.tocsr()
    t_assembly = time.perf_counter() - t0

    # ---- Loop temporal ----
    t0 = time.perf_counter()
    for _ in range(n_steps):
        b = B @ u
        b[boundary_nodes] = u_init[boundary_nodes]
        u = spsolve(A, b)
    t_loop = time.perf_counter() - t0

    return {
        "u": u,
        "X": X,
        "Y": Y,
        "n_nodes": n_nodes,
        "n_elements": len(elements),
        "t_mesh": t_mesh,
        "t_assembly": t_assembly,
        "t_loop": t_loop,
        "t_total": t_assembly + t_loop,
    }
