#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulação de Condução de Calor em Barra Unidimensional
-----------------------------------------------------
Este script implementa uma solução numérica e analítica para o problema
de condução de calor em uma barra unidimensional com geração de calor constante.
Utiliza o método das diferenças finitas para a solução numérica.

@author: lgsfarias
"""

import numpy as np
import matplotlib.pyplot as plt
from js import document
from pyscript import when
import asyncio

from mmplot import plot_begin, plot_fig, plot_show, plot_append_html  # gráfico interativo (assets/mmplot.py)

from mmcommon import i18n_t, progress  # textos traduzidos e indicador de progresso (assets/mmcommon.py)


# ==============================
# 1. Solução Analítica
# ==============================
def analytical_solution(x, Q, alpha, L):
    """
    Calcula a solução analítica da equação de condução de calor.
    
    Parâmetros:
    -----------
    x : array
        Pontos de discretização
    Q : float
        Geração de calor constante
    alpha : float
        Coeficiente de difusividade térmica
    L : float
        Comprimento da barra
    
    Retorna:
    --------
    array
        Temperatura em cada ponto x
    """
    C1 = 1 / L + (Q / (2 * alpha)) * L
    return - (Q / (2 * alpha)) * x**2 + C1 * x

# ==============================
# 2. Função Principal de Simulação
# ==============================
async def runSimulation(event=None):
    """
    Executa a simulação de condução de calor e plota os resultados.
    """
    progress("start", "spinner")
    await asyncio.sleep(0.05)

    # Parâmetros do problema
    L = 1.0  # Comprimento da barra
    nx = int(document.getElementById("nx").value) - 2  # Número de pontos internos
    alpha = float(document.getElementById("alpha").value)  # Coeficiente de difusividade
    Q = float(document.getElementById("Q").value)  # Geração de calor
    dx = L / (nx + 1)  # Passo da discretização

    # ==============================
    # 3. Montagem do Sistema Linear
    # ==============================
    # Inicialização da matriz de rigidez e vetor de fontes
    K = np.zeros((nx, nx))
    F = np.zeros(nx)

    # Preenchimento da matriz de rigidez usando diferenças finitas
    for i in range(nx):
        K[i, i] = 2
        if i > 0:
            K[i, i - 1] = -1
        if i < nx - 1:
            K[i, i + 1] = -1

    # Montagem do vetor de fontes
    F.fill(Q / alpha * dx**2)
    F[-1] -= -1  # Aplicação da condição de contorno em x = L

    # ==============================
    # 4. Solução do Sistema
    # ==============================
    T_internal = np.linalg.solve(K, F)
    T = np.concatenate(([0], T_internal, [1]))  # Adição das condições de contorno

    # ==============================
    # 5. Cálculo da Solução Analítica
    # ==============================
    x_analytical = np.linspace(0, L, 100)
    T_analytical = analytical_solution(x_analytical, Q, alpha, L)

    # ==============================
    # 6. Visualização dos Resultados
    # ==============================
    plot_begin()
    plt.figure(**plot_fig(10, 6))
    x = np.linspace(0, L, nx + 2)
    plt.plot(x, T, 'o-', label=i18n_t("py.legend.numeric"), color='#3B82F6', linewidth=2, markersize=8)
    plt.plot(x_analytical, T_analytical, '--', label=i18n_t("py.legend.analytical"), color='#EF4444', linewidth=2)
    
    # Adição dos parâmetros na legenda
    param_label = i18n_t("py.legend.params").format(nx=nx, alpha=alpha, Q=Q)
    plt.plot([], [], ' ', label=param_label)
    
    # Configuração do gráfico
    plt.title(i18n_t("py.plot.title"), fontsize=14, pad=15)
    plt.xlabel(i18n_t("py.plot.xlabel"), fontsize=12)
    plt.ylabel(i18n_t("py.plot.ylabel"), fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plot_show(i18n_t("title"))
    progress("done")


@when("click", "#run-btn")
async def runSimulation_guarded(event=None):
    """Mostra a exceção na área do resultado e reabilita o botão Executar."""
    try:
        await runSimulation(event)
    except Exception as e:
        progress("fail")
        document.getElementById("plot-output").innerHTML = i18n_t("py.error").format(e=e)
        print(e)
    finally:
        progress("done")
