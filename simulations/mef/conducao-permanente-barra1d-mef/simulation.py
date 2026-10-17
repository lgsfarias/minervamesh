#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Condução Permanente 1D (MEF) — Problema 5.2.2
Barra 1D com geração interna Q, T(0)=T0 e T(L)=TL.
Elementos lineares, solução analítica para comparação.
"""

import numpy as np
import matplotlib.pyplot as plt
from js import document
from pyscript import when
import asyncio

from mmplot import plot_begin, plot_fig, plot_show, plot_append_html  # gráfico interativo (assets/mmplot.py)

# --- Idioma da interface (v1.1.0) ---------------------------------------------
# Rótulos de gráfico e mensagens vêm de i18n.<lang>.json via window.i18n
# (assets/app.js). Sem o app.js, a simulação roda mostrando as chaves no lugar dos textos.
import js
_i18n = getattr(js, "i18n", None)
def i18n_t(key):
    return str(_i18n.t(key)) if _i18n else key
# ------------------------------------------------------------------------------

# --- Progresso (v1.1.0): window.mmProgress (assets/app.js); sem o app.js, no-op ---
_progress = getattr(js, "mmProgress", None)
def progress(method, *args):
    if _progress:
        getattr(_progress, method)(*args)
# ------------------------------------------------------------------------------

# --- Gancho de verificação (v1.1.0) -------------------------------------------
# Exporta o resultado numérico para window.__minervamesh_result, lido por
# scripts/baseline.mjs para comparar versões. Não participa de nenhum cálculo.
import js as _js
from pyodide.ffi import to_js as _to_js

def _export_result(**kw):
    def _conv(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v
    _js.window.__minervamesh_result = _to_js(
        {k: _conv(v) for k, v in kw.items()}, dict_converter=_js.Object.fromEntries
    )
# ------------------------------------------------------------------------------


# ==============================
# 1. Solução analítica
# ==============================
def analytical_solution(x, Q, k, L, T0, TL):
    # Solução da EDO: -k T'' = Q, T(0)=T0, T(L)=TL
    C1 = (TL - T0 + (Q * L * L) / (2 * k)) / L
    C2 = T0
    return - (Q / (2 * k)) * x * x + C1 * x + C2


# ==============================
# 2. Executar simulação MEF 1D
# ==============================
async def run_mef_1d(event=None):
    progress("start", "spinner")
    await asyncio.sleep(0.05)

    # Parâmetros
    L = float(document.getElementById("L").value)
    nel = int(document.getElementById("nel").value)
    T0 = float(document.getElementById("T0").value)
    TL = float(document.getElementById("TL").value)
    k = float(document.getElementById("k").value)
    Q = float(document.getElementById("Q").value)

    # Malha 1D
    nn = nel + 1
    x = np.linspace(0.0, L, nn)
    h = L / nel

    # Matrizes/vetores globais
    K = np.zeros((nn, nn))        # matriz de rigidez (apostila: K)
    b = np.zeros(nn)              # vetor de carregamento (apostila: b)

    # Elemento linear: ke = k/h * [[1,-1],[-1,1]]; be = Q*h/2 * [1,1]
    ke = (k / h) * np.array([[1.0, -1.0], [-1.0, 1.0]])
    be = (Q * h / 2.0) * np.array([1.0, 1.0])

    # Montagem
    for e in range(nel):
        n1, n2 = e, e + 1
        K[n1:n2+1, n1:n2+1] += ke
        b[n1:n2+1] += be

    # Contornos essenciais (Dirichlet)
    # T(0)=T0
    K[0, :] = 0.0
    K[0, 0] = 1.0
    b[0] = T0
    # T(L)=TL
    K[-1, :] = 0.0
    K[-1, -1] = 1.0
    b[-1] = TL

    # Resolver
    T = np.linalg.solve(K, b)

    # Analítico (grade densa para curva lisa)
    x_dense = np.linspace(0.0, L, max(600, min(2400, 8 * nn)))
    T_ana_dense = analytical_solution(x_dense, Q, k, L, T0, TL)

    # Analítico nos nós da malha para métricas de erro
    T_ana_mesh = analytical_solution(x, Q, k, L, T0, TL)
    erro_nodal = np.abs(T - T_ana_mesh)
    e_max = float(np.max(erro_nodal))
    T_max_mef = float(np.max(T))
    x_max_mef = float(x[int(np.argmax(T))])
    T_max_ana = float(np.max(T_ana_dense))
    x_max_ana = float(x_dense[int(np.argmax(T_ana_dense))])

    # Plot
    plot_begin()
    fig, ax = plt.subplots(**plot_fig(8, 4))
    ax.plot(x, T, 'o-', label=i18n_t("py.legend.fem"), color='#3B82F6')
    ax.plot(x_dense, T_ana_dense, '--', label=i18n_t("py.legend.analytical"), color='#EF4444')
    ax.set_xlabel('x [m]')
    ax.set_ylabel(i18n_t("py.plot.ylabel"))
    ax.set_title(i18n_t("py.plot.title"))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()

    plot_show(i18n_t("title"))

    metrics_html = f"""
    <div class="bg-blue-50 border-l-4 border-blue-400 p-3 rounded w-full">
        <h4 class="font-semibold text-blue-800 mb-1">{i18n_t("py.validation")}</h4>
        <p class="text-sm text-gray-700"><strong>{i18n_t("py.val.mesh_label")}</strong> {i18n_t("py.val.mesh").format(nel=nel, h=h)}</p>
        <p class="text-sm text-gray-700"><strong>{i18n_t("py.val.tmax_fem_label")}</strong> {i18n_t("py.val.at_x").format(T=T_max_mef, x=x_max_mef)}</p>
        <p class="text-sm text-gray-700"><strong>{i18n_t("py.val.tmax_ana_label")}</strong> {i18n_t("py.val.at_x").format(T=T_max_ana, x=x_max_ana)}</p>
        <p class="text-sm text-gray-700"><strong>{i18n_t("py.val.err_label")}</strong> {e_max:.2e} °C</p>
    </div>
    """
    plot_append_html(metrics_html)
    _export_result(x=x, T=T, T_ana_mesh=T_ana_mesh, e_max=e_max, T_max_mef=T_max_mef, x_max_mef=x_max_mef)
    progress("done")


@when("click", "#run-btn")
async def run_mef_1d_guarded(event=None):
    """Mostra a exceção na área do resultado e reabilita o botão Executar."""
    try:
        await run_mef_1d(event)
    except Exception as e:
        progress("fail")
        document.getElementById("plot-output").innerHTML = i18n_t("py.error").format(e=e)
        print(e)
    finally:
        progress("done")
