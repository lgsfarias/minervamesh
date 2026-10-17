#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convecção-Difusão 1D Permanente (MEF) — Problema 5.2.4
Equação: -k d²T/dx² + ρcp u dT/dx = Q
Condições: T(0)=T0, T(L)=TL
Elementos lineares com estabilização SUPG.
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
def analytical_solution(x, Q, k, rho_cp, u, L, T0, TL):
    """
    Solução analítica da equação de convecção-difusão 1D permanente.
    Equação: -k d²T/dx² + ρcp u dT/dx = Q
    Condições: T(0)=T0, T(L)=TL

    Forma estável numericamente mesmo para grandes números de Péclet: a razão
    (exp(α x) - 1) / (exp(α L) - 1) é reescrita fatorando exp(α L) quando α > 0,
    evitando overflow de exp para α·L grande.
    """
    # Caso puramente difusivo: -k T'' = Q
    if abs(rho_cp * u) < 1e-14:
        C1 = (TL - T0 + (Q * L * L) / (2 * k)) / L
        return -(Q / (2 * k)) * x * x + C1 * x + T0

    alpha = rho_cp * u / k
    Tp   = Q * x / (rho_cp * u)
    Tp_L = Q * L / (rho_cp * u)

    # g(x) := (exp(α x) - 1) / (exp(α L) - 1)
    if alpha > 0:
        # fatora exp(α L) para evitar overflow
        g = np.exp(alpha * (x - L)) * (1.0 - np.exp(-alpha * x)) \
                                    / (1.0 - np.exp(-alpha * L))
    else:
        # para α ≤ 0, expm1 já é estável
        g = np.expm1(alpha * x) / np.expm1(alpha * L)

    return T0 + Tp + (TL - T0 - Tp_L) * g


# ==============================
# 2. Executar simulação MEF 1D com convecção-difusão
# ==============================
async def run_conveccao_difusao_1d(event=None):
    progress("start", "spinner")
    await asyncio.sleep(0.05)

    # Parâmetros
    L = float(document.getElementById("L").value)
    nel = int(document.getElementById("nel").value)
    T0 = float(document.getElementById("T0").value)
    TL = float(document.getElementById("TL").value)
    k = float(document.getElementById("k").value)
    rho_cp = float(document.getElementById("rho_cp").value)
    u = float(document.getElementById("u").value)
    Q = float(document.getElementById("Q").value)
    use_supg = document.getElementById("use_supg").checked

    # Malha 1D uniforme
    nn = nel + 1
    x = np.linspace(0.0, L, nn)
    h = L / nel

    # Número de Péclet local
    Pe_local = (rho_cp * u * h) / (2 * k)
    
    # Parâmetro de estabilização SUPG
    if use_supg and abs(Pe_local) > 1e-10:
        tau = h / (2 * abs(u)) * (1 / np.tanh(abs(Pe_local)) - 1 / abs(Pe_local))
    else:
        tau = 0.0

    # Matrizes/vetores globais
    K = np.zeros((nn, nn))        # matriz de rigidez
    b = np.zeros(nn)              # vetor de carregamento

    # Elemento linear com convecção-difusão
    # Matriz de difusão: ke_diff = k/h * [[1,-1],[-1,1]]
    ke_diff = (k / h) * np.array([[1.0, -1.0], [-1.0, 1.0]])
    
    # Matriz de convecção: ke_conv = ρcp*u/2 * [[-1,1],[-1,1]]
    ke_conv = (rho_cp * u / 2.0) * np.array([[-1.0, 1.0], [-1.0, 1.0]])
    
    # Vetor de fontes: be = Q*h/2 * [1,1]
    be = (Q * h / 2.0) * np.array([1.0, 1.0])
    
    # Estabilização SUPG (se habilitada)
    if use_supg and tau > 0:
        # Matriz de estabilização SUPG
        ke_supg = tau * rho_cp * u * u / h * np.array([[1.0, -1.0], [-1.0, 1.0]])
        # Vetor de estabilização SUPG
        be_supg = tau * Q * u * np.array([-1.0, 1.0])
        
        # Combinar todas as contribuições
        ke_total = ke_diff + ke_conv + ke_supg
        be_total = be + be_supg
    else:
        ke_total = ke_diff + ke_conv
        be_total = be

    # Montagem
    for e in range(nel):
        n1, n2 = e, e + 1
        K[n1:n2+1, n1:n2+1] += ke_total
        b[n1:n2+1] += be_total

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
    T_ana_dense = analytical_solution(x_dense, Q, k, rho_cp, u, L, T0, TL)
    
    # Solução analítica nos pontos da malha numérica para cálculo do erro
    T_ana_mesh = analytical_solution(x, Q, k, rho_cp, u, L, T0, TL)

    # Plot
    plot_begin()
    fig, (ax1, ax2) = plt.subplots(2, 1, **plot_fig(10, 8))
    
    # Gráfico principal
    ax1.plot(x, T, 'o-', label=i18n_t("py.legend.fem"), color='#3B82F6', markersize=6)
    ax1.plot(x_dense, T_ana_dense, '--', label=i18n_t("py.legend.analytical"), color='#EF4444', linewidth=2)
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel(i18n_t("py.plot.ylabel"))
    ax1.set_title(i18n_t("py.plot.title"))
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    
    # Gráfico de erro
    erro = np.abs(T - T_ana_mesh)
    # Evitar erro zero para escala logarítmica
    erro = np.maximum(erro, 1e-18)
    ax2.semilogy(x, erro, 'o-', color='#10B981', markersize=6)
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel(i18n_t("py.plot.err_ylabel"))
    ax2.set_title(i18n_t("py.plot.err_title"))
    ax2.grid(True, linestyle='--', alpha=0.5)
    # Adicionar texto explicativo
    ax2.text(0.5, 0.5, i18n_t("py.plot.err_text").format(err=np.max(erro)), 
             transform=ax2.transAxes, ha='center', va='center', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()

    plot_show(i18n_t("title"))

    # Atualizar informações sobre Péclet
    pe_info = document.getElementById("pe-info")
    pe_info.innerHTML = f"""
    <div class="bg-blue-50 border-l-4 border-blue-400 p-3 rounded">
        <h4 class="font-semibold text-blue-800 mb-1">{i18n_t("py.info.title")}</h4>
        <p class="text-sm text-gray-700"><strong>{i18n_t("py.info.peclet")}</strong> {Pe_local:.3f}</p>
        <p class="text-sm text-gray-700"><strong>{i18n_t("py.info.supg")}</strong> {i18n_t("py.yes") if use_supg else i18n_t("py.no")}</p>
        <p class="text-sm text-gray-700"><strong>{i18n_t("py.info.tau")}</strong> {tau:.6f}</p>
        <p class="text-sm text-gray-700"><strong>{i18n_t("py.info.u")}</strong> {u:.3f} m/s</p>
        <p class="text-sm text-gray-700"><strong>{i18n_t("py.info.h")}</strong> {h:.4f} m</p>
        <p class="text-sm text-gray-700"><strong>{i18n_t("py.info.err")}</strong> {np.max(erro):.2e} °C</p>
    </div>
    """
    _export_result(x=x, T=T, T_ana_mesh=T_ana_mesh, erro_max=float(np.max(erro)), Pe_local=Pe_local, tau=tau, use_supg=bool(use_supg))
    progress("done")


@when("click", "#run-btn")
async def run_conveccao_difusao_1d_guarded(event=None):
    """Mostra a exceção na área do resultado e reabilita o botão Executar."""
    try:
        await run_conveccao_difusao_1d(event)
    except Exception as e:
        progress("fail")
        document.getElementById("plot-output").innerHTML = i18n_t("py.error").format(e=e)
        print(e)
    finally:
        progress("done")
