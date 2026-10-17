#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problema de Autovalor 1D - Vibração em Corpo Sólido (MEF)
Problema 5.2.6: Solução de problema de autovalor 1D - vibração em um corpo sólido

Este código resolve o problema de vibração livre de uma barra 1D usando MEF,
calculando frequências naturais e modos de vibração através da solução do problema de autovalor.
"""

import numpy as np
import matplotlib.pyplot as plt
from js import document
from pyscript import when
import asyncio
from scipy.linalg import eigh

from mmplot import plot_begin, plot_fig, plot_show, plot_append_html  # gráfico interativo (assets/mmplot.py)

# --- Idioma da interface (v1.1.0) ---------------------------------------------
# Rótulos de gráfico e mensagens vêm de i18n.<lang>.json via window.i18n
# (assets/app.js). Sem o app.js, a simulação roda mostrando as chaves no lugar dos textos.
import js
_i18n = getattr(js, "i18n", None)
def i18n_t(key):
    return str(_i18n.t(key)) if _i18n else key
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
# 1. Funções auxiliares para propriedades variáveis
# ==============================
def get_variable_properties(x, prop_type, base_value, L):
    """Calcula propriedades variáveis ao longo da barra"""
    if prop_type == "constant":
        return np.full_like(x, base_value)
    elif prop_type == "linear":
        # Propriedade varia linearmente de base_value a 0.5*base_value
        return base_value * (1 - 0.5 * x / L)
    elif prop_type == "parabolic":
        # Propriedade varia parabolicamente (máximo no centro)
        return base_value * (1 - 0.3 * (x - L/2)**2 / (L/2)**2)
    elif prop_type == "exponential":
        # Propriedade varia exponencialmente
        return base_value * np.exp(-0.5 * x / L)
    return np.full_like(x, base_value)


# ==============================
# 2. Soluções analíticas para frequências naturais
# ==============================
def analytical_frequencies_bars(L, E, rho, n_modes=5):
    """Frequências naturais analíticas para barra livre-livre"""
    c = np.sqrt(E / rho)  # Velocidade de onda
    frequencies = []
    for n in range(1, n_modes + 1):
        # Para barra livre-livre: fn = (n-1) * c / (2*L) para n > 1
        # Primeiro modo é movimento de corpo rígido (f = 0)
        if n == 1:
            frequencies.append(0.0)
        else:
            frequencies.append((n-1) * c / (2*L))
    return np.array(frequencies)


def analytical_frequencies_fixed_fixed(L, E, rho, n_modes=5):
    """Frequências naturais analíticas para barra engastada-engastada"""
    c = np.sqrt(E / rho)  # Velocidade de onda
    frequencies = []
    for n in range(1, n_modes + 1):
        frequencies.append(n * c / (2*L))
    return np.array(frequencies)


def analytical_frequencies_fixed_free(L, E, rho, n_modes=5):
    """Frequências naturais analíticas para barra engastada-livre"""
    c = np.sqrt(E / rho)  # Velocidade de onda
    frequencies = []
    for n in range(1, n_modes + 1):
        frequencies.append((2*n-1) * c / (4*L))
    return np.array(frequencies)


# ==============================
# 3. Executar simulação MEF para vibração 1D
# ==============================
@when("click", "#run-btn")
async def run_vibracao_1d(event=None):
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
    rho_base = float(document.getElementById("rho").value)  # kg/m³
    A_base = float(document.getElementById("A").value) * 1e-4  # Converter para m²
    bc_type = document.getElementById("bc_type").value
    prop_type = document.getElementById("prop_type").value
    n_modes = 5  # Fixo em 5 modos
    show_analytical = document.getElementById("show_analytical").checked


    nn = nel + 1
    x = np.linspace(0.0, L, nn)
    h = L / nel

    # Propriedades variáveis
    E = get_variable_properties(x, prop_type, E_base, L)
    rho = get_variable_properties(x, prop_type, rho_base, L)
    A = get_variable_properties(x, prop_type, A_base, L)

    # Para elementos de barra, cada nó tem 1 DOF: deslocamento axial (u)
    ndof = nn
    K = np.zeros((ndof, ndof))  # Matriz de rigidez
    M = np.zeros((ndof, ndof))  # Matriz de massa

    # Montagem das matrizes globais
    for e in range(nel):
        # Propriedades do elemento (média dos nós)
        E_elem = (E[e] + E[e+1]) / 2
        rho_elem = (rho[e] + rho[e+1]) / 2
        A_elem = (A[e] + A[e+1]) / 2

        # Matriz de rigidez do elemento de barra
        ke = (E_elem * A_elem / h) * np.array([
            [1, -1],
            [-1, 1]
        ])

        # Matriz de massa do elemento de barra (massa concentrada)
        me = (rho_elem * A_elem * h / 2) * np.array([
            [1, 0],
            [0, 1]
        ])

        # DOFs do elemento
        dofs = [e, e+1]

        # Montar matriz de rigidez
        for i in range(2):
            for j in range(2):
                K[dofs[i], dofs[j]] += ke[i, j]

        # Montar matriz de massa
        for i in range(2):
            for j in range(2):
                M[dofs[i], dofs[j]] += me[i, j]

    # Aplicar condições de contorno
    if bc_type == "free_free":
        # Barra livre-livre: sem restrições
        pass

    elif bc_type == "fixed_fixed":
        # Barra engastada-engastada: u(0) = 0, u(L) = 0
        # Aplicar penalidade nas diagonais principais
        penalty = 1e12
        K[0, 0] += penalty
        K[nn-1, nn-1] += penalty

    elif bc_type == "fixed_free":
        # Barra engastada-livre: u(0) = 0
        penalty = 1e12
        K[0, 0] += penalty

    elif bc_type == "fixed_supported":
        # Barra engastada-apoiada: u(0) = 0, u(L) = 0
        penalty = 1e12
        K[0, 0] += penalty
        K[nn-1, nn-1] += penalty

    # Resolver problema de autovalor: K * phi = lambda * M * phi
    # Usar scipy.linalg.eigh para matrizes simétricas
    eigenvalues, eigenvectors = eigh(K, M)

    # Calcular frequências naturais (Hz)
    frequencies = np.sqrt(eigenvalues) / (2 * np.pi)

    # Ordenar por frequência crescente
    idx = np.argsort(frequencies)
    frequencies = frequencies[idx]
    eigenvectors = eigenvectors[:, idx]

    # Pegar apenas os primeiros n_modes
    frequencies = frequencies[:n_modes]
    eigenvectors = eigenvectors[:, :n_modes]

    # Normalizar modos de vibração
    for i in range(n_modes):
        eigenvectors[:, i] = eigenvectors[:, i] / np.max(np.abs(eigenvectors[:, i]))

    # Solução analítica para comparação (se disponível)
    freq_analytical = np.zeros(n_modes)

    if show_analytical and prop_type == "constant":
        if bc_type == "free_free":
            freq_analytical = analytical_frequencies_bars(L, E_base, rho_base, n_modes)
        elif bc_type == "fixed_fixed":
            freq_analytical = analytical_frequencies_fixed_fixed(L, E_base, rho_base, n_modes)
        elif bc_type == "fixed_free":
            freq_analytical = analytical_frequencies_fixed_free(L, E_base, rho_base, n_modes)

    # Determinar título da condição de contorno
    bc_titles = {k: i18n_t("py.bc." + k) for k in ("free_free", "fixed_fixed", "fixed_free", "fixed_supported")}
    bc_title = bc_titles.get(bc_type, i18n_t("py.bc.unknown"))

    # Tabela de frequências naturais (em HTML, abaixo da figura; ver plot_append_html)
    show_analytical_table = show_analytical and prop_type == "constant"
    rows_html = []
    for i in range(min(5, n_modes)):
        cells = [i18n_t("py.table.mode_n").format(n=i+1), f"{frequencies[i]:.2f}"]
        if show_analytical_table and i < len(freq_analytical) and freq_analytical[i] > 0:
            error = abs(frequencies[i] - freq_analytical[i]) / freq_analytical[i] * 100
            cells += [f"{freq_analytical[i]:.2f}", f"{error:.1f}%"]
        elif show_analytical_table:
            cells += ["-", "-"]
        rows_html.append("<tr>" + "".join(f"<td class='px-3 py-1 text-center'>{c}</td>" for c in cells) + "</tr>")
    headers = [i18n_t("py.table.mode"), i18n_t("py.table.fem"), i18n_t("py.table.analytical"), i18n_t("py.table.error")] if show_analytical_table else [i18n_t("py.table.mode"), i18n_t("py.table.fem")]
    metrics_html = (
        "<div class=\"bg-blue-50 border-l-4 border-blue-400 p-3 rounded w-full\">"
        f"<h4 class=\"font-semibold text-blue-800 mb-2\">{i18n_t('py.validation')}</h4>"
        f"<p class=\"text-sm text-gray-700 mb-2\"><strong>{i18n_t('py.val.freq_label')}</strong> {i18n_t('py.val.case').format(bc=bc_title, prop=prop_type)}</p>"
        "<table class=\"text-sm text-gray-700 border-collapse\"><thead><tr>"
        + "".join(f"<th class='px-3 py-1 text-center font-semibold text-blue-800 border-b border-blue-200'>{h}</th>" for h in headers)
        + "</tr></thead><tbody>" + "".join(rows_html) + "</tbody></table></div>"
    )

    # Plot avançado com múltiplos gráficos
    plot_begin()
    fig = plt.figure(**plot_fig(14, 20))

    # Layout: 6x1 grid (5 modos + 1 propriedades)
    gs = fig.add_gridspec(6, 1, height_ratios=[2.2, 2.2, 2.2, 2.2, 2.2, 1.5], hspace=0.5)

    # 2-6. Modos de vibração (5 modos fixos)
    colors = ['#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6']
    for i in range(min(5, n_modes)):
        ax = fig.add_subplot(gs[i, :])

        # Plotar modo de vibração
        ax.plot(x, eigenvectors[:, i], 'o-', color=colors[i], markersize=6, linewidth=3, 
                label=i18n_t("py.plot.mode_legend").format(n=i+1, f=frequencies[i]))

        # Adicionar linha de referência em zero
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

        ax.set_xlabel('x [m]', fontsize=14, fontweight='bold')
        ax.set_ylabel(i18n_t("py.plot.amp_ylabel"), fontsize=14, fontweight='bold')
        ax.set_title(i18n_t("py.plot.mode_title").format(n=i+1), fontsize=16, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=12)

        # Adicionar nós da malha
        ax.scatter(x, eigenvectors[:, i], color=colors[i], s=50, zorder=5)

    # 7. Propriedades materiais
    ax_props = fig.add_subplot(gs[5, :])

    # Plotar linha vermelha primeiro (E) com estilo mais visível
    line1 = ax_props.plot(x, E/1e9, 'r-', linewidth=4, label='E [GPa]', alpha=0.9, zorder=3)

    # Criar eixo secundário para linha azul (rho) - tracejada e mais fina
    ax_props_twin = ax_props.twinx()
    line2 = ax_props_twin.plot(x, rho, 'b--', linewidth=2, label='ρ [kg/m³]', alpha=0.8, zorder=2)

    # Configurar eixos e labels
    ax_props.set_xlabel('x [m]', fontsize=14, fontweight='bold')
    ax_props.set_ylabel(i18n_t("py.plot.E_ylabel"), color='r', fontweight='bold', fontsize=14)
    ax_props_twin.set_ylabel(i18n_t("py.plot.rho_ylabel"), color='b', fontweight='bold', fontsize=14)
    ax_props.set_title(i18n_t("py.plot.props_title"), fontweight='bold', fontsize=16)

    # Configurar cores dos ticks
    ax_props.tick_params(axis='y', labelcolor='r', labelsize=12)
    ax_props_twin.tick_params(axis='y', labelcolor='b', labelsize=12)
    ax_props.tick_params(axis='x', labelsize=12)

    # Grid mais sutil
    ax_props.grid(True, linestyle='--', alpha=0.3)

    # Ajustar limites dos eixos para melhor visualização
    E_min, E_max = np.min(E/1e9), np.max(E/1e9)
    rho_min, rho_max = np.min(rho), np.max(rho)

    ax_props.set_ylim(0.9 * E_min, 1.1 * E_max)
    ax_props_twin.set_ylim(0.9 * rho_min, 1.1 * rho_max)

    # Adicionar legendas com cores corretas e mais visíveis
    ax_props.legend(loc='upper left', fontsize=12, framealpha=0.9, edgecolor='r')
    ax_props_twin.legend(loc='upper right', fontsize=12, framealpha=0.9, edgecolor='b')

    # Sem bbox_inches='tight' (figura ao vivo), reduz as margens externas da figura
    fig.subplots_adjust(top=0.98, bottom=0.03)
    plot_show(i18n_t("title"))
    plot_append_html(metrics_html)
    _export_result(x=x, frequencies=frequencies, freq_analytical=freq_analytical, eigenvectors=eigenvectors)

    try:
        sim_loading.close()
    except Exception:
        pass


# ==============================
# 4. Função para casos pré-definidos
# ==============================
@when("click", "#loadCase")
async def load_preset_case(event=None):
    case = document.getElementById("preset_cases").value

    if case == "steel_bar":
        document.getElementById("L").value = "2.0"
        document.getElementById("E").value = "200.0"
        document.getElementById("rho").value = "7850.0"
        document.getElementById("A").value = "100.0"
        document.getElementById("bc_type").value = "fixed_fixed"
        document.getElementById("prop_type").value = "constant"

    elif case == "aluminum_bar":
        document.getElementById("L").value = "1.5"
        document.getElementById("E").value = "70.0"
        document.getElementById("rho").value = "2700.0"
        document.getElementById("A").value = "50.0"
        document.getElementById("bc_type").value = "fixed_free"
        document.getElementById("prop_type").value = "constant"

    elif case == "concrete_column":
        document.getElementById("L").value = "3.0"
        document.getElementById("E").value = "30.0"
        document.getElementById("rho").value = "2400.0"
        document.getElementById("A").value = "400.0"
        document.getElementById("bc_type").value = "fixed_fixed"
        document.getElementById("prop_type").value = "linear"

    elif case == "composite_bar":
        document.getElementById("L").value = "1.0"
        document.getElementById("E").value = "100.0"
        document.getElementById("rho").value = "2000.0"
        document.getElementById("A").value = "25.0"
        document.getElementById("bc_type").value = "free_free"
        document.getElementById("prop_type").value = "parabolic"
