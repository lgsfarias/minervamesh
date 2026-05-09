"""
Gera ``tcc_minervamesh/figuras/benchmark_scaling.pdf`` a partir de
``results_local.csv`` e (opcionalmente) ``results_browser.csv``.

Uso:
    python3 benchmark/make_figure.py

Se ``results_browser.csv`` ainda não existir, gera a figura só com os
pontos do CPython local (linha pontilhada). Quando o CSV do navegador
estiver presente, sobrepõe os dois ambientes.
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_CSV = os.path.join(HERE, "results", "results_local.csv")
BROWSER_CSV = os.path.join(HERE, "results", "results_browser.csv")
OUT_PDF = os.path.normpath(
    os.path.join(HERE, "..", "tcc_minervamesh", "figuras", "benchmark_scaling.pdf")
)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def filter_case(rows, case):
    return [r for r in rows if r["case"] == case]


def to_float(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def main():
    if not os.path.exists(LOCAL_CSV):
        sys.exit(f"Faltando {LOCAL_CSV}. Rode primeiro run_benchmark_local.py.")

    local = read_csv(LOCAL_CSV)
    browser = read_csv(BROWSER_CSV) if os.path.exists(BROWSER_CSV) else []

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    panels = [
        ("cond_1d", "Condução 1D MEF (denso)"),
        ("transcalor_2d", "Transcalor 2D MEF (esparso, 50 passos)"),
    ]
    for ax, (case, title) in zip(axes, panels):
        for env_label, rows, marker, ls in [
            ("CPython local", filter_case(local, case), "o", "-"),
            ("Pyodide (navegador)", filter_case(browser, case), "s", "--"),
        ]:
            if not rows:
                continue
            xs = [to_float(r["size_n"]) for r in rows]
            ys = [to_float(r["t_total_median"]) for r in rows]
            ax.plot(xs, ys, marker=marker, linestyle=ls, label=env_label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Número de nós $N$")
        ax.set_ylabel("Tempo total do solver [s]")
        ax.set_title(title)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    plt.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    print(f"Figura salva em {OUT_PDF}")
    print(f"  local: {len(local)} linhas; browser: {len(browser)} linhas")


if __name__ == "__main__":
    main()
