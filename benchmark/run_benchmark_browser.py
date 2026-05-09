"""
Benchmark no navegador (PyScript / Pyodide / WebAssembly).

Mesmo conjunto de varreduras do runner local: cond_1d (nel in {100, 1000, 10000})
e transcalor_2d (nx=ny in {20, 40, 80}, n_steps=50). Mediana e MAD de 5 runs
após 1 warm-up. Resultado é exposto na página e oferecido como download de CSV.

Carregado via ``run_benchmark_browser.html``. Os arquivos ``bench_lib.py``,
``core/__init__.py``, ``core/cond_1d.py``, ``core/transcalor_2d.py`` são
fornecidos pelo bloco ``[[fetch]]`` em ``config.toml``.
"""

import io
import platform
import sys
import time

from js import Blob, URL, document, navigator
from pyscript import when

from bench_lib import run_repeated, write_csv
from core.cond_1d import solve_cond_1d
from core.transcalor_2d import solve_transcalor_2d


COND_1D_NEL = [100, 1000, 10000]
TRANSCALOR_NX = [20, 40, 80]
TRANSCALOR_N_STEPS = 50


def env_html():
    return (
        "<ul class='text-sm leading-relaxed'>"
        f"<li><b>Python:</b> {sys.version.split()[0]}</li>"
        f"<li><b>Plataforma (Pyodide):</b> {platform.platform()}</li>"
        f"<li><b>navigator.userAgent:</b> <code class='text-xs'>{navigator.userAgent}</code></li>"
        "</ul>"
    )


def append_log(msg):
    log_el = document.getElementById("bench-log")
    log_el.innerHTML = log_el.innerHTML + f"<div>{msg}</div>"


def _bench_cond_1d():
    rows = []
    for nel in COND_1D_NEL:
        append_log(f"[cond_1d] nel={nel} ...")
        agg = run_repeated(
            lambda nel=nel: solve_cond_1d(L=1.0, nel=nel, T0=0.0, TL=0.0, k=1.0, Q=1.0),
            n_runs=5, warmup=1,
        )
        agg.update({"case": "cond_1d", "param": f"nel={nel}", "size_n": nel + 1})
        rows.append(agg)
        append_log(
            f"  median t_total = {agg['t_total_median']*1000:.3f} ms "
            f"(MAD {agg['t_total_mad']*1000:.3f} ms)"
        )
    return rows


def _bench_transcalor_2d():
    rows = []
    for nx in TRANSCALOR_NX:
        append_log(f"[transcalor_2d] nx=ny={nx}, n_steps={TRANSCALOR_N_STEPS} ...")
        n_runs = 3 if nx >= 80 else 5
        agg = run_repeated(
            lambda nx=nx: solve_transcalor_2d(
                nx=nx, ny=nx, dt=0.01, n_steps=TRANSCALOR_N_STEPS,
            ),
            n_runs=n_runs, warmup=1,
        )
        agg.update({
            "case": "transcalor_2d",
            "param": f"nx=ny={nx}, n_steps={TRANSCALOR_N_STEPS}",
            "size_n": nx * nx,
        })
        rows.append(agg)
        append_log(
            f"  median t_total = {agg['t_total_median']:.3f} s "
            f"(MAD {agg['t_total_mad']:.3f} s)"
        )
    return rows


def offer_download(csv_text, filename):
    blob = Blob.new([csv_text], {"type": "text/csv"})
    url = URL.createObjectURL(blob)
    a = document.createElement("a")
    a.href = url
    a.download = filename
    a.textContent = f"Baixar {filename}"
    a.className = "inline-block mt-3 px-4 py-2 bg-blue-600 text-white rounded shadow hover:bg-blue-700"
    container = document.getElementById("bench-download")
    container.innerHTML = ""
    container.appendChild(a)


@when("click", "#bench-run")
def on_run(event=None):
    document.getElementById("bench-log").innerHTML = ""
    document.getElementById("bench-download").innerHTML = ""
    document.getElementById("bench-run").disabled = True
    document.getElementById("bench-run").textContent = "Rodando..."

    document.getElementById("bench-env").innerHTML = env_html()

    t0 = time.time()
    rows = []
    rows.extend(_bench_cond_1d())
    rows.extend(_bench_transcalor_2d())
    elapsed = time.time() - t0
    append_log(f"<b>Wall-clock total:</b> {elapsed:.1f} s")

    fieldnames = [
        "case", "param", "size_n", "_n_runs",
        "n_nodes", "n_elements", "nn",
        "t_total_median", "t_total_mad", "t_total_min", "t_total_max",
        "t_assembly_median", "t_assembly_mad",
        "t_solve_median", "t_solve_mad",
        "t_loop_median", "t_loop_mad",
        "t_mesh_median", "t_mesh_mad",
    ]
    buf = io.StringIO()

    # write_csv escreve em arquivo; faço aqui em memória usando csv.DictWriter direto
    import csv as _csv
    seen_keys = []
    for r in rows:
        for k in r.keys():
            if k not in seen_keys:
                seen_keys.append(k)
    use_fields = [f for f in fieldnames if f in seen_keys]
    w = _csv.DictWriter(buf, fieldnames=use_fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in use_fields})

    offer_download(buf.getvalue(), "results_browser.csv")

    document.getElementById("bench-run").disabled = False
    document.getElementById("bench-run").textContent = "Rodar benchmark novamente"


# Confirma carregamento bem-sucedido
document.getElementById("bench-status").textContent = "Pyodide pronto. Clique em 'Rodar benchmark'."
