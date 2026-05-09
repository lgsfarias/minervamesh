"""
Benchmark local (CPython nativo) dos solvers do MinervaMesh.

Roda os mesmos solvers que o navegador (importados de ``benchmark.core``)
em uma varredura de tamanhos, mede tempos com mediana e MAD de 5 rodadas
após 1 warm-up, e salva ``benchmark/results/results_local.csv``.

Uso:
    python3 benchmark/run_benchmark_local.py
"""

import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from bench_lib import run_repeated, write_csv  # noqa: E402
from core.cond_1d import solve_cond_1d  # noqa: E402
from core.transcalor_2d import solve_transcalor_2d  # noqa: E402


COND_1D_NEL = [100, 1000, 10000]
TRANSCALOR_NX = [20, 40, 80]
TRANSCALOR_N_STEPS = 50


def report_environment():
    print("=" * 60)
    print("Ambiente de execução")
    print("=" * 60)
    print(f"  Plataforma : {platform.platform()}")
    print(f"  Processador: {platform.processor()}")
    print(f"  Python     : {sys.version.split()[0]}")
    print(f"  NumPy      : {np.__version__}")
    print("  NumPy backend (np.show_config()):")
    np.show_config()
    print("=" * 60)


def bench_cond_1d():
    rows = []
    for nel in COND_1D_NEL:
        print(f"[cond_1d] nel={nel} ...", flush=True)
        agg = run_repeated(
            lambda: solve_cond_1d(L=1.0, nel=nel, T0=0.0, TL=0.0, k=1.0, Q=1.0),
            n_runs=5, warmup=1,
        )
        agg.update({"case": "cond_1d", "param": f"nel={nel}", "size_n": nel + 1})
        rows.append(agg)
        print(f"  median t_total = {agg['t_total_median']*1000:.3f} ms (MAD {agg['t_total_mad']*1000:.3f} ms)")
    return rows


def bench_transcalor_2d():
    rows = []
    for nx in TRANSCALOR_NX:
        print(f"[transcalor_2d] nx=ny={nx}, n_steps={TRANSCALOR_N_STEPS} ...", flush=True)
        # Reduzir n_runs para 3 quando nx=80 para não inflar o wall-clock
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
        print(f"  median t_total = {agg['t_total_median']:.3f} s (MAD {agg['t_total_mad']:.3f} s)")
    return rows


def main():
    report_environment()
    t_start = time.time()

    rows = []
    rows.extend(bench_cond_1d())
    rows.extend(bench_transcalor_2d())

    out_path = os.path.join(HERE, "results", "results_local.csv")
    fieldnames = [
        "case", "param", "size_n", "_n_runs",
        "n_nodes", "n_elements", "nn",
        "t_total_median", "t_total_mad", "t_total_min", "t_total_max",
        "t_assembly_median", "t_assembly_mad",
        "t_solve_median", "t_solve_mad",
        "t_loop_median", "t_loop_mad",
        "t_mesh_median", "t_mesh_mad",
    ]
    write_csv(rows, out_path, fieldnames=fieldnames)
    print(f"\nCSV salvo em {out_path}")
    print(f"Wall-clock total: {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
