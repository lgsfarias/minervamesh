"""
Utilidades de benchmarking compartilhadas entre o runner local (CPython) e o
runner no navegador (PyScript). Define um *context manager* simples para
medir fases, função para rodadas repetidas com estatísticas robustas
(mediana e MAD), e escrita de CSV em formato comum.
"""

import csv
import statistics
import time


class Phase:
    """Mede o tempo de uma fase do código.

    Uso:
        with Phase('solve') as p:
            ...
        print(p.dt)
    """

    def __init__(self, name=""):
        self.name = name
        self.dt = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dt = time.perf_counter() - self._t0
        return False


def run_repeated(fn, n_runs=5, warmup=1):
    """Executa ``fn`` ``warmup`` vezes (descartadas) e depois ``n_runs`` vezes,
    coletando o dict de tempos retornado.

    ``fn`` deve retornar um dict com chaves do tipo ``t_assembly``,
    ``t_solve``, ``t_total`` etc. — qualquer chave começando com ``t_`` é
    tratada como métrica de tempo. As demais chaves do último run são
    preservadas no resultado (e.g., ``n_nodes``, ``n_elements``).

    Retorna um dict ``{"<chave>_median": ..., "<chave>_mad": ..., ...,
    "_n_runs": n_runs}`` agregando as métricas de tempo, e propaga as
    chaves não-temporais do último run.
    """
    for _ in range(warmup):
        fn()

    samples = [fn() for _ in range(n_runs)]
    out = {"_n_runs": n_runs}
    last = samples[-1]

    time_keys = [k for k in last.keys() if isinstance(last[k], (int, float)) and k.startswith("t_")]
    for k in time_keys:
        vals = [s[k] for s in samples]
        med = statistics.median(vals)
        mad = statistics.median([abs(v - med) for v in vals])
        out[f"{k}_median"] = med
        out[f"{k}_mad"] = mad
        out[f"{k}_min"] = min(vals)
        out[f"{k}_max"] = max(vals)

    for k, v in last.items():
        if k.startswith("t_"):
            continue
        if isinstance(v, (int, float, str)):
            out[k] = v

    return out


def write_csv(rows, path, fieldnames=None):
    """Escreve uma lista de dicts em CSV. Se ``fieldnames`` não for dado,
    usa a união ordenada das chaves de todos os dicts."""
    if not rows:
        return
    if fieldnames is None:
        seen = []
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.append(k)
        fieldnames = seen
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
