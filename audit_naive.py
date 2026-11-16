#!/usr/bin/env python3
"""Auditoria: código "bruto" (tsp/naive.py) x código refatorado, com profiling.

Gera results/audit.csv com três comparações:
  1. bnb      — B&B bruto (best-first + cópias + limite em Python puro) x
                variantes refatoradas (mesmo limite 'twomin' em best-first e
                DFS; 1-árvore e Held-Karp em DFS). Todas recebem o mesmo
                warm-start (vizinho mais próximo) e o mesmo teto de nós
                abertos; os best-first também rodam com o GC desligado.
  2. approx   — TATT e Christofides via NetworkX completo x versões NumPy.
  3. children_mst — MST de cada filho: m Prims sequenciais x versão em lote x
                atalho das folhas (MST única + Prim em lote só nos internos).

Tempo: execução sem tracemalloc. Memória: execução separada com tracemalloc
(pico de alocação durante a chamada).

    python audit_naive.py                     # padrão (alguns minutos)
    python audit_naive.py --bnb-time 30 --bnb-instances burma14 gr17
"""
from __future__ import annotations

import argparse
import gc
import statistics
import tracemalloc
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from tsp.approx import christofides, twice_around_the_tree
from tsp.bnb import BranchAndBound
from tsp.heuristics import nearest_neighbor_tour
from tsp.mst import mst_without_each_vertex
from tsp.naive import naive_bnb, naive_christofides, naive_tatt, sequential_children_mst
from tsp.tsplib import find_instance, load_instance, tour_cost


def measure(fn, reps: int = 1):
    times, out = [], None
    for _ in range(reps):
        gc.collect()
        t0 = perf_counter()
        out = fn()
        times.append(perf_counter() - t0)
    gc.collect()
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, statistics.median(times), peak


def random_euclidean(n: int, seed: int = 0) -> np.ndarray:
    P = np.random.default_rng(seed).uniform(0, 10_000, (n, 2))
    return np.floor(np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1)) + 0.5)


def audit_bnb(names, time_limit, rows):
    for name in names:
        inst = load_instance(find_instance(name))
        D, ub_tour = inst.D, nearest_neighbor_tour(inst.D)
        variants = {
            "naive_bestfirst_twomin": lambda: naive_bnb(D, time_limit),
            "bestfirst_twomin": lambda: BranchAndBound(D, "twomin", "best", time_limit=time_limit).solve(ub_tour),
            "dfs_twomin": lambda: BranchAndBound(D, "twomin", "dfs", time_limit=time_limit).solve(ub_tour),
            "dfs_onetree": lambda: BranchAndBound(D, "onetree", "dfs", time_limit=time_limit).solve(ub_tour),
            "dfs_hk": lambda: BranchAndBound(D, "hk", "dfs", time_limit=time_limit).solve(ub_tour),
        }
        for variant, fn in variants.items():
            res, t, mem = measure(fn)
            rows.append({"experiment": "bnb", "case": name, "n": inst.n, "variant": variant,
                         "status": res["status"], "cost": res["cost"],
                         "nodes_expanded": res["nodes_expanded"],
                         "max_open": res.get("max_open"), "time": t, "mem_peak": mem})
            print(f"bnb {name:10s} {variant:24s} {res['status']:8s} "
                  f"exp={res['nodes_expanded']:>9} t={t:8.2f}s mem={mem / 2**20:8.1f} MiB")


def audit_approx(cases, rows):
    for label, D in cases:
        n = D.shape[0]
        variants = {
            "naive_tatt": lambda: naive_tatt(D),
            "tatt": lambda: twice_around_the_tree(D),
            "naive_christofides": lambda: naive_christofides(D),
            "christofides": lambda: christofides(D),
        }
        for variant, fn in variants.items():
            tour, t, mem = measure(fn, reps=3)
            rows.append({"experiment": "approx", "case": label, "n": n, "variant": variant,
                         "cost": tour_cost(D, tour), "time": t, "mem_peak": mem})
            print(f"approx {label:12s} {variant:20s} t={t:8.4f}s mem={mem / 2**20:8.2f} MiB")


def audit_children_mst(sizes, rows):
    rng = np.random.default_rng(0)
    for m in sizes:
        P = rng.uniform(0, 1, (m, 2))
        S = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
        np.fill_diagonal(S, np.inf)
        for variant, fn in (("sequential", lambda: sequential_children_mst(S)),
                            ("batched", lambda: mst_without_each_vertex(S))):
            _, t, mem = measure(fn, reps=7)
            rows.append({"experiment": "children_mst", "case": f"m={m}", "n": m,
                         "variant": variant, "time": t, "mem_peak": mem})
            print(f"children_mst m={m:3d} {variant:10s} t={t * 1e6:9.1f}us")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bnb-instances", nargs="+", default=["burma14", "ulysses16", "gr17"])
    ap.add_argument("--bnb-time", type=float, default=30.0)
    ap.add_argument("--approx-instances", nargs="+", default=["gr24", "att48", "st70"])
    ap.add_argument("--random-sizes", nargs="+", type=int, default=[200, 500])
    ap.add_argument("--out", default="results/audit.csv")
    args = ap.parse_args()

    rows: list[dict] = []
    audit_children_mst([5, 10, 20, 40, 70], rows)
    cases = [(nm, load_instance(find_instance(nm)).D) for nm in args.approx_instances]
    cases += [(f"random{n}", random_euclidean(n)) for n in args.random_sizes]
    audit_approx(cases, rows)
    audit_bnb(args.bnb_instances, args.bnb_time, rows)
    Path(args.out).parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
