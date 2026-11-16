#!/usr/bin/env python3
"""Profiling por subrotina em instâncias aleatórias de tamanho crescente.

Com a TSPLIB na faixa 14 <= N <= 76, curvas como n^2 e n^3 ficam quase
indistinguíveis. Para a regressão de complexidade empírica, este script roda
os algoritmos polinomiais em instâncias uniformes aleatórias (pontos em
[0, 10000]^2, distância EUC_2D da TSPLIB) com N bem maior, e opcionalmente o
B&B em N pequeno (crescimento exponencial).

Saída: results/scaling.csv, uma linha por (algoritmo, N, semente), com
time_<subrotina> (mediana de --reps execuções sem tracemalloc) e
mem_peak_<subrotina> (execução separada com tracemalloc), custo, peso da MST
(limite inferior: custo/MST <= 2 no TATT) e coletas do GC.

    python profile_scaling.py
    python profile_scaling.py --sizes 100 200 400 800 --seeds 0 1 2 --no-gc
    python profile_scaling.py --sizes --bnb-sizes 10 15 20 25 30 35 40 --bnb-time 300

Cada medição de tempo tem 1 execução de aquecimento descartada; as coletas
do GC são contadas só dentro da janela cronometrada.
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
from tsp.heuristics import convex_hull_insertion_tour, nearest_neighbor_tour
from tsp.mst import prim
from tsp.profiler import Profiler
from tsp.tsplib import dist_euc_2d, tour_cost


def random_instance(n: int, seed: int):
    P = np.random.default_rng(seed).uniform(0, 10_000, (n, 2))
    return P, dist_euc_2d(P)


def algorithms(P, D):
    """nome -> função(prof) que devolve um tour."""
    def sec(name, fn):
        def run(prof):
            with prof.section(name):
                return fn()
        return run
    return {
        "nn": sec("nn", lambda: nearest_neighbor_tour(D)),
        "hull": lambda prof: _hull(prof, D, P),
        "tatt": lambda prof: twice_around_the_tree(D, prof),
        "christofides": lambda prof: christofides(D, prof, matching="blossom"),
        "christofides_greedy": lambda prof: christofides(D, prof, matching="greedy"),
    }


def _hull(prof, D, P):
    with prof.section("hull"):
        return convex_hull_insertion_tour(D, P)


def gc_count() -> int:
    return sum(s["collections"] for s in gc.get_stats())


def profile_one(fn, reps: int) -> dict:
    """1 execução de aquecimento + `reps` cronometradas (mediana) + 1 com
    tracemalloc. Coletas do GC contadas só dentro da janela cronometrada."""
    times, sections, tour = [], {}, None
    gc0 = gc_count()
    for _ in range(reps):
        prof = Profiler()
        gc.collect()
        t0 = perf_counter()
        tour = fn(prof)
        times.append(perf_counter() - t0)
        for k, v in prof.times.items():
            sections.setdefault(k, []).append(v)
    out = {"time_total": statistics.median(times), "time_total_min": min(times),
           "gc_collections": (gc_count() - gc0) / reps, "tour": tour}
    out.update({f"time_{k}": statistics.median(v) for k, v in sections.items()})
    prof = Profiler(trace_memory=True)
    gc.collect()
    tracemalloc.start()
    with prof.section("total"):
        fn(prof)
    tracemalloc.stop()
    out.update({k: v for k, v in prof.as_dict().items() if k.startswith("mem_peak_")})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", nargs="*", type=int,
                    default=[25, 50, 100, 200, 400, 800, 1000, 1600, 3200, 6400])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--algos", nargs="+",
                    default=["nn", "hull", "tatt", "christofides", "christofides_greedy"])
    ap.add_argument("--blossom-max-n", type=int, default=1000,
                    help="não roda o Christofides com blossom acima deste N (O(n^3) em Python)")
    ap.add_argument("--bnb-sizes", nargs="*", type=int, default=[])
    ap.add_argument("--bnb-time", type=float, default=300.0)
    ap.add_argument("--bnb-bound", default="hk")
    ap.add_argument("--no-gc", action="store_true")
    ap.add_argument("--out", default="results/scaling.csv")
    args = ap.parse_args()
    if args.no_gc:
        gc.disable()

    rows = []
    for n in args.sizes:
        for seed in args.seeds:
            P, D = random_instance(n, seed)
            mst_w = prim(D)[1]
            algos = algorithms(P, D)
            for name in args.algos:
                if name == "christofides" and n > args.blossom_max_n:
                    continue
                res = profile_one(algos[name], args.reps)
                cost = tour_cost(D, res.pop("tour"))
                rows.append({"algo": name, "n": n, "seed": seed, "cost": cost,
                             "mst_weight": mst_w, "ratio_to_mst": cost / mst_w,
                             "gc_enabled": not args.no_gc, **res})
                print(f"{name:20s} n={n:5d} seed={seed} t={res['time_total']:9.4f}s "
                      f"mem={res.get('mem_peak_total', 0) / 2**20:7.2f} MiB", flush=True)

    for n in args.bnb_sizes:
        for seed in args.seeds:
            P, D = random_instance(n, seed)
            tour0 = nearest_neighbor_tour(D)
            res = BranchAndBound(D, bound=args.bnb_bound, time_limit=args.bnb_time).solve(tour0)
            rows.append({"algo": f"bnb_{args.bnb_bound}", "n": n, "seed": seed,
                         "cost": res["cost"], "status": res["status"],
                         "time_total": res["time_bnb"],
                         "time_root_bound": res["time_root_bound"],
                         "time_child_bounds": res["time_child_bounds"],
                         "nodes_expanded": res["nodes_expanded"],
                         "nodes_pruned": res["nodes_pruned"],
                         "max_depth": res["max_depth"],
                         "initial_ub": res["initial_ub"], "root_lb": res["root_lb"]})
            print(f"bnb_{args.bnb_bound:7s} n={n:5d} seed={seed} {res['status']:8s} "
                  f"t={res['time_bnb']:8.2f}s exp={res['nodes_expanded']}", flush=True)

    Path(args.out).parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
