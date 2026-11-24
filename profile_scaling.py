#!/usr/bin/env python3
"""Profiling por subrotina em instâncias uniformes de N crescente (regressão de
complexidade) e, opcionalmente, o B&B em N pequeno.

    python profile_scaling.py
    python profile_scaling.py --no-gc --out results/scaling_nogc.csv
    python profile_scaling.py --sizes --bnb-sizes 10 15 20 --bnb-bound hk --out results/scaling_bnb_hk.csv
"""
from __future__ import annotations

import os

# BLAS com 1 thread (antes de importar o NumPy): só adicionaria ruído
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import gc  # noqa: E402
import statistics  # noqa: E402
import tracemalloc  # noqa: E402
from pathlib import Path  # noqa: E402
from time import perf_counter, process_time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tsp.approx import christofides, twice_around_the_tree  # noqa: E402
from tsp.bnb import BranchAndBound, held_karp_bound, one_tree  # noqa: E402
from tsp.heuristics import convex_hull_insertion_tour, nearest_neighbor_tour  # noqa: E402
from tsp.mst import prim  # noqa: E402
from tsp.profiler import Profiler  # noqa: E402
from tsp.tsplib import dist_euc_2d, tour_cost  # noqa: E402


def random_instance(n: int, seed: int):
    P = np.random.default_rng(seed).uniform(0, 10_000, (n, 2))
    return P, dist_euc_2d(P)


def algorithms(P, D):
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
        return convex_hull_insertion_tour(D, P, prof)


def gc_count() -> int:
    return sum(s["collections"] for s in gc.get_stats())


def profile_one(fn, reps: int) -> dict:
    """1 execução de aquecimento + `reps` cronometradas (mediana) + 1 sob tracemalloc.
    Coletas do GC contadas só dentro da janela cronometrada.
    """
    times, cpu, sections, tour, gc_auto = [], [], {}, None, 0
    fn(Profiler())                                   # aquecimento (descartado)
    for _ in range(reps):
        prof = Profiler()
        gc.collect()
        gc0 = gc_count()
        c0, t0 = process_time(), perf_counter()
        tour = fn(prof)
        times.append(perf_counter() - t0)
        cpu.append(process_time() - c0)
        gc_auto += gc_count() - gc0
        for k, v in prof.times.items():
            sections.setdefault(k, []).append(v)
    out = {"time_total": statistics.median(times), "time_total_min": min(times),
           "cpu_time_total": statistics.median(cpu),
           "gc_collections": gc_auto / reps, "tour": tour}
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
    ap.add_argument("--bnb-mem-time", type=float, default=20.0,
                    help="limite (s) da execução extra do B&B com tracemalloc")
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
            gc.collect()
            gc0 = gc_count()
            c0 = process_time()
            res = BranchAndBound(D, bound=args.bnb_bound, time_limit=args.bnb_time).solve(tour0)
            cpu = process_time() - c0
            gcs = gc_count() - gc0
            # memória: execução curta sob tracemalloc (a DFS atinge o pico cedo)
            gc.collect()
            tracemalloc.start()
            if args.bnb_bound == "hk":
                held_karp_bound(D)
            elif args.bnb_bound == "onetree":
                Dinf = D.astype(float)
                np.fill_diagonal(Dinf, np.inf)
                one_tree(Dinf)
            mem_root = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            gc.collect()
            tracemalloc.start()
            BranchAndBound(D, bound=args.bnb_bound,
                           time_limit=min(args.bnb_mem_time, args.bnb_time)).solve(tour0)
            mem_bnb = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            rows.append({"algo": f"bnb_{args.bnb_bound}", "n": n, "seed": seed,
                         "cost": res["cost"], "status": res["status"],
                         "time_total": res["time_bnb"], "cpu_time_total": cpu,
                         "time_root_bound": res["time_root_bound"],
                         "time_child_bounds": res["time_child_bounds"],
                         "nodes_expanded": res["nodes_expanded"],
                         "nodes_pruned": res["nodes_pruned"],
                         "max_depth": res["max_depth"],
                         "initial_ub": res["initial_ub"], "root_lb": res["root_lb"],
                         "gc_collections": gcs,
                         "mem_peak_root_bound": mem_root, "mem_peak_bnb": mem_bnb,
                         "gc_enabled": not args.no_gc})
            print(f"bnb_{args.bnb_bound:7s} n={n:5d} seed={seed} {res['status']:8s} "
                  f"t={res['time_bnb']:8.2f}s exp={res['nodes_expanded']}", flush=True)

    Path(args.out).parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
