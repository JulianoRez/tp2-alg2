#!/usr/bin/env python3
"""Executa UMA configuração (instância x algoritmo) e imprime um JSON por linha.

Exemplos:
    python run_one.py gr17 --algo christofides --reps 5 --trace-memory
    python run_one.py att48 --algo bnb --start hull --bound hk --child-order index
    python run_one.py bays29 --algo bnb --start trivial --bound onetree --time-limit 60

Cada execução de B&B roda num processo separado (é o que run_experiments.py
faz), então `peak_rss_kb` (pico de memória residente do processo, via
getrusage) mede só aquela execução.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import statistics
import sys
import tracemalloc
from time import perf_counter

import numpy as np

from tsp.approx import christofides, twice_around_the_tree
from tsp.bnb import BOUNDS, CHILD_ORDERS, SEARCHES, BranchAndBound
from tsp.heuristics import convex_hull_insertion_tour, nearest_neighbor_tour, trivial_tour
from tsp.profiler import NullProfiler, Profiler
from tsp.tsplib import find_instance, load_instance

HEURISTICS = ("trivial", "nn", "hull", "tatt", "christofides")
ALGOS = HEURISTICS + ("bnb",)


def build_tour(name: str, inst, prof: Profiler, matching: str = "blossom") -> list[int]:
    """Roda uma heurística/aproximação e devolve o tour."""
    D = inst.D
    if name == "trivial":
        with prof.section("trivial"):
            return trivial_tour(inst.n)
    if name == "nn":
        with prof.section("nn"):
            return nearest_neighbor_tour(D)
    if name == "hull":
        with prof.section("hull"):
            return convex_hull_insertion_tour(D, inst.coords)
    if name == "tatt":
        return twice_around_the_tree(D, prof)
    if name == "christofides":
        return christofides(D, prof, matching=matching)
    raise ValueError(name)


def gc_collections() -> int:
    return sum(s["collections"] for s in gc.get_stats())


def peak_rss_kb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Linux: KB


def run_heuristic(args, inst) -> dict:
    """1 execução de aquecimento (descartada) + `reps` execuções cronometradas
    (sem tracemalloc) + 1 opcional com tracemalloc.

    O aquecimento existe porque a 1ª chamada paga custos que não são do
    algoritmo (importações tardias do NetworkX, caches da CPU, alocador).
    As coletas do GC são contadas só DENTRO da janela cronometrada: o
    gc.collect() que limpa o lixo antes de cada repetição fica de fora.
    """
    totals, sections, tour = [], {}, None
    gc0 = gc_collections()
    for _ in range(args.reps):
        prof = Profiler()
        gc.collect()
        t0 = perf_counter()
        tour = build_tour(args.algo, inst, prof, args.matching)
        totals.append(perf_counter() - t0)
        for k, v in prof.times.items():
            sections.setdefault(k, []).append(v)
    gc_auto = gc_collections() - gc0
    out = {
        "status": "ok",
        "cost": inst.tour_cost(tour),
        "tour": tour,
        "reps": args.reps,
        "time_total": statistics.median(totals),
        "time_total_min": min(totals),
        "time_total_all": totals,
        "gc_collections": gc_auto / args.reps,      # coletas automáticas por execução
    }
    for k, v in sections.items():
        out[f"time_{k}"] = statistics.median(v)
    if args.trace_memory:
        prof = Profiler(trace_memory=True)
        gc.collect()
        tracemalloc.start()
        with prof.section("total"):
            build_tour(args.algo, inst, prof, args.matching)
        tracemalloc.stop()
        out.update({k: v for k, v in prof.as_dict().items() if k.startswith("mem_peak_")})
    return out


def run_bnb(args, inst) -> dict:
    """Heurística de warm-start + Branch-and-Bound, tudo dentro do tempo limite.

    Mede tempo de relógio (perf_counter) e tempo de CPU do processo
    (process_time): com vários processos em paralelo, a diferença entre os
    dois mostra quanto a contenção da máquina pesou na medição.

    Com --trace-memory, faz uma 2ª execução SEPARADA sob tracemalloc (que
    deixa o Python bem mais lento), limitada a --mem-time-limit segundos:
    a memória da DFS é O(n^2) e atinge o pico logo no 1º mergulho, então uma
    execução curta já mede o pico. O RSS do processo não serve aqui: os
    ~170 MiB do interpretador + NumPy/SciPy/NetworkX escondem os KiB da DFS.
    """
    gc0 = gc_collections()
    t0 = perf_counter()
    prof = Profiler()
    start_tour = build_tour(args.start, inst, prof, args.matching)
    t_heur = perf_counter() - t0
    bnb = BranchAndBound(inst.D, bound=args.bound, search=args.search,
                         child_order=args.child_order,
                         time_limit=max(args.time_limit - t_heur, 0.0),
                         max_open=args.max_open)
    res = bnb.solve(start_tour)
    res["time_heuristic"] = t_heur
    res["time_total"] = perf_counter() - t0
    res["gc_collections"] = gc_collections() - gc0
    return res


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instance", help="nome (ex: gr17) ou caminho do .tsp/.tsp.gz")
    ap.add_argument("--algo", choices=ALGOS, required=True)
    ap.add_argument("--start", choices=HEURISTICS, default="trivial",
                    help="warm-start do B&B (limite superior inicial)")
    ap.add_argument("--bound", choices=BOUNDS, default="hk")
    ap.add_argument("--search", choices=SEARCHES, default="dfs")
    ap.add_argument("--child-order", choices=CHILD_ORDERS, default="bound")
    ap.add_argument("--matching", choices=("blossom", "greedy"), default="blossom",
                    help="emparelhamento usado pelo Christofides")
    ap.add_argument("--time-limit", type=float, default=1800.0, help="segundos (padrão 30 min)")
    ap.add_argument("--max-open", type=int, default=2_000_000,
                    help="teto de nós abertos no best-first (proteção de memória)")
    ap.add_argument("--reps", type=int, default=5, help="repetições p/ heurísticas")
    ap.add_argument("--trace-memory", action="store_true",
                    help="execução extra com tracemalloc (pico por subrotina)")
    ap.add_argument("--no-gc", action="store_true", help="desliga o garbage collector")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args(argv)

    inst = load_instance(find_instance(args.instance, args.data_dir))
    if args.no_gc:
        gc.disable()
    rss_before = peak_rss_kb()
    res = run_bnb(args, inst) if args.algo == "bnb" else run_heuristic(args, inst)

    record = {
        "instance": inst.name, "n": inst.n, "edge_weight_type": inst.edge_weight_type,
        "algo": args.algo,
        "start": args.start if args.algo == "bnb" else None,
        "bound": args.bound if args.algo == "bnb" else None,
        "search": args.search if args.algo == "bnb" else None,
        "child_order": args.child_order if args.algo == "bnb" else None,
        "matching": args.matching if args.algo == "christofides" else None,
        "time_limit": args.time_limit if args.algo == "bnb" else None,
        "gc_enabled": not args.no_gc,
        "known_opt": inst.known_opt,
    }
    record.update(res)
    cost = record.get("cost")
    record["gap"] = (cost - inst.known_opt) / inst.known_opt if cost and inst.known_opt else None
    record["peak_rss_kb"] = peak_rss_kb()
    record["peak_rss_before_run_kb"] = rss_before   # baseline: interpretador + imports
    record["python"] = platform.python_version()
    record["numpy"] = np.__version__
    print(json.dumps(record))
    return record


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
