#!/usr/bin/env python3
"""Experimento de sensibilidade em instâncias sintéticas (tsp/synthetic.py).

Para cada instância (família x N x semente):
  1. extrai as mesmas features das instâncias da TSPLIB (tsp/features.py);
  2. roda as heurísticas e aproximativos (custo e tempo);
  3. acha o ÓTIMO EXATO com o B&B (Held-Karp, DFS, filhos por limite,
     warm-start = melhor entre fecho convexo e Christofides) -> gap exato;
  4. mede a dificuldade do B&B para cada limite inferior: gap na raiz
     (2 menores arestas, 1-árvore, Held-Karp) e nós que o B&B com 1-árvore
     precisa para PROVAR o ótimo quando já recebe o tour ótimo como incumbente
     (isola a força do limite da qualidade do warm-start; só para N <= 20).

Saída: results/synthetic.jsonl (retomável) e results/synthetic.csv.

    python run_synthetic.py --workers 8
    python run_synthetic.py --sizes 20 30 --seeds 0 1 --families uniform circle
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from tsp.approx import christofides, twice_around_the_tree
from tsp.bnb import BranchAndBound, one_tree
from tsp.features import instance_features
from tsp.heuristics import convex_hull_insertion_tour, nearest_neighbor_tour, trivial_tour
from tsp.synthetic import FAMILIES, make_instance
from tsp.tsplib import tour_cost

OUT = Path("results/synthetic.jsonl")


def heuristics(inst) -> dict[str, list[int]]:
    D, P = inst.D, inst.coords
    return {
        "trivial": lambda: trivial_tour(inst.n),
        "nn": lambda: nearest_neighbor_tour(D),
        "hull": lambda: convex_hull_insertion_tour(D, P),
        "tatt": lambda: twice_around_the_tree(D),
        "christofides_blossom": lambda: christofides(D, matching="blossom"),
        "christofides_greedy": lambda: christofides(D, matching="greedy"),
    }


def run_instance(family: str, n: int, seed: int, time_limit: float,
                 prove_limit: float, prove_max_n: int = 20) -> dict:
    inst = make_instance(family, n, seed)
    rec = {"instance": inst.name, "family": family, "n": n, "seed": seed}
    t0 = perf_counter()
    feats = instance_features(inst)
    rec["time_features"] = perf_counter() - t0
    rec.update({k: v for k, v in feats.items() if k not in rec})

    tours = {}
    for name, fn in heuristics(inst).items():
        t0 = perf_counter()
        tours[name] = fn()
        rec[f"time_{name}"] = perf_counter() - t0
        rec[f"cost_{name}"] = tour_cost(inst.D, tours[name])

    # ótimo exato: Held-Karp com a melhor incumbente disponível
    start = min(("hull", "christofides_blossom"), key=lambda h: rec[f"cost_{h}"])
    res = BranchAndBound(inst.D, bound="hk", search="dfs", child_order="bound",
                         time_limit=time_limit).solve(tours[start])
    rec.update({"exact_status": res["status"], "opt": res["best_found"],
                "hk_root_lb": res["root_lb"], "hk_iterations": res["hk_iterations"],
                "hk_nodes_expanded": res["nodes_expanded"],
                "hk_time": res["time_bnb"]})
    opt = res["best_found"]
    for name in tours:
        rec[f"gap_{name}"] = (rec[f"cost_{name}"] - opt) / opt

    # limites na raiz
    Dinf = inst.D.astype(float).copy()
    np.fill_diagonal(Dinf, np.inf)
    rec["twomin_root_lb"] = float(np.partition(Dinf, 1, axis=1)[:, :2].sum() / 2)
    rec["onetree_root_lb"] = float(one_tree(Dinf)[0])
    for b in ("twomin", "onetree", "hk"):
        rec[f"root_gap_{b}"] = (opt - rec[f"{b}_root_lb"]) / opt

    # esforço para PROVAR o ótimo com a 1-árvore (incumbente = tour ótimo).
    # Acima de N ~ 20 quase tudo estoura o limite, então só roda em N pequeno.
    if n <= prove_max_n:
        prove = BranchAndBound(inst.D, bound="onetree", search="dfs", child_order="bound",
                               time_limit=prove_limit).solve(res["tour"])
        rec.update({"onetree_prove_status": prove["status"],
                    "onetree_prove_nodes": prove["nodes_expanded"],
                    "onetree_prove_time": prove["time_bnb"]})
    return rec


def _refeatures(rec: dict) -> dict:
    inst = make_instance(rec["family"], rec["n"], rec["seed"])
    rec.update({k: v for k, v in instance_features(inst).items()
                if k not in ("instance", "n")})
    return rec


def refeatures(workers: int):
    """Recalcula só as features (ex.: depois de corrigir uma feature), sem
    rodar de novo heurísticas e B&B."""
    rows = [json.loads(line) for line in OUT.open() if line.strip()]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_refeatures, rows, chunksize=4))
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    pd.DataFrame(rows).to_csv("results/synthetic.csv", index=False)
    print(f"features recalculadas para {len(rows)} instâncias")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--sizes", nargs="+", type=int, default=[20, 30, 40, 50])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(6)))
    ap.add_argument("--time-limit", type=float, default=900.0,
                    help="limite do B&B Held-Karp que acha o ótimo (s)")
    ap.add_argument("--prove-limit", type=float, default=120.0,
                    help="limite do B&B 1-árvore que só prova o ótimo (s)")
    ap.add_argument("--prove-max-n", type=int, default=20,
                    help="só mede a prova com 1-árvore para N <= este valor")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--refeatures", action="store_true",
                    help="só recalcula as features das instâncias já rodadas")
    args = ap.parse_args()
    if args.refeatures:
        refeatures(args.workers)
        return

    OUT.parent.mkdir(exist_ok=True)
    done = set()
    if OUT.exists():
        done = {json.loads(line)["instance"] for line in OUT.open() if line.strip()}
    jobs = [(f, n, s) for n in args.sizes for f in args.families for s in args.seeds
            if f"{f}_n{n}_s{s}" not in done]
    print(f"{len(done)} prontas, {len(jobs)} a rodar", flush=True)

    with ProcessPoolExecutor(args.workers) as ex, OUT.open("a") as fh:
        futs = {ex.submit(run_instance, f, n, s, args.time_limit, args.prove_limit,
                            args.prove_max_n): (f, n, s)
                for f, n, s in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[{i}/{len(jobs)}] {rec['instance']:24s} {rec['exact_status']:8s} "
                  f"opt={rec['opt']} hk_nodes={rec['hk_nodes_expanded']} "
                  f"gap_chr={rec['gap_christofides_blossom']:.3f} "
                  f"1tree_nodes={rec.get('onetree_prove_nodes')}", flush=True)

    rows = [json.loads(line) for line in OUT.open() if line.strip()]
    pd.DataFrame(rows).to_csv("results/synthetic.csv", index=False)
    print(f"-> results/synthetic.csv ({len(rows)} instâncias)")


if __name__ == "__main__":
    main()
