"""Versões ingênuas (estilo de código gerado sem revisão), usadas só como linha
de base em audit_naive.py: best-first com cópia do caminho, `in` em lista e
limite recalculado em Python puro; TATT/Christofides no grafo completo do NetworkX.
"""
from __future__ import annotations

import heapq
import math
import time

import networkx as nx
import numpy as np

from .mst import prim


def _naive_bound(D, path, n):
    total = 0
    for v in range(n):
        if v in path:
            i = path.index(v)
            fixed = []
            if i > 0:
                fixed.append(D[v][path[i - 1]])
            if i < len(path) - 1:
                fixed.append(D[v][path[i + 1]])
        else:
            fixed = []
        if len(fixed) == 2:
            total += fixed[0] + fixed[1]
            continue
        others = sorted(D[v][u] for u in range(n) if u != v)
        if len(fixed) == 1:
            # a 2ª aresta é a menor que não seja a fixada
            rest = list(others)
            rest.remove(fixed[0])
            total += fixed[0] + rest[0]
        else:
            total += others[0] + others[1]
    return math.ceil(total / 2)


def naive_bnb(D: np.ndarray, time_limit: float = 60.0, max_open: int = 2_000_000,
              initial_tour: list[int] | None = None) -> dict:
    D = D.tolist()
    n = len(D)
    t0 = time.perf_counter()
    best_cost, best_path = math.inf, None
    if initial_tour is not None:
        best_path = list(initial_tour)
        best_cost = sum(D[best_path[i]][best_path[(i + 1) % n]] for i in range(n))
    heap = [(_naive_bound(D, [0], n), 0, [0])]
    expanded = generated = pruned = max_open_seen = leaves = 0
    status = "optimal"
    while heap:
        if time.perf_counter() - t0 > time_limit:
            status = "timeout"
            break
        bound, cost, path = heapq.heappop(heap)
        if bound >= best_cost:
            pruned += 1
            continue
        if len(path) == n:
            leaves += 1
            total = cost + D[path[-1]][0]
            if total < best_cost:
                best_cost, best_path = total, path
            continue
        expanded += 1
        for v in range(n):
            if v not in path:                       # O(n) em lista
                new_path = path + [v]               # cópia O(n) por filho
                generated += 1
                b = _naive_bound(D, new_path, n)    # O(n^2) do zero
                if b < best_cost:
                    heapq.heappush(heap, (b, cost + D[path[-1]][v], new_path))
                else:
                    pruned += 1
        max_open_seen = max(max_open_seen, len(heap))
        if len(heap) > max_open:
            status = "memout"
            break
    return {"status": status, "cost": best_cost if status == "optimal" else None,
            "best_found": None if math.isinf(best_cost) else best_cost,
            "tour": best_path, "nodes_expanded": expanded, "nodes_generated": generated,
            "leaves": leaves,
            "nodes_pruned": pruned, "max_open": max_open_seen,
            "time": time.perf_counter() - t0}


def _complete_graph(D: np.ndarray) -> nx.Graph:
    n = D.shape[0]
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=float(D[i, j]))
    return G


def naive_tatt(D: np.ndarray) -> list[int]:
    G = _complete_graph(D)
    T = nx.minimum_spanning_tree(G)
    return list(nx.dfs_preorder_nodes(T, source=0))


def naive_christofides(D: np.ndarray) -> list[int]:
    G = _complete_graph(D)
    T = nx.minimum_spanning_tree(G)
    odd = [v for v, d in T.degree() if d % 2 == 1]
    M = nx.min_weight_matching(G.subgraph(odd))
    MG = nx.MultiGraph(T)
    MG.add_edges_from(M)
    tour, seen = [], set()
    for u, _ in nx.eulerian_circuit(MG, source=0):
        if u not in seen:
            seen.add(u)
            tour.append(u)
    return tour


def sequential_children_mst(S: np.ndarray) -> np.ndarray:
    """Versão anterior de mst.mst_without_each_vertex: m chamadas do Prim."""
    m = S.shape[0]
    out = np.zeros(m)
    for c in range(m):
        idx = np.array([i for i in range(m) if i != c])
        if idx.size > 1:
            out[c] = prim(S[np.ix_(idx, idx)])[1]
    return out
