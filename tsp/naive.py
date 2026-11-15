"""Versões "brutas" (estilo código gerado sem revisão) — LINHA DE BASE.

NÃO são usadas nos experimentos principais. Existem só para a auditoria
(audit_naive.py): medir, com profiling, o que as refatorações ganharam.

Problemas típicos reproduzidos aqui (e como foram resolvidos na versão final):

naive_bnb
  1. Best-first com heap guardando CÓPIAS do caminho em cada nó aberto: a
     fila cresce exponencialmente -> estouro de memória.
     -> versão final: DFS recursiva com um único caminho/vetor de visitados
        mutado no lugar (memória O(n^2)).
  2. `v not in path` em lista: O(n) por teste.  -> vetor booleano `visited`.
  3. Limite inferior recalculado do zero em Python puro, O(n^2) por nó, com
     o limite fraco das "duas menores arestas".
     -> limites 1-árvore/Held-Karp calculados para todos os filhos de uma vez
        com NumPy (mst_without_each_vertex).

naive_christofides / naive_tatt
  1. Monta o grafo completo no NetworkX: n(n-1)/2 arestas como dicionários
     Python (~centenas de bytes por aresta) só para rodar uma MST.
     -> Prim O(n^2) direto sobre a matriz NumPy.
  2. Matching sobre `G.subgraph(odd)` + `min_weight_matching`, que copia o
     subgrafo inteiro de novo para inverter os pesos.
     -> monta só o grafo dos vértices ímpares já com pesos invertidos.
  3. MultiGraph + nx.eulerian_circuit (objetos pesados).
     -> Hierholzer com listas de adjacência.
"""
from __future__ import annotations

import heapq
import math
import time

import networkx as nx
import numpy as np

from .mst import prim


def _naive_bound(D, path, n):
    """Limite das 2 menores arestas, recalculado do zero (O(n^2) em Python)."""
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


def naive_bnb(D: np.ndarray, time_limit: float = 60.0, max_open: int = 5_000_000) -> dict:
    D = D.tolist()            # lista de listas, como o código bruto costuma fazer
    n = len(D)
    t0 = time.perf_counter()
    best_cost, best_path = math.inf, None
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
