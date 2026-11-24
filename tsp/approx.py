"""Twice-around-the-tree e Christofides (blossom ou guloso), com seções de profiling."""
from __future__ import annotations

import networkx as nx
import numpy as np

from .heuristics import normalize_tour
from .mst import prim, tree_degrees
from .profiler import NullProfiler, Profiler


def preorder(parent: np.ndarray, root: int = 0) -> list[int]:
    """Pré-ordem iterativa da árvore dada por `parent` (filhos em ordem crescente)."""
    n = len(parent)
    children: list[list[int]] = [[] for _ in range(n)]
    for v in range(n):
        p = parent[v]
        if p >= 0:
            children[p].append(v)
    tour, stack = [], [root]
    while stack:
        v = stack.pop()
        tour.append(v)
        stack.extend(reversed(children[v]))
    return tour


def twice_around_the_tree(D: np.ndarray, prof: Profiler | None = None) -> list[int]:
    prof = prof or NullProfiler()
    with prof.section("mst"):
        parent, _ = prim(D, root=0)
    with prof.section("preorder"):
        tour = preorder(parent, root=0)
    return tour


def odd_degree_vertices(parent: np.ndarray) -> np.ndarray:
    """Vértices de grau ímpar na MST (sempre em número par: handshaking lemma)."""
    return np.flatnonzero(tree_degrees(parent) % 2 == 1)


def blossom_matching(D: np.ndarray, odd: np.ndarray,
                     prof: Profiler | None = None) -> list[tuple[int, int]]:
    """Emparelhamento perfeito de peso mínimo nos vértices `odd`.

    O NetworkX só faz peso máximo: com w' = C - d e maxcardinality=True, todo
    emparelhamento perfeito tem |odd|/2 arestas, então maximizar w' minimiza d.
    """
    prof = prof or NullProfiler()
    k = len(odd)
    with prof.section("matching_build"):
        sub = D[np.ix_(odd, odd)]
        C = float(sub.max()) + 1.0
        iu, ju = np.triu_indices(k, 1)
        G = nx.Graph()
        G.add_weighted_edges_from(zip(iu.tolist(), ju.tolist(), (C - sub[iu, ju]).tolist()))
    with prof.section("matching_solve"):
        M = nx.max_weight_matching(G, maxcardinality=True)
    return [(int(odd[u]), int(odd[v])) for u, v in M]


def greedy_matching(D: np.ndarray, odd: np.ndarray,
                    prof: Profiler | None = None) -> list[tuple[int, int]]:
    """Emparelhamento guloso por distância crescente (sem a garantia de 1,5)."""
    prof = prof or NullProfiler()
    with prof.section("matching_solve"):
        k = len(odd)
        sub = D[np.ix_(odd, odd)]
        iu, ju = np.triu_indices(k, 1)
        order = np.argsort(sub[iu, ju], kind="stable")
        matched = bytearray(k)
        pairs = []
        odd_l = odd.tolist()
        for u, v in zip(iu[order].tolist(), ju[order].tolist()):
            if not matched[u] and not matched[v]:
                matched[u] = matched[v] = 1
                pairs.append((odd_l[u], odd_l[v]))
                if len(pairs) == k // 2:
                    break
    return pairs


def eulerian_circuit(n: int, edges: list[tuple[int, int]], start: int = 0) -> list[int]:
    """Hierholzer iterativo em multigrafo, O(V + E)."""
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for eid, (u, v) in enumerate(edges):
        adj[u].append((v, eid))
        adj[v].append((u, eid))
    used = bytearray(len(edges))
    ptr = [0] * n
    stack, circuit = [start], []
    while stack:
        v = stack[-1]
        while ptr[v] < len(adj[v]) and used[adj[v][ptr[v]][1]]:
            ptr[v] += 1
        if ptr[v] == len(adj[v]):
            circuit.append(stack.pop())
        else:
            u, eid = adj[v][ptr[v]]
            used[eid] = 1
            stack.append(u)
    circuit.reverse()
    return circuit


def shortcut(circuit: list[int], n: int) -> list[int]:
    seen = bytearray(n)
    tour = []
    for v in circuit:
        if not seen[v]:
            seen[v] = 1
            tour.append(v)
    return tour


def christofides(D: np.ndarray, prof: Profiler | None = None,
                 matching: str = "blossom") -> list[int]:
    prof = prof or NullProfiler()
    n = D.shape[0]
    with prof.section("mst"):
        parent, _ = prim(D, root=0)
    with prof.section("odd_vertices"):
        odd = odd_degree_vertices(parent)
    if matching == "blossom":
        pairs = blossom_matching(D, odd, prof)
    elif matching == "greedy":
        pairs = greedy_matching(D, odd, prof)
    else:
        raise ValueError(f"matching desconhecido: {matching}")
    with prof.section("euler"):
        edges = [(int(parent[v]), v) for v in range(n) if parent[v] >= 0] + pairs
        circuit = eulerian_circuit(n, edges, start=0)
    with prof.section("shortcut"):
        tour = shortcut(circuit, n)
    return normalize_tour(tour)
