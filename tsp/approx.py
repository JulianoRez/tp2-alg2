"""Algoritmos aproximativos para o TSP métrico.

* Twice-around-the-tree (2-aproximado): MST + percurso em pré-ordem.
* Christofides (1.5-aproximado): MST + emparelhamento perfeito de peso mínimo
  nos vértices de grau ímpar + circuito euleriano + atalhos.

As garantias valem quando d satisfaz a desigualdade triangular. Distâncias
arredondadas (EUC_2D, GEO, ATT) ou matrizes EXPLICIT de estradas podem violá-la
levemente (ver feature `triangle_violation_frac` em features.py).

Cada etapa roda dentro de `prof.section(...)` para profiling por subrotina.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from .heuristics import normalize_tour
from .mst import prim, tree_degrees
from .profiler import NullProfiler, Profiler


# --------------------------------------------------------------------------
# Twice-around-the-tree
# --------------------------------------------------------------------------

def preorder(parent: np.ndarray, root: int = 0) -> list[int]:
    """Pré-ordem (DFS iterativa) da árvore dada por `parent`.

    Equivale a percorrer a árvore "duas vezes em volta" (cada aresta 2x) e
    pular os vértices repetidos: pela desigualdade triangular, o atalho nunca
    aumenta o custo, logo custo(tour) <= 2 * MST <= 2 * OPT.
    """
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
        stack.extend(reversed(children[v]))   # visita filhos em ordem crescente
    return tour


def twice_around_the_tree(D: np.ndarray, prof: Profiler | None = None) -> list[int]:
    prof = prof or NullProfiler()
    with prof.section("mst"):
        parent, _ = prim(D, root=0)
    with prof.section("preorder"):
        tour = preorder(parent, root=0)
    return tour


# --------------------------------------------------------------------------
# Christofides
# --------------------------------------------------------------------------

def odd_degree_vertices(parent: np.ndarray) -> np.ndarray:
    """Vértices de grau ímpar na MST (sempre em número par: handshaking lemma)."""
    return np.flatnonzero(tree_degrees(parent) % 2 == 1)


def blossom_matching(D: np.ndarray, odd: np.ndarray,
                     prof: Profiler | None = None) -> list[tuple[int, int]]:
    """Emparelhamento perfeito de PESO MÍNIMO nos vértices `odd` (Edmonds/blossom).

    O NetworkX só oferece emparelhamento de peso MÁXIMO. Transformação:
    w'(u,v) = C - d(u,v), com C > max d. Com maxcardinality=True o resultado
    é perfeito (grafo completo com |odd| par) e, como todo emparelhamento
    perfeito tem exatamente |odd|/2 arestas, maximizar sum(C - d) equivale a
    minimizar sum(d). Complexidade do blossom: O(k^3), k = |odd|.
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
    """Emparelhamento guloso: pares em ordem crescente de distância.

    O(k^2 log k), mas SEM garantia de mínimo -> Christofides perde a garantia
    de 1.5. Serve para medir o trade-off tempo x qualidade do matching.
    """
    prof = prof or NullProfiler()
    with prof.section("matching_solve"):
        k = len(odd)
        sub = D[np.ix_(odd, odd)]
        iu, ju = np.triu_indices(k, 1)
        order = np.argsort(sub[iu, ju], kind="stable")
        matched = bytearray(k)
        pairs = []
        odd_l = odd.tolist()
        # listas Python no laço: indexar arrays NumPy elemento a elemento é lento
        for u, v in zip(iu[order].tolist(), ju[order].tolist()):
            if not matched[u] and not matched[v]:
                matched[u] = matched[v] = 1
                pairs.append((odd_l[u], odd_l[v]))
                if len(pairs) == k // 2:
                    break
    return pairs


def eulerian_circuit(n: int, edges: list[tuple[int, int]], start: int = 0) -> list[int]:
    """Algoritmo de Hierholzer (iterativo) em multigrafo, O(V + E).

    Cada vértice guarda um ponteiro para a próxima aresta ainda não testada,
    então cada aresta é examinada O(1) vezes no total.
    """
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
    """Remove repetições do circuito euleriano (atalhos), mantendo a 1ª visita."""
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
