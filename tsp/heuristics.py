"""Heurísticas construtivas usadas como warm-start do Branch-and-Bound.

Todas devolvem um tour como lista de vértices (0..n-1) começando em 0.
Complexidades (n = número de vértices):
    trivial_tour                 O(n)
    nearest_neighbor_tour        O(n^2)
    convex_hull_insertion_tour   O(n log n) fecho + O(n^2) inserções (típico)
"""
from __future__ import annotations

import numpy as np


def normalize_tour(tour) -> list[int]:
    """Rotaciona o tour para começar no vértice 0 (facilita comparar tours)."""
    tour = [int(v) for v in tour]
    k = tour.index(0)
    return tour[k:] + tour[:k]


# --------------------------------------------------------------------------
# 1. Cold start / trivial
# --------------------------------------------------------------------------

def trivial_tour(n: int) -> list[int]:
    """Solução ingênua 1 -> 2 -> ... -> n -> 1 (aqui indexada a partir de 0)."""
    return list(range(n))


# --------------------------------------------------------------------------
# 2. Vizinho mais próximo (guloso)
# --------------------------------------------------------------------------

def nearest_neighbor_tour(D: np.ndarray, start: int = 0) -> list[int]:
    """A partir de `start`, vai sempre para o vizinho não visitado mais próximo."""
    n = D.shape[0]
    # np.asarray não copia se D já é float64. (A 1ª versão usava D.astype(float),
    # que SEMPRE copia a matriz: o profiling de memória mostrou 7.7 MiB extras
    # em n = 1000, contra O(n) necessários.)
    dist = np.asarray(D, dtype=float)
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]
    cur = start
    for _ in range(n - 1):
        row = np.where(visited, np.inf, dist[cur])
        cur = int(np.argmin(row))
        visited[cur] = True
        tour.append(cur)
    return normalize_tour(tour)


# --------------------------------------------------------------------------
# 3. Inserção a partir do fecho convexo
# --------------------------------------------------------------------------

def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points: np.ndarray) -> list[int]:
    """Fecho convexo por Andrew's monotone chain, O(n log n).

    Devolve os índices dos vértices do fecho em sentido anti-horário. Pontos
    colineares sobre uma aresta do fecho NÃO entram (são tratados como
    internos e inseridos depois).
    """
    pts = [tuple(p) for p in np.asarray(points, dtype=float)]
    order = sorted(range(len(pts)), key=lambda i: pts[i])
    if len(order) <= 2:
        return order

    def half(seq):
        chain: list[int] = []
        for i in seq:
            while len(chain) >= 2 and _cross(pts[chain[-2]], pts[chain[-1]], pts[i]) <= 0:
                chain.pop()
            chain.append(i)
        return chain

    lower = half(order)
    upper = half(reversed(order))
    return lower[:-1] + upper[:-1]


def convex_hull_insertion_tour(D: np.ndarray, coords: np.ndarray, prof=None) -> list[int]:
    """Sub-rota inicial = fecho convexo; depois inserção mais barata (gulosa).

    A cada passo escolhe, entre todos os vértices ainda fora do tour e todas as
    arestas (a, b) do tour, o par que minimiza o acréscimo
        d(a, k) + d(k, b) - d(a, b)
    e insere k entre a e b.

    Implementação: o tour é uma lista ligada (succ[a] = b, cada aresta é
    identificada pela sua cauda `a`). Para cada vértice de fora guardamos a
    melhor inserção conhecida (best_inc, best_tail). Inserir k entre a e b
    destrói só a aresta (a, b) e cria (a, k) e (k, b); então basta comparar
    cada vértice com as 2 arestas novas, e só quem tinha (a, b) como melhor
    aresta precisa ser recalculado contra o tour inteiro.
    """
    from .profiler import NullProfiler
    prof = prof or NullProfiler()
    n = D.shape[0]
    with prof.section("hull_fecho"):
        hull = convex_hull(coords)
    with prof.section("hull_insercao"):
        return _cheapest_insertion(D, hull, n)


def _cheapest_insertion(D: np.ndarray, hull: list[int], n: int) -> list[int]:
    if len(hull) < 2:
        hull = [0, 1] if n > 1 else [0]
    if n <= len(hull):
        return normalize_tour(hull)

    succ = np.full(n, -1, dtype=np.int64)
    for a, b in zip(hull, hull[1:] + hull[:1]):
        succ[a] = b
    in_tour = np.zeros(n, dtype=bool)
    in_tour[hull] = True

    def all_edges():
        tails = np.flatnonzero(in_tour)
        return tails, succ[tails]

    outside = np.flatnonzero(~in_tour)
    tails, heads = all_edges()
    inc = D[outside[:, None], tails] + D[outside[:, None], heads] - D[tails, heads]
    best_inc = np.full(n, np.inf)
    best_tail = np.full(n, -1, dtype=np.int64)
    best_inc[outside] = inc.min(axis=1)
    best_tail[outside] = tails[inc.argmin(axis=1)]

    for _ in range(len(outside)):
        k = int(np.argmin(best_inc))       # quem já está no tour tem best_inc = inf
        a = int(best_tail[k])
        b = int(succ[a])
        succ[a], succ[k] = k, b            # insere k entre a e b
        in_tour[k] = True
        best_inc[k] = np.inf

        rest = np.flatnonzero(~in_tour)
        if rest.size == 0:
            break
        # compara com as duas arestas novas (a,k) e (k,b)
        inc1 = D[rest, a] + D[rest, k] - D[a, k]
        inc2 = D[rest, k] + D[rest, b] - D[k, b]
        stale = best_tail[rest] == a       # melhor aresta era (a,b), que sumiu
        for new_inc, tail in ((inc1, a), (inc2, k)):
            upd = ~stale & (new_inc < best_inc[rest])
            best_inc[rest[upd]] = new_inc[upd]
            best_tail[rest[upd]] = tail
        if stale.any():
            s = rest[stale]
            tails, heads = all_edges()
            inc = D[s[:, None], tails] + D[s[:, None], heads] - D[tails, heads]
            best_inc[s] = inc.min(axis=1)
            best_tail[s] = tails[inc.argmin(axis=1)]

    tour = [0]
    v = int(succ[0])
    while v != 0:
        tour.append(v)
        v = int(succ[v])
    return tour
