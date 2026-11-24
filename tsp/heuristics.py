"""Heurísticas construtivas de warm-start. Todos os tours começam no vértice 0."""
from __future__ import annotations

import numpy as np


def normalize_tour(tour) -> list[int]:
    """Rotaciona o tour para começar no vértice 0."""
    tour = [int(v) for v in tour]
    k = tour.index(0)
    return tour[k:] + tour[:k]


def trivial_tour(n: int) -> list[int]:
    return list(range(n))


def nearest_neighbor_tour(D: np.ndarray, start: int = 0) -> list[int]:
    n = D.shape[0]
    # asarray não copia a matriz se ela já é float64 (astype sempre copiava)
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


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points: np.ndarray) -> list[int]:
    """Fecho convexo (cadeia monótona de Andrew), anti-horário, sem pontos colineares."""
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
    """Fecho convexo + inserção mais barata.

    O tour é uma lista ligada (succ) e cada vértice de fora guarda sua melhor
    inserção; inserir k em (a, b) só invalida quem tinha (a, b) como melhor aresta.
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
        succ[a], succ[k] = k, b
        in_tour[k] = True
        best_inc[k] = np.inf

        rest = np.flatnonzero(~in_tour)
        if rest.size == 0:
            break
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
