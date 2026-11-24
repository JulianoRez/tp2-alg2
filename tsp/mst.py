"""Árvore geradora mínima: Prim O(n^2) em matriz densa e MST sem cada vértice (B&B)."""
from __future__ import annotations

import numpy as np


def prim(W: np.ndarray, root: int = 0) -> tuple[np.ndarray, float]:
    """MST do grafo completo W. Devolve (parent, peso), com parent[root] = -1."""
    n = W.shape[0]
    parent = np.full(n, root, dtype=np.int64)
    parent[root] = -1
    if n == 1:
        return parent, 0.0
    in_tree = np.zeros(n, dtype=bool)
    in_tree[root] = True
    key = W[root].astype(float)
    key[root] = np.inf
    total = 0.0
    for _ in range(n - 1):
        u = int(np.argmin(key))
        total += key[u]
        in_tree[u] = True
        key[u] = np.inf
        row = W[u]
        better = (row < key) & ~in_tree
        key[better] = row[better]
        parent[better] = u
    return parent, float(total)


def tree_adjacency(parent: np.ndarray) -> list[list[int]]:
    """Lista de adjacência (não dirigida) da árvore dada por `parent`."""
    adj: list[list[int]] = [[] for _ in range(len(parent))]
    for v, p in enumerate(parent):
        if p >= 0:
            adj[v].append(int(p))
            adj[int(p)].append(v)
    return adj


def tree_degrees(parent: np.ndarray) -> np.ndarray:
    n = len(parent)
    deg = np.zeros(n, dtype=np.int64)
    mask = parent >= 0
    deg[mask] += 1
    np.add.at(deg, parent[mask], 1)
    return deg


def mst_without_each_vertex(S: np.ndarray, leaf_shortcut: bool = False) -> np.ndarray:
    """Para cada vértice c de S, peso da MST de S sem c (S: m x m, diagonal +inf).

    Padrão: Prim em lote para todos os vértices. Com `leaf_shortcut=True`, as
    folhas da MST de S saem em O(1) (T - c é MST de S - c) e o lote roda só nos
    vértices internos; é assintoticamente melhor, mas mais lento para m <= 70.
    """
    m = S.shape[0]
    if m <= 2:
        return np.zeros(m)
    if not leaf_shortcut:
        return _batched_prim_without(S, np.arange(m))
    parent, total = prim(S)
    deg = tree_degrees(parent)
    out = np.empty(m)
    leaves = np.flatnonzero(deg == 1)
    # a raiz 0 também pode ser folha: a aresta dela é a do único filho
    nb = parent[leaves].copy()
    if nb.size and nb[0] == -1:
        nb[0] = int(np.flatnonzero(parent == 0)[0])
    out[leaves] = total - S[leaves, nb]
    internal = np.flatnonzero(deg > 1)
    if internal.size:
        out[internal] = _batched_prim_without(S, internal)
    return out


def _batched_prim_without(S: np.ndarray, removed: np.ndarray) -> np.ndarray:
    """MST de S - {removed[k]} para cada k, com os Prims rodando em lote."""
    m = S.shape[0]
    p = removed.size
    rows = np.arange(p)
    start = np.where(removed == 0, 1, 0)
    key = S[start].copy()
    pen = np.zeros((p, m))
    pen[rows, removed] = np.inf
    pen[rows, start] = np.inf
    key += pen
    total = np.zeros(p)
    for _ in range(m - 2):
        j = key.argmin(axis=1)
        total += key[rows, j]
        pen[rows, j] = np.inf
        np.minimum(key, S[j], out=key)
        key += pen                           # inf + x = inf: usados/removidos ficam fora
    return total
