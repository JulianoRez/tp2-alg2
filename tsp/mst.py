"""Árvore Geradora Mínima (algoritmo de Prim em O(n^2) sobre matriz densa).

Por que Prim O(n^2) e não Kruskal ou Prim com heap?
O grafo do TSP é completo (m = n(n-1)/2 arestas). Kruskal custa
O(m log m) = O(n^2 log n) e precisa materializar todas as arestas; Prim com
heap custa O(m log n) = O(n^2 log n). Prim com vetor de chaves custa O(n^2),
que é ótimo para grafo denso (cada aresta é olhada uma vez) e usa memória O(n)
além da matriz. Cada iteração é vetorizada com NumPy (o laço Python é só
sobre os n vértices, não sobre as n^2 arestas).
"""
from __future__ import annotations

import numpy as np


def prim(W: np.ndarray, root: int = 0) -> tuple[np.ndarray, float]:
    """MST do grafo completo com matriz de pesos W (n x n).

    Devolve (parent, peso_total): parent[v] é o pai de v na árvore enraizada
    em `root` (parent[root] = -1).
    """
    n = W.shape[0]
    parent = np.full(n, root, dtype=np.int64)
    parent[root] = -1
    if n == 1:
        return parent, 0.0
    in_tree = np.zeros(n, dtype=bool)
    in_tree[root] = True
    key = W[root].astype(float)          # custo da aresta mais barata até a árvore
    key[root] = np.inf
    total = 0.0
    for _ in range(n - 1):
        u = int(np.argmin(key))          # vértice fora da árvore mais próximo
        total += key[u]
        in_tree[u] = True
        key[u] = np.inf
        row = W[u]
        better = (row < key) & ~in_tree  # relaxamento: u oferece aresta melhor?
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
