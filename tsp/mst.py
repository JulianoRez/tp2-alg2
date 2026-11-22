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


def mst_without_each_vertex(S: np.ndarray, leaf_shortcut: bool = False) -> np.ndarray:
    """Para cada vértice c de S, peso da MST do grafo S sem o vértice c.

    É a subrotina mais cara do Branch-and-Bound: num nó com m vértices ainda
    não visitados, cada um dos m filhos precisa da MST dos m-1 restantes.

    1. Atalho das folhas (O(m^2) no total): calcula UMA MST T de S. Se c é
       folha de T, então T - c é MST de S - c (se existisse árvore geradora
       de S - c mais leve, somando a aresta de c teríamos uma árvore de S mais
       leve que T). Logo MST(S - c) = peso(T) - peso da única aresta de c.
    2. Vértices internos de T: removê-los parte a árvore, então rodamos o Prim
       só para eles, todos "em paralelo" (_batched_prim_without): o laço
       Python tem m-2 iterações e cada uma é uma operação NumPy sobre os
       problemas. Custo O(k m^2) para k vértices internos, em vez de O(m^3).

    S: m x m com diagonal = +inf. Devolve vetor de tamanho m.

    Padrão: `leaf_shortcut=False` (Prim em lote para TODOS os vértices).
    O atalho das folhas é assintoticamente melhor, mas medido (audit_naive.py,
    experimento children_mst) ficou 1,3 a 2,1x MAIS LENTO para m <= 70: ele
    exige um Prim extra com m iterações do interpretador, e nesse tamanho o
    custo é dominado pelo número de iterações Python, não pela aritmética
    (que o NumPy faz em microssegundos). Fica disponível para a auditoria.
    """
    m = S.shape[0]
    if m <= 2:                       # sobra no máximo 1 vértice: MST vazia
        return np.zeros(m)
    if not leaf_shortcut:
        return _batched_prim_without(S, np.arange(m))
    parent, total = prim(S)
    deg = tree_degrees(parent)
    out = np.empty(m)
    leaves = np.flatnonzero(deg == 1)
    # peso da única aresta de cada folha: (v, parent[v]); a raiz 0 pode ser folha,
    # e aí a aresta dela é a do seu único filho
    nb = parent[leaves].copy()
    if nb.size and nb[0] == -1:              # leaves está ordenado: a raiz 0 vem primeiro
        nb[0] = int(np.flatnonzero(parent == 0)[0])
    out[leaves] = total - S[leaves, nb]
    internal = np.flatnonzero(deg > 1)
    if internal.size:
        out[internal] = _batched_prim_without(S, internal)
    return out


def _batched_prim_without(S: np.ndarray, removed: np.ndarray) -> np.ndarray:
    """MST de S - {removed[k]} para cada k, com os Prims rodando em lote.

    key[k, v]: aresta mais barata de v até a árvore do problema k.
    pen[k, v] = inf se v já está na árvore do problema k ou foi removido.
    """
    m = S.shape[0]
    p = removed.size
    rows = np.arange(p)
    start = np.where(removed == 0, 1, 0)     # problema k começa no 0 (ou no 1 se removeu o 0)
    key = S[start].copy()
    pen = np.zeros((p, m))
    pen[rows, removed] = np.inf
    pen[rows, start] = np.inf
    key += pen
    total = np.zeros(p)
    for _ in range(m - 2):
        j = key.argmin(axis=1)               # próximo vértice de cada problema
        total += key[rows, j]
        pen[rows, j] = np.inf
        np.minimum(key, S[j], out=key)       # relaxamento com a linha do vértice novo
        key += pen                           # (inf + x = inf mantém os já usados fora)
    return total
