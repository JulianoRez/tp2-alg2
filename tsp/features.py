"""Extração de características (features) geométricas/topológicas das instâncias.

Usadas na análise de sensibilidade (gap das aproximações x características).
Features geométricas usam `inst.coords` (coordenadas reais, projeção da GEO,
DISPLAY_DATA ou MDS — ver coluna `coords_source`); features de grafo usam a
matriz de distâncias real `inst.D`. Todas são invariantes à escala.
"""
from __future__ import annotations

import warnings

import numpy as np

from .heuristics import convex_hull
from .mst import prim, tree_degrees
from .tsplib import Instance, classical_mds


def _cv(x: np.ndarray) -> float:
    m = float(np.mean(x))
    return float(np.std(x) / m) if m > 0 else 0.0


def polygon_area(P: np.ndarray) -> float:
    """Fórmula do laço (shoelace) para a área do polígono de vértices P (em ordem)."""
    if len(P) < 3:
        return 0.0
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def spatial_entropy(P: np.ndarray, k: int | None = None) -> float:
    """Entropia de Shannon normalizada da ocupação de uma grade k x k sobre o
    bounding box. 1 = pontos espalhados uniformemente; perto de 0 = concentrados.
    k ~ sqrt(n/2) -> ~2 pontos por célula em média."""
    n = len(P)
    k = k or max(2, int(round(np.sqrt(n / 2))))
    lo, hi = P.min(axis=0), P.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    cell = np.clip(np.floor((P - lo) / span * k).astype(int), 0, k - 1)
    counts = np.bincount(cell[:, 0] * k + cell[:, 1], minlength=k * k)
    p = counts[counts > 0] / n
    return float(-(p * np.log(p)).sum() / np.log(k * k))


def triangle_violations(D: np.ndarray) -> tuple[float, float]:
    """Fração de triplas (i, j, k) com d(i,k) > d(i,j) + d(j,k) e a maior
    violação relativa. As garantias de 2 e 1.5 assumem violação zero."""
    n = D.shape[0]
    via = D[:, :, None] + D[None, :, :]          # via[i, j, k] = d(i,j) + d(j,k)
    direct = D[:, None, :]                       # d(i,k)
    valid = np.ones((n, n, n), dtype=bool)
    idx = np.arange(n)
    valid[idx, idx, :] = False                   # j == i
    valid[:, idx, idx] = False                   # j == k
    valid[idx, :, idx] = False                   # i == k
    excess = np.where(valid, direct - via, -np.inf)
    frac = float((excess > 0).sum() / valid.sum())
    rel = np.where(valid & (excess > 0), excess / np.maximum(direct, 1e-12), 0.0)
    return frac, float(rel.max())


def clusterability(P: np.ndarray, kmax: int = 8) -> tuple[float, int]:
    """Melhor silhueta do K-Means para k = 2..kmax e o k correspondente.
    Silhueta alta (> ~0.5) indica pontos organizados em clusters bem separados."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(P)
    best, best_k = -1.0, 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(2, min(kmax, n - 1) + 1):
            labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(P)
            if len(set(labels)) < 2:
                continue
            s = float(silhouette_score(P, labels))
            if s > best:
                best, best_k = s, k
    return best, best_k


def instance_features(inst: Instance) -> dict:
    P, D, n = np.asarray(inst.coords, dtype=float), inst.D, inst.n
    Dinf = D.copy()
    np.fill_diagonal(Dinf, np.inf)
    f: dict = {"instance": inst.name, "n": n,
               "edge_weight_type": inst.edge_weight_type,
               "coords_source": inst.coords_source}
    if inst.coords_source == "mds":
        f["mds_quality"] = classical_mds(D)[1]

    # --- geometria ----------------------------------------------------
    w, h = P.max(axis=0) - P.min(axis=0)
    f["aspect_ratio"] = float(min(w, h) / max(w, h)) if max(w, h) > 0 else 1.0
    hull = convex_hull(P)
    f["hull_fraction"] = len(hull) / n
    f["spatial_entropy"] = spatial_entropy(P)
    G = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(G, np.inf)
    nn_geo = G.min(axis=1)
    area = polygon_area(P[hull])
    f["clark_evans"] = float(nn_geo.mean() / (0.5 * np.sqrt(area / n))) if area > 0 else np.nan
    f["silhouette_best"], f["kmeans_best_k"] = clusterability(P)

    # --- métrica real (matriz D) ---------------------------------------
    nn_d = Dinf.min(axis=1)
    f["nn_dist_cv"] = _cv(nn_d)                       # heterogeneidade local
    iu = np.triu_indices(n, 1)
    f["dist_cv"] = _cv(D[iu])

    parent, mst_w = prim(D)
    deg = tree_degrees(parent)
    edge_w = D[np.arange(n)[parent >= 0], parent[parent >= 0]]
    f["mst_weight"] = mst_w
    f["mst_max_degree"] = int(deg.max())              # presença de hubs
    f["mst_mean_degree_nonleaf"] = float(deg[deg > 1].mean()) if (deg > 1).any() else 0.0
    f["mst_leaf_fraction"] = float((deg == 1).mean())
    f["mst_odd_fraction"] = float((deg % 2 == 1).mean())  # tamanho do matching do Christofides
    f["mst_edge_cv"] = _cv(edge_w)

    f["triangle_violation_frac"], f["triangle_violation_max_rel"] = triangle_violations(D)
    if inst.known_opt:
        f["known_opt"] = inst.known_opt
        f["opt_over_mst"] = inst.known_opt / mst_w if mst_w > 0 else np.nan
    return f
