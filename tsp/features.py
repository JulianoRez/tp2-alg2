"""Features geométricas e da MST das instâncias (análise de sensibilidade)."""
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
    """Entropia normalizada da ocupação de uma grade k x k (1 = uniforme)."""
    n = len(P)
    k = k or max(2, int(round(np.sqrt(n / 2))))
    lo, hi = P.min(axis=0), P.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    cell = np.clip(np.floor((P - lo) / span * k).astype(int), 0, k - 1)
    counts = np.bincount(cell[:, 0] * k + cell[:, 1], minlength=k * k)
    p = counts[counts > 0] / n
    return float(-(p * np.log(p)).sum() / np.log(k * k))


def triangle_violations(D: np.ndarray) -> tuple[float, float]:
    """Fração de triplas com d(i,k) > d(i,j) + d(j,k) e maior violação relativa."""
    n = D.shape[0]
    via = D[:, :, None] + D[None, :, :]          # via[i, j, k] = d(i,j) + d(j,k)
    direct = D[:, None, :]
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
    """Melhor silhueta do k-means para k = 2..kmax e o k correspondente."""
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

    w, h = P.max(axis=0) - P.min(axis=0)
    f["aspect_ratio"] = float(min(w, h) / max(w, h)) if max(w, h) > 0 else 1.0
    hull = convex_hull(P)
    f["hull_fraction"] = len(hull) / n
    f["spatial_entropy"] = spatial_entropy(P)
    G = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(G, np.inf)
    nn_geo = G.min(axis=1)
    # Clark-Evans (<1 agrupado, ~1 aleatório, >1 regular) com a correção de
    # borda de Donnelly (1978) para o bounding box de área A e perímetro L
    area_bb, perim_bb = float(w * h), float(2 * (w + h))
    esperado = 0.5 * np.sqrt(area_bb / n) + (0.0514 + 0.041 / np.sqrt(n)) * perim_bb / n
    f["clark_evans"] = float(nn_geo.mean() / esperado) if area_bb > 0 else np.nan
    f["silhouette_best"], f["kmeans_best_k"] = clusterability(P)

    nn_d = Dinf.min(axis=1)
    f["nn_dist_cv"] = _cv(nn_d)
    iu = np.triu_indices(n, 1)
    f["dist_cv"] = _cv(D[iu])

    parent, mst_w = prim(D)
    deg = tree_degrees(parent)
    edge_w = D[np.arange(n)[parent >= 0], parent[parent >= 0]]
    f["mst_weight"] = mst_w
    f["mst_max_degree"] = int(deg.max())
    f["mst_mean_degree_nonleaf"] = float(deg[deg > 1].mean()) if (deg > 1).any() else 0.0
    f["mst_leaf_fraction"] = float((deg == 1).mean())
    f["mst_odd_fraction"] = float((deg % 2 == 1).mean())
    f["mst_edge_cv"] = _cv(edge_w)

    f["triangle_violation_frac"], f["triangle_violation_max_rel"] = triangle_violations(D)
    if inst.known_opt:
        f["known_opt"] = inst.known_opt
        f["opt_over_mst"] = inst.known_opt / mst_w if mst_w > 0 else np.nan
    return f
