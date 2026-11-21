"""Instâncias sintéticas "adversariais" (EUC_2D) para a análise de sensibilidade.

Com só 15 instâncias da TSPLIB, um modelo que relaciona ~20 features ao gap
das heurísticas tem mais variáveis do que pontos. Estas famílias geram
instâncias com características controladas, cada uma pensada para estressar
uma feature pedida no enunciado:

    uniform        pontos uniformes no quadrado (referência)
    cluster        3 a 6 aglomerados gaussianos (silhueta alta, Clark-Evans < 1)
    grid           grade regular com pequeno ruído (Clark-Evans > 1, NN empata muito)
    circle         pontos num anel (quase todos no fecho convexo)
    strip          faixa estreita (razão de aspecto 0,05 a 0,2)
    core_outliers  núcleo denso + 20% de pontos espalhados (CV do vizinho mais próximo alto)
    spokes         pontos sobre 5 a 8 raios a partir do centro (hubs na MST)

Coordenadas em [0, 10000]^2 e distância EUC_2D oficial da TSPLIB (nint), como
nas instâncias reais. Tudo é determinístico dado (família, n, semente).
"""
from __future__ import annotations

import zlib

import numpy as np

from .tsplib import Instance, dist_euc_2d

L = 10_000.0
FAMILIES = ("uniform", "cluster", "grid", "circle", "strip", "core_outliers", "spokes")


def _rng(family: str, n: int, seed: int) -> np.random.Generator:
    # semente estável entre execuções (hash() do Python é aleatorizado por processo)
    return np.random.default_rng([zlib.crc32(family.encode()), n, seed])


def _points(family: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if family == "uniform":
        return rng.uniform(0, L, (n, 2))
    if family == "cluster":
        k = int(rng.integers(3, 7))
        centers = rng.uniform(0.1 * L, 0.9 * L, (k, 2))
        sigma = rng.uniform(0.02, 0.05) * L
        lab = rng.integers(0, k, n)
        return centers[lab] + rng.normal(0, sigma, (n, 2))
    if family == "grid":
        g = int(np.ceil(np.sqrt(n)))
        step = L / g
        cells = rng.choice(g * g, n, replace=False)
        P = np.stack([cells % g, cells // g], axis=1) * step + step / 2
        return P + rng.normal(0, 0.03 * step, (n, 2))
    if family == "circle":
        ang = rng.uniform(0, 2 * np.pi, n)
        r = 0.45 * L * (1 + rng.normal(0, 0.01, n))
        return L / 2 + np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
    if family == "strip":
        a = rng.uniform(0.05, 0.2)
        return rng.uniform(0, 1, (n, 2)) * [L, a * L]
    if family == "core_outliers":
        m = max(2, int(round(0.2 * n)))
        core = L / 2 + rng.normal(0, 0.04 * L, (n - m, 2))
        return np.vstack([core, rng.uniform(0, L, (m, 2))])
    if family == "spokes":
        k = int(rng.integers(5, 9))
        ang = rng.integers(0, k, n) * (2 * np.pi / k) + rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0.02, 0.45, n) * L
        P = L / 2 + np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
        return P + rng.normal(0, 0.005 * L, (n, 2))
    raise ValueError(f"família desconhecida: {family}")


def make_instance(family: str, n: int, seed: int) -> Instance:
    P = _points(family, n, _rng(family, n, seed))
    P = np.round(P, 1)                      # como nos arquivos da TSPLIB
    D = dist_euc_2d(P)
    np.fill_diagonal(D, 0)
    return Instance(name=f"{family}_n{n}_s{seed}", n=n, edge_weight_type="EUC_2D",
                    D=D, coords=P, coords_source="node_coord",
                    comment=f"sintética: {family}")
