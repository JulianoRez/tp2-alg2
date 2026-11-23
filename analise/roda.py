#!/usr/bin/env python3
"""Roda toda a análise: figuras (relatorio/figuras), tabelas LaTeX
(relatorio/tabelas), CSVs para a planilha (results/analise) e os números
citados no texto (results/analise/numeros.json).

    python -m analise.roda
"""
from __future__ import annotations

import warnings

import pandas as pd

from . import auditoria, complexidade, sensibilidade, warmstart
from .comum import RES, fmt_num, salva_numeros, tabela_latex

FONTE = {"node_coord": "coordenadas", "display_data": "DISPLAY\\_DATA", "mds": "MDS"}


def tabela_instancias():
    f = pd.read_csv(RES / "features.csv").sort_values(["n", "instance"])
    t = pd.DataFrame({
        "Instância": f.instance, "$N$": f.n,
        "Tipo": f.edge_weight_type.str.replace("_", "\\_", regex=False),
        "Ótimo": f.known_opt.astype(int),
        "Geometria": f.coords_source.map(FONTE),
        "Fração no fecho": f.hull_fraction.map(lambda v: fmt_num(v, 2)),
        "Clark-Evans": f.clark_evans.map(lambda v: fmt_num(v, 2)),
        "Viol. triangular": f.triangle_violation_frac.map(lambda v: fmt_num(100 * v, 2) + "\\%"),
    })
    tabela_latex(t, "instancias",
                 "Instâncias da TSPLIB usadas. Geometria: origem das coordenadas das "
                 "features geométricas. Viol. triangular: fração das triplas $(i,j,k)$ "
                 "com $d(i,k)>d(i,j)+d(j,k)$.", "tab:instancias", "lrlrlrrr")


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    tabela_instancias()
    for nome, mod in (("complexidade", complexidade), ("warm-start", warmstart),
                      ("sensibilidade", sensibilidade), ("auditoria", auditoria)):
        print(f"== {nome}", flush=True)
        mod.roda()
    salva_numeros()
    print("ok: relatorio/figuras, relatorio/tabelas, results/analise")


if __name__ == "__main__":
    main()
