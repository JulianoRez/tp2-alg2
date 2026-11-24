#!/usr/bin/env python3
"""Tabelas para a planilha (results/planilha/*.csv), a partir dos resultados brutos."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiment_config import STARTS

RES = Path("results")
OUT = RES / "planilha"
MS = 1000.0
MIB = 2.0 ** 20


def load_runs() -> pd.DataFrame:
    rows = [json.loads(line) for line in (RES / "runs.jsonl").open() if line.strip()]
    return pd.DataFrame(rows)


def bnb_table(runs: pd.DataFrame, gc_on: bool = True) -> pd.DataFrame:
    b = runs[(runs.algo == "bnb") & (runs.gc_enabled.fillna(True) == gc_on)].copy()
    if b.empty:
        return b
    opt = b.known_opt
    b["initial_gap"] = (b.initial_ub - opt) / opt
    b["best_gap"] = (b.best_found - opt) / opt
    b["root_gap"] = (opt - b.root_lb) / opt
    if "mem_peak_bnb" in b:
        b["mem_bnb_kib"] = b.mem_peak_bnb / 1024
        b["mem_raiz_kib"] = b.mem_peak_root_bound / 1024
    cols = ["instance", "n", "bound", "child_order", "start", "status",
            "nodes_expanded", "nodes_pruned", "time_total", "cost", "best_found",
            "known_opt", "best_gap", "initial_ub", "initial_gap", "root_lb", "root_gap",
            "nodes_generated", "leaves", "ub_updates", "max_depth",
            "time_heuristic", "time_root_bound", "time_child_bounds", "time_search",
            "time_to_best", "hk_iterations", "cpu_time_total", "mem_bnb_kib",
            "mem_raiz_kib", "gc_collections", "edge_weight_type", "lote"]
    b["start"] = pd.Categorical(b.start, STARTS, ordered=True)
    b = b.sort_values(["bound", "child_order", "n", "instance", "start"])
    return b[[c for c in cols if c in b.columns]]


def warmstart_table(bnb: pd.DataFrame) -> pd.DataFrame:
    if bnb.empty:
        return bnb
    keys = ["instance", "n", "bound", "child_order"]
    exp = bnb.pivot_table(index=keys, columns="start", values="nodes_expanded",
                          aggfunc="first", observed=False)
    pru = bnb.pivot_table(index=keys, columns="start", values="nodes_pruned",
                          aggfunc="first", observed=False)
    tim = bnb.pivot_table(index=keys, columns="start", values="time_total",
                          aggfunc="first", observed=False)
    exp.columns = [f"expandidos_{s}" for s in exp.columns]
    pru.columns = [f"podados_{s}" for s in pru.columns]
    tim.columns = [f"tempo_s_{s}" for s in tim.columns]
    unfinished = bnb[bnb.status != "optimal"].astype({"start": str})
    to = (unfinished.groupby(keys)["start"].agg(", ".join).rename("sem_terminar"))
    w = exp.join(pru).join(tim).join(to).reset_index()
    w["sem_terminar"] = w["sem_terminar"].astype(object).fillna("")
    return w.sort_values(["bound", "child_order", "n", "instance"])


def heuristics_table(runs: pd.DataFrame) -> pd.DataFrame:
    h = runs[runs.algo != "bnb"].copy()
    h["algoritmo"] = h.algo.where(h.algo != "christofides",
                                  "christofides_" + h.matching.fillna(""))
    cols = ["instance", "n", "algoritmo", "cost", "known_opt", "gap"]
    cols += time_mem_columns(h) + ["gc_collections", "edge_weight_type"]
    return h.sort_values(["n", "instance", "algoritmo"])[cols]


SECTIONS = ["mst", "preorder", "odd_vertices", "matching_build", "matching_solve",
            "euler", "shortcut", "nn", "hull", "hull_fecho", "hull_insercao", "trivial"]


def time_mem_columns(df: pd.DataFrame) -> list[str]:
    """Cria colunas em ms (tempos) e KiB (memória) e devolve seus nomes."""
    df["time_total_ms"] = df.time_total * MS
    df["time_min_ms"] = df.time_total_min * MS
    cols = ["time_total_ms", "time_min_ms"]
    for s in SECTIONS:
        if f"time_{s}" in df:
            df[f"t_{s}_ms"] = df[f"time_{s}"] * MS
            cols.append(f"t_{s}_ms")
    for s in ["total"] + SECTIONS:
        if f"mem_peak_{s}" in df:
            df[f"mem_{s}_kib"] = df[f"mem_peak_{s}"] / 1024
            cols.append(f"mem_{s}_kib")
    return cols


def scaling_table() -> pd.DataFrame:
    s = pd.read_csv(RES / "scaling.csv")
    cols = ["algo", "n", "seed", "cost", "mst_weight", "ratio_to_mst"]
    cols += time_mem_columns(s) + ["gc_collections", "gc_enabled"]
    return s.sort_values(["algo", "n", "seed"])[cols]


def instances_table(runs: pd.DataFrame, heur: pd.DataFrame, bnb: pd.DataFrame) -> pd.DataFrame:
    f = pd.read_csv(RES / "features.csv")
    gaps = heur.pivot_table(index="instance", columns="algoritmo", values="gap",
                            aggfunc="first")
    gaps.columns = [f"gap_{c}" for c in gaps.columns]
    f = f.merge(gaps, left_on="instance", right_index=True, how="left")
    if not bnb.empty:
        rg = bnb.groupby(["instance", "bound"]).root_gap.first().unstack()
        rg.columns = [f"root_gap_{c}" for c in rg.columns]
        f = f.merge(rg, left_on="instance", right_index=True, how="left")
    first = ["instance", "n", "edge_weight_type", "coords_source", "known_opt"]
    gap_cols = [c for c in f.columns if c.startswith(("gap_", "root_gap_"))]
    rest = [c for c in f.columns if c not in first + gap_cols]
    return f.sort_values("n")[first + gap_cols + rest]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    bnb = bnb_table(runs)
    heur = heuristics_table(runs)
    tables = {
        "1_warmstart": warmstart_table(bnb),
        "2_bnb": bnb,
        "3_heuristicas": heur,
        "4_instancias": instances_table(runs, heur, bnb),
    }
    nogc = bnb_table(runs, gc_on=False)
    if not nogc.empty:
        tables["8_bnb_sem_gc"] = nogc
    if (RES / "scaling.csv").exists():
        tables["5_escala"] = scaling_table()
    escala_bnb = sorted(RES.glob("scaling_bnb_*.csv"))
    if escala_bnb:
        tables["6_escala_bnb"] = pd.concat([pd.read_csv(a) for a in escala_bnb], ignore_index=True)
    if (RES / "scaling_nogc.csv").exists():
        s = pd.read_csv(RES / "scaling_nogc.csv")
        cols = ["algo", "n", "seed", "cost"] + time_mem_columns(s) + ["gc_collections", "gc_enabled"]
        tables["5b_escala_sem_gc"] = s.sort_values(["algo", "n", "seed"])[cols]
    if (RES / "synthetic.csv").exists():
        tables["9_sinteticas"] = pd.read_csv(RES / "synthetic.csv")
    if (RES / "audit.csv").exists():
        a = pd.read_csv(RES / "audit.csv")
        a["mem_peak_mib"] = a.pop("mem_peak") / MIB
        tables["7_auditoria"] = a
    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", index=False, na_rep="NA")
        print(f"{name:16s} {len(df):4d} linhas x {df.shape[1]:3d} colunas")


if __name__ == "__main__":
    main()
