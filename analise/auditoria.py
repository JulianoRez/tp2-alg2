"""Auditoria: código "bruto" (tsp/naive.py) x refatorado (results/audit.csv)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .comum import ANALISE, RES, fmt_num, numero, plt, salva_fig, tabela_latex

VAR_ROTULO = {
    "sequential": "m Prims sequenciais", "batched": "Prim em lote",
    "leaf_shortcut": "atalho das folhas",
    "naive_tatt": "TATT bruto (NetworkX)", "tatt": "TATT refatorado",
    "naive_christofides": "Christofides bruto (NetworkX)", "christofides": "Christofides refatorado",
    "naive_bestfirst_twomin": "bruto: best-first, cópias, LB em Python",
    "naive_bestfirst_twomin_semgc": "bruto, GC desligado",
    "bestfirst_twomin": "best-first refatorado",
    "bestfirst_twomin_semgc": "best-first refatorado, GC desligado",
    "dfs_twomin": "DFS, 2 menores arestas", "dfs_onetree": "DFS, 1-árvore",
    "dfs_hk": "DFS, Held-Karp",
}


def roda():
    a = pd.read_csv(RES / "audit.csv")
    a["mem_mib"] = a.mem_peak / 2 ** 20

    # 1) MST de cada filho
    c = a[a.experiment == "children_mst"].pivot(index="n", columns="variant", values="time") * 1e6
    c.to_csv(ANALISE / "auditoria_children_mst.csv")
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for v, mk in (("sequential", "o-"), ("batched", "s-"), ("leaf_shortcut", "^--")):
        if v in c:
            ax.loglog(c.index, c[v], mk, ms=3, lw=1, label=VAR_ROTULO[v])
    ax.set_xlabel("$m$ (vértices não visitados)")
    ax.set_ylabel("tempo por nó ($\\mu$s)")
    ax.legend(frameon=False, fontsize=6)
    salva_fig(fig, "auditoria_children_mst")
    mmax = c.index.max()
    numero("aud_mst_speedup_lote_mmax", float(c.loc[mmax, "sequential"] / c.loc[mmax, "batched"]))
    numero("aud_mst_mmax", int(mmax))
    if "leaf_shortcut" in c:
        numero("aud_folhas_vs_lote_min", float((c.leaf_shortcut / c.batched).min()))
        numero("aud_folhas_vs_lote_max", float((c.leaf_shortcut / c.batched).max()))

    # 2) aproximativos
    ap = a[a.experiment == "approx"]
    linhas = []
    for case, g in ap.groupby("case", sort=False):
        g = g.set_index("variant")
        for alg in ("tatt", "christofides"):
            b, r = g.loc[f"naive_{alg}"], g.loc[alg]
            linhas.append({"caso": case, "n": int(r.n), "algoritmo": alg,
                           "tempo_bruto_ms": b.time * 1e3, "tempo_ref_ms": r.time * 1e3,
                           "speedup": b.time / r.time,
                           "mem_bruto_mib": b.mem_mib, "mem_ref_mib": r.mem_mib,
                           "reducao_mem": b.mem_mib / r.mem_mib})
    apx = pd.DataFrame(linhas)
    apx.to_csv(ANALISE / "auditoria_aproximativos.csv", index=False)
    t = pd.DataFrame({
        "Caso": apx.caso, "$N$": apx.n,
        "Algoritmo": apx.algoritmo.map({"tatt": "TATT", "christofides": "Christofides"}),
        "Tempo bruto (ms)": apx.tempo_bruto_ms.map(lambda v: fmt_num(v, 2)),
        "Tempo ref. (ms)": apx.tempo_ref_ms.map(lambda v: fmt_num(v, 2)),
        "Speedup": apx.speedup.map(lambda v: fmt_num(v, 1) + "$\\times$"),
        "Mem. bruto (MiB)": apx.mem_bruto_mib.map(lambda v: fmt_num(v, 2)),
        "Mem. ref. (MiB)": apx.mem_ref_mib.map(lambda v: fmt_num(v, 2)),
    })
    tabela_latex(t, "auditoria_aproximativos",
                 "Auditoria dos aproximativos: versão bruta (grafo completo no NetworkX) "
                 "x refatorada (Prim $O(n^2)$ na matriz, matching só nos vértices ímpares). "
                 "Tempo: mediana de 3 execuções; memória: pico do tracemalloc.",
                 "tab:auditoria_aprox", "lrlrrrrr")
    big = apx[apx.n == apx.n.max()]
    for _, r in big.iterrows():
        numero(f"aud_{r.algoritmo}_speedup_nmax", float(r.speedup))
        numero(f"aud_{r.algoritmo}_redmem_nmax", float(r.reducao_mem))

    # 3) B&B
    b = a[a.experiment == "bnb"].copy()
    b.to_csv(ANALISE / "auditoria_bnb.csv", index=False)
    t = pd.DataFrame({
        "Instância": b.case, "Variante": b.variant.map(VAR_ROTULO),
        "Status": b.status.map({"optimal": "ótimo", "timeout": "timeout", "memout": "memória"}),
        "Nós exp.": b.nodes_expanded.map(lambda v: fmt_num(v, 0)),
        "Tempo (s)": b.time.map(lambda v: fmt_num(v, 2)),
        "Pico (MiB)": b.mem_mib.map(lambda v: fmt_num(v, 1)),
        "B/nó aberto": b.bytes_per_open_node.map(lambda v: "--" if pd.isna(v) else fmt_num(v, 0)),
    })
    tabela_latex(t, "auditoria_bnb",
                 "Auditoria do B\\&B (limite de 30 s por execução; todas as variantes com o "
                 "mesmo warm-start, vizinho mais próximo, e o mesmo teto de 2 milhões de nós "
                 "abertos). B/nó aberto: pico de memória dividido pelo máximo de nós abertos, "
                 "ambos da mesma execução com tracemalloc.",
                 "tab:auditoria_bnb", "llcrrrr")
    g = b.set_index(["case", "variant"])
    for inst in b.case.unique():
        try:
            numero(f"aud_bnb_{inst}_naive_t", float(g.loc[(inst, "naive_bestfirst_twomin"), "time"]))
            numero(f"aud_bnb_{inst}_dfshk_t", float(g.loc[(inst, "dfs_hk"), "time"]))
            numero(f"aud_bnb_{inst}_naive_mem", float(g.loc[(inst, "naive_bestfirst_twomin"), "mem_mib"]))
            numero(f"aud_bnb_{inst}_best_mem", float(g.loc[(inst, "bestfirst_twomin"), "mem_mib"]))
            numero(f"aud_bnb_{inst}_dfs_mem", float(g.loc[(inst, "dfs_twomin"), "mem_mib"]))
        except KeyError:
            pass
    gcr = []
    for inst in b.case.unique():
        for base in ("naive_bestfirst_twomin", "bestfirst_twomin"):
            try:
                on, off = g.loc[(inst, base)], g.loc[(inst, base + "_semgc")]
            except KeyError:
                continue
            gcr.append({"caso": inst, "variante": base, "tempo_gc": on.time, "tempo_sem_gc": off.time,
                        "nos_gc": on.nodes_expanded, "nos_sem_gc": off.nodes_expanded,
                        "nos_por_s_gc": on.nodes_expanded / on.time,
                        "nos_por_s_sem_gc": off.nodes_expanded / off.time})
    gcr = pd.DataFrame(gcr)
    gcr.to_csv(ANALISE / "auditoria_gc_bestfirst.csv", index=False)
    if not gcr.empty:
        gcr["razao"] = gcr.nos_por_s_sem_gc / gcr.nos_por_s_gc
        numero("aud_gc_bestfirst_razao_vazao_max", float(gcr.razao.max()))
        numero("aud_gc_bestfirst_razao_vazao_min", float(gcr.razao.min()))
    return c, apx, b, gcr
