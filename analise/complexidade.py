"""Seção 1 do enunciado: profiling por subrotina e complexidade empírica.

Entradas: results/scaling.csv (GC ligado), results/scaling_nogc.csv (GC
desligado), results/scaling_bnb_hk.csv e results/scaling_bnb_onetree.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .comum import (ANALISE, MODELO_TEX, RES, ajuste_modelo, ajuste_potencia, fmt_num,
                    melhor_modelo, numero, plt, salva_fig, tabela_latex)

# subrotinas medidas por algoritmo e a complexidade teórica de cada uma
SECOES = {
    "nn": ["nn"],
    "hull": ["hull"],
    "tatt": ["mst", "preorder"],
    "christofides": ["mst", "odd_vertices", "matching_build", "matching_solve", "euler", "shortcut"],
    "christofides_greedy": ["mst", "odd_vertices", "matching_solve", "euler", "shortcut"],
}
TEORIA_TEMPO = {
    ("nn", "nn"): "n^2", ("hull", "hull"): "n^2",
    ("tatt", "mst"): "n^2", ("tatt", "preorder"): "n", ("tatt", "total"): "n^2",
    ("christofides", "mst"): "n^2", ("christofides", "odd_vertices"): "n",
    ("christofides", "matching_build"): "n^2", ("christofides", "matching_solve"): "n^3",
    ("christofides", "euler"): "n", ("christofides", "shortcut"): "n",
    ("christofides", "total"): "n^3",
    ("christofides_greedy", "matching_solve"): "n^2 log n",
    ("christofides_greedy", "total"): "n^2 log n",
    ("nn", "total"): "n^2", ("hull", "total"): "n^2",
}
TEORIA_MEM = {
    ("nn", "total"): "n", ("hull", "total"): "n", ("tatt", "total"): "n",
    ("christofides", "matching_build"): "n^2", ("christofides", "matching_solve"): "n^2",
    ("christofides", "total"): "n^2", ("christofides_greedy", "total"): "n^2",
    ("tatt", "mst"): "n", ("christofides", "mst"): "n",
}
NOME = {"nn": "Viz. próximo", "hull": "Fecho convexo", "tatt": "TATT",
        "christofides": "Christofides (blossom)", "christofides_greedy": "Christofides (guloso)"}
SEC_NOME = {"total": "total", "nn": "total", "hull": "total", "mst": "MST (Prim)",
            "preorder": "pré-ordem", "odd_vertices": "vértices ímpares",
            "matching_build": "matching: grafo", "matching_solve": "matching: blossom",
            "euler": "Hierholzer", "shortcut": "atalhos"}


def _mediana(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["algo", "n"]).median(numeric_only=True).reset_index()


def tabela_tempo(sc: pd.DataFrame) -> pd.DataFrame:
    med = _mediana(sc)
    linhas = []
    for algo, secs in SECOES.items():
        d = med[med.algo == algo]
        if d.empty:
            continue
        for sec in ["total"] + [s for s in secs if s not in ("nn", "hull")]:
            col = "time_total" if sec == "total" else f"time_{sec}"
            if col not in d or d[col].isna().all():
                continue
            n, y = d.n.values, d[col].values
            pw = ajuste_potencia(n, y)
            best, tab = melhor_modelo(n, y)
            teo = TEORIA_TEMPO.get((algo, sec))
            lin = ajuste_modelo(n, y, teo) if teo else None
            linhas.append({
                "algo": algo, "secao": sec, "teoria": teo, "expoente": pw["b"],
                "expoente_ic95": pw["b_ic"], "r2_log_potencia": pw["r2_log"],
                "melhor_forma": best,
                "r2_teoria": lin["r2"] if lin else np.nan,
                "rmse_teoria_ms": lin["rmse"] * 1e3 if lin else np.nan,
                "rmse_log10_teoria": lin["rmse_log10_forma"] if lin else np.nan,
                "t_max_ms": float(y.max() * 1e3), "n_max": int(n.max()),
            })
    return pd.DataFrame(linhas)


def tabela_memoria(sc: pd.DataFrame) -> pd.DataFrame:
    med = _mediana(sc)
    linhas = []
    for algo, secs in SECOES.items():
        d = med[med.algo == algo]
        for sec in ["total"] + [s for s in secs if s not in ("nn", "hull")]:
            col = f"mem_peak_{sec}"
            if col not in d or d[col].isna().all():
                continue
            n, y = d.n.values, d[col].values.astype(float)
            if (y <= 0).all():
                continue
            pw = ajuste_potencia(n, y)
            best, _ = melhor_modelo(n, np.maximum(y, 1.0))
            linhas.append({"algo": algo, "secao": sec, "teoria": TEORIA_MEM.get((algo, sec)),
                           "expoente": pw["b"], "expoente_ic95": pw["b_ic"],
                           "r2_log_potencia": pw["r2_log"], "melhor_forma": best,
                           "mem_max_kib": float(y.max() / 1024), "n_max": int(n.max())})
    return pd.DataFrame(linhas)


def fig_tempo(sc: pd.DataFrame):
    med = _mediana(sc)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    ax = axes[0]
    for algo in SECOES:
        d = med[med.algo == algo]
        if d.empty:
            continue
        ax.loglog(d.n, d.time_total * 1e3, "o-", ms=3, lw=1, label=NOME[algo])
    ax.set_xlabel("N (vértices)")
    ax.set_ylabel("tempo total (ms)")
    ax.set_title("(a) algoritmos completos")
    ax.legend(frameon=False)
    ax = axes[1]
    d = med[med.algo == "christofides"]
    for sec in SECOES["christofides"]:
        ax.loglog(d.n, d[f"time_{sec}"] * 1e3, "o-", ms=3, lw=1, label=SEC_NOME[sec])
    # retas de referência
    n0 = d.n.values
    for k, ls in ((2, ":"), (3, "--")):
        ref = d.time_matching_solve.values[-1] * 1e3 * (n0 / n0[-1]) ** k
        ax.loglog(n0, ref, ls, color="k", lw=0.7, label=f"$\\propto n^{k}$")
    ax.set_xlabel("N (vértices)")
    ax.set_ylabel("tempo (ms)")
    ax.set_title("(b) subrotinas do Christofides")
    ax.legend(frameon=False, ncol=2, fontsize=6)
    salva_fig(fig, "tempo_vs_n")


def fig_memoria(sc: pd.DataFrame):
    med = _mediana(sc)
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for algo in SECOES:
        d = med[med.algo == algo]
        if d.empty:
            continue
        ax.loglog(d.n, d.mem_peak_total / 1024, "o-", ms=3, lw=1, label=NOME[algo])
    d = med[med.algo == "christofides"]
    ax.loglog(d.n, d.mem_peak_matching_build / 1024, "k--", lw=0.8, label="matching: grafo")
    ax.set_xlabel("N (vértices)")
    ax.set_ylabel("pico de memória extra (KiB)")
    ax.legend(frameon=False, fontsize=6)
    salva_fig(fig, "memoria_vs_n")


def fig_participacao(sc: pd.DataFrame):
    """Fração do tempo do Christofides em cada subrotina, por N."""
    med = _mediana(sc)
    d = med[med.algo == "christofides"].sort_values("n")
    secs = SECOES["christofides"]
    tot = d[[f"time_{s}" for s in secs]].sum(axis=1).values
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    base = np.zeros(len(d))
    for s in secs:
        frac = d[f"time_{s}"].values / tot
        ax.bar(range(len(d)), frac, bottom=base, label=SEC_NOME[s], width=0.8)
        base += frac
    ax.set_xticks(range(len(d)), d.n.astype(int).tolist(), rotation=45)
    ax.set_xlabel("N (vértices)")
    ax.set_ylabel("fração do tempo")
    ax.legend(frameon=False, fontsize=5.5, ncol=2, loc="lower left")
    salva_fig(fig, "christofides_participacao")
    numero("chr_frac_matching_nmax", float((d.time_matching_build.values[-1]
                                             + d.time_matching_solve.values[-1]) / tot[-1]))
    numero("chr_frac_matching_n25", float((d.time_matching_build.values[0]
                                            + d.time_matching_solve.values[0]) / tot[0]))


def gc_analise(on: pd.DataFrame, off: pd.DataFrame) -> pd.DataFrame:
    a = _mediana(on)[["algo", "n", "time_total", "gc_collections", "mem_peak_total"]]
    b = _mediana(off)[["algo", "n", "time_total", "mem_peak_total"]]
    m = a.merge(b, on=["algo", "n"], suffixes=("_gc", "_sem_gc"))
    m["razao_tempo"] = m.time_total_gc / m.time_total_sem_gc
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for algo in SECOES:
        d = m[m.algo == algo]
        if d.empty:
            continue
        ax.semilogx(d.n, d.razao_tempo, "o-", ms=3, lw=1, label=NOME[algo])
    ax.axhline(1, color="k", lw=0.6)
    ax.set_xlabel("N (vértices)")
    ax.set_ylabel("tempo com GC / sem GC")
    ax.legend(frameon=False, fontsize=6)
    salva_fig(fig, "gc_razao")
    # expoentes com e sem GC (o GC muda a curva?)
    linhas = []
    for algo in SECOES:
        for nome, df in (("gc", on), ("sem_gc", off)):
            d = _mediana(df)
            d = d[d.algo == algo]
            if d.empty:
                continue
            pw = ajuste_potencia(d.n, d.time_total)
            linhas.append({"algo": algo, "gc": nome, "expoente": pw["b"], "ic95": pw["b_ic"],
                           "r2_log": pw["r2_log"]})
    exp = pd.DataFrame(linhas).pivot(index="algo", columns="gc", values="expoente")
    exp.to_csv(ANALISE / "gc_expoentes.csv")
    m.to_csv(ANALISE / "gc_comparacao.csv", index=False)
    chr_ = m[m.algo == "christofides"]
    if not chr_.empty:
        numero("gc_razao_chr_nmax", float(chr_.razao_tempo.values[-1]))
        numero("gc_colecoes_chr_nmax", float(chr_.gc_collections.values[-1]))
    numero("gc_razao_max", float(m.razao_tempo.max()))
    numero("gc_razao_mediana", float(m.razao_tempo.median()))
    return exp


def bnb_escala() -> pd.DataFrame:
    """Crescimento do B&B em instâncias aleatórias: exponencial x polinomial."""
    from scipy import stats
    linhas, dados = [], {}
    for bound in ("hk", "onetree"):
        arqs = sorted(RES.glob(f"scaling_bnb_{bound}*.csv"))
        if not arqs:
            continue
        d = pd.concat([pd.read_csv(a) for a in arqs], ignore_index=True)
        dados[bound] = d
        ok = d[d.status == "optimal"]
        med = ok.groupby("n").agg(nos=("nodes_expanded", "median"),
                                  tempo=("time_total", "median"),
                                  resolvidas=("status", "size")).reset_index()
        tot = d.groupby("n").size()
        med["total"] = med.n.map(tot)
        y = np.log10(np.maximum(med.nos.values, 1))
        ex = stats.linregress(med.n, y)                       # log N_nós = a + b n
        po = stats.linregress(np.log10(med.n), y)              # log N_nós = a + b log n
        tpn = ok[ok.nodes_expanded > 0]
        tpn = tpn.assign(tpn=tpn.time_child_bounds / tpn.nodes_expanded)
        tp = ajuste_potencia(tpn.n, tpn.tpn)
        mem = (ajuste_potencia(d.n, d.mem_peak_bnb) if "mem_peak_bnb" in d
               else {"b": np.nan, "r2_log": np.nan})
        linhas.append({"limite": bound, "n_min": int(d.n.min()), "n_max": int(d.n.max()),
                       "sementes": int(d.seed.nunique()),
                       "timeouts": int((d.status != "optimal").sum()),
                       "fator_por_vertice": 10 ** ex.slope, "r2_exponencial": ex.rvalue ** 2,
                       "expoente_polinomial": po.slope, "r2_polinomial": po.rvalue ** 2,
                       "expoente_tempo_por_no": tp["b"], "r2_tempo_por_no": tp["r2_log"],
                       "expoente_memoria": mem["b"], "r2_memoria": mem["r2_log"],
                       "mem_kib_nmax": float(d[d.n == d.n.max()].mem_peak_bnb.median() / 1024)
                       if "mem_peak_bnb" in d else np.nan})
        med.to_csv(ANALISE / f"bnb_escala_{bound}.csv", index=False)
        numero(f"bnb_fator_{bound}", float(10 ** ex.slope))
        numero(f"bnb_r2exp_{bound}", float(ex.rvalue ** 2))
        numero(f"bnb_r2pol_{bound}", float(po.rvalue ** 2))
        numero(f"bnb_expo_tpn_{bound}", float(tp["b"]))
        numero(f"bnb_expo_mem_{bound}", float(mem["b"]))
    if dados:
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
        for bound, d in dados.items():
            cor = "#1f77b4" if bound == "hk" else "#d62728"
            rot = "Held-Karp" if bound == "hk" else "1-árvore"
            ok = d[d.status == "optimal"]
            axes[0].semilogy(ok.n, np.maximum(ok.nodes_expanded, 1), ".", color=cor, alpha=0.35, ms=4)
            med = ok.groupby("n").nodes_expanded.median()
            axes[0].semilogy(med.index, np.maximum(med.values, 1), "-", color=cor, label=rot)
            to = d[d.status != "optimal"]
            if len(to):
                axes[0].semilogy(to.n, np.maximum(to.nodes_expanded, 1), "x", color=cor, ms=4)
            axes[1].semilogy(ok.n, ok.time_total, ".", color=cor, alpha=0.35, ms=4)
            axes[1].semilogy(ok.groupby("n").time_total.median(), "-", color=cor, label=rot)
        axes[0].set_xlabel("N")
        axes[0].set_ylabel("nós expandidos (×: timeout)")
        axes[0].legend(frameon=False)
        axes[1].set_xlabel("N")
        axes[1].set_ylabel("tempo (s)")
        axes[0].set_title("(a) tamanho da árvore")
        axes[1].set_title("(b) tempo")
        salva_fig(fig, "bnb_escala")
    return pd.DataFrame(linhas)


def tabelas_latex(tt: pd.DataFrame, tm: pd.DataFrame, bnb: pd.DataFrame, gc_exp: pd.DataFrame):
    t = tt.copy()
    t = pd.DataFrame({
        "Algoritmo": t.algo.map(NOME), "Subrotina": t.secao.map(SEC_NOME),
        "Teoria": t.teoria.map(lambda m: MODELO_TEX.get(m, "--") if isinstance(m, str) else "--"),
        "$b$ (IC 95\\%)": [f"{fmt_num(b)} $\\pm$ {fmt_num(c)}" for b, c in zip(t.expoente, t.expoente_ic95)],
        "$R^2_{\\log}$": t.r2_log_potencia.map(lambda v: fmt_num(v, 3)),
        "Melhor forma": t.melhor_forma.map(MODELO_TEX),
        "$R^2$ teoria": t.r2_teoria.map(lambda v: fmt_num(v, 3)),
        "RMSE (ms)": t.rmse_teoria_ms.map(lambda v: fmt_num(v, 3)),
    })
    tabela_latex(t, "complexidade_tempo",
                 "Ajuste do tempo em função de $N$ (medianas de 3 sementes, $N=25$ a $6400$; "
                 "blossom até $1000$). "
                 "$b$: expoente da lei de potência $t=aN^b$ (log-log). Melhor forma: modelo "
                 "teórico com menor RMSE em escala log. $R^2$/RMSE teoria: ajuste "
                 "$t=c_0+c_1f(N)$ do modelo teórico em escala linear.",
                 "tab:complexidade", "llcccccc")
    m = tm.copy()
    m = pd.DataFrame({
        "Algoritmo": m.algo.map(NOME), "Subrotina": m.secao.map(SEC_NOME),
        "Teoria": m.teoria.map(lambda x: MODELO_TEX.get(x, "--") if isinstance(x, str) else "--"),
        "$b$ (IC 95\\%)": [f"{fmt_num(b)} $\\pm$ {fmt_num(c)}" for b, c in zip(m.expoente, m.expoente_ic95)],
        "$R^2_{\\log}$": m.r2_log_potencia.map(lambda v: fmt_num(v, 3)),
        "Pico em $N_{\\max}$ (KiB)": m.mem_max_kib.map(lambda v: fmt_num(v, 1)),
    })
    tabela_latex(m, "complexidade_memoria",
                 "Expoente do pico de memória extra (tracemalloc) em função de $N$.",
                 "tab:memoria", "llcccr")
    if not bnb.empty:
        b = pd.DataFrame({
            "Limite": bnb.limite.map({"hk": "Held-Karp", "onetree": "1-árvore"}),
            "$N$": [f"{a}--{c}" for a, c in zip(bnb.n_min, bnb.n_max)],
            "Timeouts": bnb.timeouts,
            "Fator/vértice": bnb.fator_por_vertice.map(lambda v: fmt_num(v, 2)),
            "$R^2$ exp.": bnb.r2_exponencial.map(lambda v: fmt_num(v, 3)),
            "$R^2$ polin.": bnb.r2_polinomial.map(lambda v: fmt_num(v, 3)),
            "$b$ tempo/nó": bnb.expoente_tempo_por_no.map(lambda v: fmt_num(v, 2)),
            "$b$ memória": bnb.expoente_memoria.map(lambda v: fmt_num(v, 2)),
        })
        tabela_latex(b, "bnb_escala",
                     "Crescimento do B\\&B (DFS, filhos por limite, warm-start vizinho mais "
                     "próximo) em instâncias uniformes. Fator/vértice: $10^{\\beta}$ do ajuste "
                     "$\\log_{10}(\\text{nós}) = \\alpha + \\beta N$. $b$ tempo/nó: expoente "
                     "do tempo de cálculo dos limites por nó expandido; $b$ memória: expoente "
                     "do pico de memória (tracemalloc) em função de $N$.",
                     "tab:bnbescala", "lcrccccc")
    if gc_exp is not None and not gc_exp.empty:
        g = gc_exp.reset_index()
        g = pd.DataFrame({"Algoritmo": g.algo.map(NOME),
                          "$b$ com GC": g["gc"].map(fmt_num),
                          "$b$ sem GC": g["sem_gc"].map(fmt_num)})
        tabela_latex(g, "gc_expoentes", "Expoente do tempo total com e sem o coletor de lixo.",
                     "tab:gc", "lcc")


def roda():
    on = pd.read_csv(RES / "scaling.csv")
    tt = tabela_tempo(on)
    tm = tabela_memoria(on)
    tt.to_csv(ANALISE / "complexidade_tempo.csv", index=False)
    tm.to_csv(ANALISE / "complexidade_memoria.csv", index=False)
    fig_tempo(on)
    fig_memoria(on)
    fig_participacao(on)
    gc_exp = None
    if (RES / "scaling_nogc.csv").exists():
        gc_exp = gc_analise(on, pd.read_csv(RES / "scaling_nogc.csv"))
    bnb = bnb_escala()
    bnb.to_csv(ANALISE / "bnb_escala_ajuste.csv", index=False)
    tabelas_latex(tt, tm, bnb, gc_exp)
    for _, r in tt.iterrows():
        numero(f"expo_{r.algo}_{r.secao}", float(r.expoente))
    return tt, tm, bnb
