"""Seção 2 do enunciado: impacto do warm-start na árvore do Branch-and-Bound.

Entrada: results/runs.jsonl (300 execuções de B&B na TSPLIB: 15 instâncias x
5 warm-starts x 4 configurações de limite/ordem dos filhos, DFS, 30 min).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from .comum import (ANALISE, CORES, RES, START_ROTULO, STARTS, fmt_num, numero, plt,
                    salva_fig, tabela_latex)

LIMITE = 1800.0
CONFIG_ROTULO = {("hk", "bound"): "Held-Karp, filhos por LB",
                 ("hk", "index"): "Held-Karp, filhos por índice",
                 ("onetree", "bound"): "1-árvore, filhos por LB",
                 ("onetree", "index"): "1-árvore, filhos por índice"}
CONFIGS = list(CONFIG_ROTULO)


def carrega() -> pd.DataFrame:
    rows = [json.loads(line) for line in (RES / "runs.jsonl").open() if line.strip()]
    b = pd.DataFrame([r for r in rows if r["algo"] == "bnb" and r.get("gc_enabled", True)
                      and r.get("search", "dfs") == "dfs"])
    b["initial_gap"] = (b.initial_ub - b.known_opt) / b.known_opt
    b["root_gap"] = (b.known_opt - b.root_lb) / b.known_opt
    b["frac_podada"] = b.nodes_pruned / b.nodes_generated
    b["resolvido"] = b.status == "optimal"
    b["tempo_censurado"] = np.where(b.resolvido, b.time_total, LIMITE)
    return b


def resumo(b: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for (bound, order) in CONFIGS:
        d = b[(b.bound == bound) & (b.child_order == order)]
        # conjunto comum: instâncias que TODAS as estratégias resolveram
        comuns = d.groupby("instance").resolvido.all()
        comuns = comuns[comuns].index
        triv = d[d.start == "trivial"].set_index("instance")
        for s in STARTS:
            e = d[d.start == s].set_index("instance")
            ec = e.loc[comuns]
            razao = (ec.nodes_expanded.clip(lower=1) / triv.loc[comuns].nodes_expanded.clip(lower=1))
            linhas.append({
                "bound": bound, "child_order": order, "start": s,
                "resolvidas": int(e.resolvido.sum()), "instancias_comuns": len(comuns),
                "gap_inicial_medio": float(e.initial_gap.mean()),
                "nos_expandidos_media_geo": float(np.exp(np.log(ec.nodes_expanded.clip(lower=1)).mean())) if len(ec) else np.nan,
                "nos_podados_media_geo": float(np.exp(np.log(ec.nodes_pruned.clip(lower=1)).mean())) if len(ec) else np.nan,
                "razao_nos_vs_trivial_geo": float(np.exp(np.log(razao).mean())) if len(ec) else np.nan,
                "frac_podada_mediana": float(e.frac_podada.median()),
                "tempo_mediano_s": float(e.tempo_censurado.median()),
                "tempo_total_s": float(e.tempo_censurado.sum()),
                "tempo_heuristica_ms_mediano": float(e.time_heuristic.median() * 1e3),
            })
    return pd.DataFrame(linhas)


def friedman(b: pd.DataFrame) -> pd.DataFrame:
    """Teste de Friedman (blocos = instâncias) sobre o tempo (timeout = 1800 s,
    o teste usa só postos, então a censura não atrapalha) e sobre os nós
    expandidos no conjunto comum. Também o posto médio de cada estratégia."""
    linhas = []
    for (bound, order) in CONFIGS:
        d = b[(b.bound == bound) & (b.child_order == order)]
        t = d.pivot(index="instance", columns="start", values="tempo_censurado")[STARTS]
        st, p = stats.friedmanchisquare(*[t[c] for c in STARTS])
        postos = t.rank(axis=1).mean()
        com = d.groupby("instance").resolvido.all()
        e = d[d.instance.isin(com[com].index)].pivot(index="instance", columns="start",
                                                     values="nodes_expanded")[STARTS]
        if len(e) >= 3 and e.nunique(axis=1).max() > 1:
            st_n, p_n = stats.friedmanchisquare(*[e[c] for c in STARTS])
        else:
            st_n, p_n = np.nan, np.nan
        linha = {"bound": bound, "child_order": order, "friedman_tempo": st, "p_tempo": p,
                 "friedman_nos": st_n, "p_nos": p_n}
        linha.update({f"posto_{s}": postos[s] for s in STARTS})
        linhas.append(linha)
    return pd.DataFrame(linhas)


def tradeoff(b: pd.DataFrame) -> pd.DataFrame:
    """O tempo da heurística compensa? Economia = tempo do B&B com a solução
    trivial - tempo com a heurística (ambos incluindo a heurística)."""
    linhas = []
    for (bound, order) in CONFIGS:
        d = b[(b.bound == bound) & (b.child_order == order)]
        triv = d[d.start == "trivial"].set_index("instance").sort_index()
        for s in STARTS[1:]:
            e = d[d.start == s].set_index("instance").sort_index()
            eco = triv.tempo_censurado - e.tempo_censurado
            bnb_triv = triv.tempo_censurado - triv.time_heuristic
            bnb_s = e.tempo_censurado - e.time_heuristic
            linhas.append({
                "bound": bound, "child_order": order, "start": s,
                "compensou": int((eco > 0).sum()), "nao_compensou": int((eco <= 0).sum()),
                "economia_mediana_s": float(eco.median()),
                "economia_total_s": float(eco.sum()),
                "heuristica_total_s": float(e.time_heuristic.sum()),
                "busca_mais_rapida": int((bnb_s < bnb_triv).sum()),
            })
    return pd.DataFrame(linhas)


def correlacoes(b: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for (bound, order) in CONFIGS:
        d = b[(b.bound == bound) & (b.child_order == order) & b.resolvido]
        # dentro de cada instância: gap inicial x nós (tira o efeito do tamanho)
        rhos = []
        for _, g in d.groupby("instance"):
            if g.initial_gap.nunique() > 2 and g.nodes_expanded.nunique() > 1:
                rhos.append(stats.spearmanr(g.initial_gap, g.nodes_expanded)[0])
        linhas.append({"bound": bound, "child_order": order,
                       "rho_medio_dentro_instancia": float(np.nanmean(rhos)) if rhos else np.nan,
                       "instancias": len(rhos)})
    return pd.DataFrame(linhas)


def fig_heatmap(b: pd.DataFrame):
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 3.3), sharey=True)
    inst = (b[["instance", "n"]].drop_duplicates().sort_values(["n", "instance"]).instance.tolist())
    vmax = np.log10(b.nodes_expanded.clip(lower=1).max())
    for ax, (bound, order) in zip(axes, CONFIGS):
        d = b[(b.bound == bound) & (b.child_order == order)]
        M = d.pivot(index="instance", columns="start", values="nodes_expanded").loc[inst, STARTS]
        S = d.pivot(index="instance", columns="start", values="resolvido").loc[inst, STARTS]
        im = ax.imshow(np.log10(M.clip(lower=1).values.astype(float)), aspect="auto",
                       cmap="viridis", vmin=0, vmax=vmax)
        for i in range(len(inst)):
            for j in range(len(STARTS)):
                if not S.values[i, j]:
                    ax.text(j, i, "T", ha="center", va="center", color="w", fontsize=6,
                            fontweight="bold")
        ax.set_xticks(range(len(STARTS)), [START_ROTULO[s] for s in STARTS], rotation=60,
                      ha="right", fontsize=6)
        ax.set_title(CONFIG_ROTULO[(bound, order)].replace(", ", ",\n"), fontsize=6.5)
    axes[0].set_yticks(range(len(inst)), inst, fontsize=6)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("$\\log_{10}$ nós expandidos (T = timeout)", fontsize=6.5)
    salva_fig(fig, "warmstart_heatmap")


def fig_profundidade(b: pd.DataFrame, instance: str, bound: str, order: str, nome: str):
    d = b[(b.instance == instance) & (b.bound == bound) & (b.child_order == order)]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.3), sharex=True)
    for _, r in d.iterrows():
        e = np.array(r.expanded_by_depth, float)
        p = np.array(r.pruned_by_depth, float)
        k = np.arange(len(e))
        rot = START_ROTULO[r.start] + ("" if r.resolvido else " (T)")
        axes[0].semilogy(k, np.where(e > 0, e, np.nan), "-", color=CORES[r.start], lw=1, label=rot)
        axes[1].semilogy(k, np.where(p > 0, p, np.nan), "-", color=CORES[r.start], lw=1)
    axes[0].set_ylabel("nós expandidos")
    axes[1].set_ylabel("nós podados")
    for ax in axes:
        ax.set_xlabel("profundidade")
    axes[0].legend(frameon=False, fontsize=6)
    fig.suptitle(f"{instance}: {CONFIG_ROTULO[(bound, order)]}", fontsize=7.5)
    salva_fig(fig, nome)


def fig_gap_vs_nos(b: pd.DataFrame):
    d = b[(b.bound == "hk") & (b.child_order == "index")]
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for s in STARTS:
        e = d[d.start == s]
        ax.scatter(e.initial_gap.clip(lower=1e-3), e.nodes_expanded.clip(lower=1),
                   s=10, color=CORES[s], label=START_ROTULO[s],
                   marker="o" if s != "trivial" else "s", alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("gap do limite superior inicial")
    ax.set_ylabel("nós expandidos")
    ax.set_title("Held-Karp, filhos por índice", fontsize=7.5)
    ax.legend(frameon=False, fontsize=5.5)
    salva_fig(fig, "warmstart_gap_vs_nos")


def colapso(b: pd.DataFrame) -> pd.DataFrame:
    """Maior N resolvido por configuração e onde o B&B começa a estourar 30 min."""
    linhas = []
    for (bound, order) in CONFIGS:
        d = b[(b.bound == bound) & (b.child_order == order)]
        melhor = d.groupby(["instance", "n"]).resolvido.any().reset_index()
        algum = d.groupby(["instance", "n"]).resolvido.all().reset_index()
        linhas.append({
            "bound": bound, "child_order": order,
            "resolvidas_por_alguma_estrategia": int(melhor.resolvido.sum()),
            "resolvidas_por_todas": int(algum.resolvido.sum()),
            "maior_n_resolvido": int(melhor[melhor.resolvido].n.max()),
            "menor_n_com_timeout": int(d[~d.resolvido].n.min()) if (~d.resolvido).any() else None,
        })
    return pd.DataFrame(linhas)


def perfil_tempo(b: pd.DataFrame) -> pd.DataFrame:
    """Onde o B&B gasta o tempo (profiling por subrotina)."""
    g = b.assign(f_heur=b.time_heuristic / b.time_total,
                 f_raiz=b.time_root_bound / b.time_total,
                 f_filhos=b.time_child_bounds / b.time_total,
                 us_por_no=b.time_child_bounds / b.nodes_expanded.clip(lower=1) * 1e6)
    return g.groupby(["bound", "child_order"]).agg(
        frac_heuristica=("f_heur", "median"), frac_limite_raiz=("f_raiz", "median"),
        frac_limites_filhos=("f_filhos", "median"),
        us_por_no_mediano=("us_por_no", "median")).reset_index()


def tabela_perfil(pt: pd.DataFrame):
    t = pd.DataFrame({
        "Configuração": pt.apply(lambda r: CONFIG_ROTULO[(r.bound, r.child_order)], axis=1),
        "Heurística": pt.frac_heuristica.map(lambda v: fmt_num(100 * v, 2) + "\\%"),
        "Limite na raiz": pt.frac_limite_raiz.map(lambda v: fmt_num(100 * v, 1) + "\\%"),
        "Limites dos filhos": pt.frac_limites_filhos.map(lambda v: fmt_num(100 * v, 1) + "\\%"),
        "$\\mu$s por nó": pt.us_por_no_mediano.map(lambda v: fmt_num(v, 0)),
    })
    tabela_latex(t, "bnb_perfil",
                 "Profiling do B\\&B: mediana, nas 75 execuções de cada configuração, da "
                 "fração do tempo total em cada subrotina, e tempo de cálculo dos limites "
                 "por nó expandido. O restante é a própria busca (recursão, ordenação).",
                 "tab:perfil", "lrrrr")


def tabelas_latex(res: pd.DataFrame, fr: pd.DataFrame, to: pd.DataFrame, col: pd.DataFrame):
    linhas = []
    for (bound, order) in CONFIGS:
        d = res[(res.bound == bound) & (res.child_order == order)]
        f = fr[(fr.bound == bound) & (fr.child_order == order)].iloc[0]
        for i, (_, r) in enumerate(d.iterrows()):
            linhas.append({
                "Configuração": (CONFIG_ROTULO[(bound, order)] if i == 0 else ""),
                "Warm-start": START_ROTULO[r.start],
                "Resolv.": f"{r.resolvidas}/15",
                "Gap inicial": fmt_num(100 * r.gap_inicial_medio, 1) + "\\%",
                "Nós exp. (geo)": fmt_num(r.nos_expandidos_media_geo, 0),
                "Razão vs trivial": fmt_num(r.razao_nos_vs_trivial_geo, 3),
                "Posto tempo": fmt_num(f[f"posto_{r.start}"], 2),
                "Tempo total (s)": fmt_num(r.tempo_total_s, 0),
            })
        linhas.append({k: ("\\midrule" if k == "Configuração" else "") for k in linhas[-1]})
    t = pd.DataFrame(linhas[:-1])
    # \midrule como linha separadora
    tex_rows = []
    for _, r in t.iterrows():
        if r["Configuração"] == "\\midrule":
            tex_rows.append(None)
        else:
            tex_rows.append(r)
    _tabela_com_regras(tex_rows, list(t.columns), "warmstart_resumo",
                       "Impacto do warm-start no B\\&B (DFS, 15 instâncias, limite de 30 min). "
                       "Nós exp. (geo): média geométrica dos nós expandidos nas instâncias "
                       "que todas as estratégias resolveram; razão vs trivial: média geométrica "
                       "da razão por instância. Posto tempo: posto médio do tempo total "
                       "(timeout = 1800 s; 1 = mais rápido). Tempo total: soma das 15 execuções.",
                       "tab:warmstart", "llcrrrrr")
    f = pd.DataFrame({
        "Configuração": fr.apply(lambda r: CONFIG_ROTULO[(r.bound, r.child_order)], axis=1),
        "$\\chi^2_F$ tempo": fr.friedman_tempo.map(lambda v: fmt_num(v, 1)),
        "$p$ tempo": fr.p_tempo.map(lambda v: f"{v:.1e}".replace(".", "{,}")),
        "$\\chi^2_F$ nós": fr.friedman_nos.map(lambda v: fmt_num(v, 1)),
        "$p$ nós": fr.p_nos.map(lambda v: f"{v:.1e}".replace(".", "{,}") if np.isfinite(v) else "--"),
    })
    tabela_latex(f, "warmstart_friedman",
                 "Teste de Friedman (4 graus de liberdade) para a hipótese de que as cinco "
                 "estratégias de warm-start têm o mesmo desempenho.", "tab:friedman", "lcccc")
    c = pd.DataFrame({
        "Configuração": col.apply(lambda r: CONFIG_ROTULO[(r.bound, r.child_order)], axis=1),
        "Resolvidas (alguma)": col.resolvidas_por_alguma_estrategia.map(lambda v: f"{v}/15"),
        "Resolvidas (todas)": col.resolvidas_por_todas.map(lambda v: f"{v}/15"),
        "Maior $N$ resolvido": col.maior_n_resolvido,
        "Menor $N$ com timeout": col.menor_n_com_timeout.map(lambda v: "--" if pd.isna(v) else int(v)),
    })
    tabela_latex(c, "colapso", "Ponto de colapso do B\\&B exato na TSPLIB (30 min).",
                 "tab:colapso", "lcccc")


def _tabela_com_regras(rows, cols, nome, legenda, rotulo, alinhamento):
    from .comum import TAB
    linhas = [r"\begin{table}[htb]", r"\centering", r"\caption{" + legenda + "}",
              r"\label{" + rotulo + "}", r"\small", r"\setlength{\tabcolsep}{4pt}",
              r"\begin{adjustbox}{max width=\linewidth}",
              r"\begin{tabular}{" + alinhamento + "}", r"\toprule",
              " & ".join(cols) + r" \\", r"\midrule"]
    for r in rows:
        linhas.append(r"\midrule" if r is None else " & ".join(map(str, r.values)) + r" \\")
    linhas += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    (TAB / f"{nome}.tex").write_text("\n".join(linhas) + "\n")


def gc_bnb() -> pd.DataFrame:
    """B&B Held-Karp (filhos por LB) com e sem GC, mesmas condições (4 processos)."""
    rows = [json.loads(line) for line in (RES / "runs.jsonl").open() if line.strip()]
    r = pd.DataFrame([x for x in rows if x["algo"] == "bnb" and x["bound"] == "hk"
                      and x["child_order"] == "bound"])
    if r.empty or r.gc_enabled.all():
        return pd.DataFrame()
    on = r[r.gc_enabled].set_index(["instance", "start"])
    off = r[~r.gc_enabled].set_index(["instance", "start"])
    m = on[["n", "nodes_expanded", "time_total", "cpu_time_total", "gc_collections"]].join(
        off[["nodes_expanded", "time_total", "cpu_time_total"]], rsuffix="_sem_gc", how="inner")
    m["razao_cpu"] = m.cpu_time_total / m.cpu_time_total_sem_gc
    m.reset_index().to_csv(ANALISE / "bnb_gc_comparacao.csv", index=False)
    g = m[m.time_total > 1.0]                 # execuções curtas: só ruído
    numero("bnb_gc_razao_cpu_mediana", float(g.razao_cpu.median()) if len(g) else float("nan"))
    numero("bnb_gc_colecoes_mediana", float(m.gc_collections.median()))
    numero("bnb_gc_colecoes_max", float(m.gc_collections.max()))
    return m


def roda():
    gc_bnb()
    b = carrega()
    res = resumo(b)
    fr = friedman(b)
    to = tradeoff(b)
    co = correlacoes(b)
    col = colapso(b)
    pt = perfil_tempo(b)
    for df, nome in ((res, "warmstart_resumo"), (fr, "warmstart_friedman"),
                     (to, "warmstart_tradeoff"), (co, "warmstart_correlacao"),
                     (col, "bnb_colapso"), (pt, "bnb_perfil_tempo")):
        df.to_csv(ANALISE / f"{nome}.csv", index=False)
    fig_heatmap(b)
    fig_gap_vs_nos(b)
    fig_profundidade(b, "bayg29", "hk", "index", "warmstart_profundidade_bayg29")
    fig_profundidade(b, "gr24", "onetree", "index", "warmstart_profundidade_gr24")
    tabelas_latex(res, fr, to, col)
    tabela_perfil(pt)
    # números para o texto
    for _, r in res.iterrows():
        numero(f"ws_{r.bound}_{r.child_order}_{r.start}_razao", r.razao_nos_vs_trivial_geo)
        numero(f"ws_{r.bound}_{r.child_order}_{r.start}_resolvidas", r.resolvidas)
    for _, r in fr.iterrows():
        numero(f"friedman_{r.bound}_{r.child_order}_p", r.p_tempo)
    for _, r in to.iterrows():
        numero(f"tradeoff_{r.bound}_{r.child_order}_{r.start}_compensou", r.compensou)
    for _, r in pt.iterrows():
        numero(f"perfil_{r.bound}_{r.child_order}_frac_filhos", r.frac_limites_filhos)
    numero("heur_ms_mediano_christofides", float(b[b.start == "christofides"].time_heuristic.median() * 1e3))
    numero("heur_ms_max", float(b.time_heuristic.max() * 1e3))
    return res, fr, to, co, col, pt
