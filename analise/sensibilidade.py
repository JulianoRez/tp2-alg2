"""Seção 3 do enunciado: sensibilidade do gap às características das instâncias.

Base: 15 instâncias da TSPLIB + instâncias sintéticas adversariais
(results/synthetic.csv, tsp/synthetic.py), todas com ótimo exato conhecido.
Modelos: correlação de Spearman, regressão linear múltipla com variáveis
padronizadas (coeficientes comparáveis = "peso" de cada feature), árvore de
decisão (regras legíveis) e floresta aleatória (importância por permutação,
captura relações não lineares). Desempenho medido por validação cruzada.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

from .comum import ANALISE, RES, fmt_num, numero, plt, salva_fig, tabela_latex

FEATURES = ["n", "aspect_ratio", "hull_fraction", "spatial_entropy", "clark_evans",
            "silhouette_best", "nn_dist_cv", "dist_cv", "mst_max_degree",
            "mst_leaf_fraction", "mst_odd_fraction", "mst_edge_cv",
            "triangle_violation_frac"]
FEAT_ROTULO = {"n": "$N$", "aspect_ratio": "razão de aspecto", "hull_fraction": "fração no fecho",
               "spatial_entropy": "entropia espacial", "clark_evans": "Clark-Evans",
               "silhouette_best": "silhueta (k-means)", "nn_dist_cv": "CV dist. viz. próximo",
               "dist_cv": "CV distâncias", "mst_max_degree": "grau máx. MST",
               "mst_leaf_fraction": "fração folhas MST", "mst_odd_fraction": "fração ímpares MST",
               "mst_edge_cv": "CV arestas MST", "triangle_violation_frac": "violação triangular"}
ALVOS = ["gap_nn", "gap_hull", "gap_tatt", "gap_christofides_blossom",
         "gap_christofides_greedy", "root_gap_onetree"]
ALVO_ROTULO = {"gap_nn": "Viz. próximo", "gap_hull": "Fecho convexo", "gap_tatt": "TATT",
               "gap_christofides_blossom": "Christofides", "gap_christofides_greedy": "Christ. guloso",
               "root_gap_onetree": "gap raiz 1-árvore", "root_gap_hk": "gap raiz Held-Karp",
               "log_prova_1arvore": "$\\log_{10}$ nós p/ provar (1-árvore)"}
FAMILIA_ROTULO = {"uniform": "uniforme", "cluster": "aglomerados", "grid": "grade",
                  "circle": "anel", "strip": "faixa", "core_outliers": "núcleo+dispersos",
                  "spokes": "raios", "tsplib": "TSPLIB"}
CV = RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)


def carrega() -> pd.DataFrame:
    sy = pd.read_csv(RES / "synthetic.csv")
    sy = sy[sy.exact_status == "optimal"].copy()
    sy["log_prova_1arvore"] = np.where(
        sy.get("onetree_prove_status", pd.Series(index=sy.index)).eq("optimal"),
        np.log10(sy.onetree_prove_nodes.fillna(0) + 1), np.nan)
    # TSPLIB: features + gaps das heurísticas + gap do limite na raiz
    f = pd.read_csv(RES / "features.csv")
    runs = [json.loads(line) for line in (RES / "runs.jsonl").open() if line.strip()]
    r = pd.DataFrame(runs)
    h = r[r.algo != "bnb"].copy()
    h["alg"] = np.where(h.algo == "christofides", "christofides_" + h.matching.fillna(""), h.algo)
    gaps = h.pivot_table(index="instance", columns="alg", values="gap", aggfunc="first")
    gaps.columns = [f"gap_{c}" for c in gaps.columns]
    b = r[r.algo == "bnb"]
    rg = b.groupby(["instance", "bound"]).root_lb.first().unstack()
    for bound in rg.columns:
        f[f"root_gap_{bound}"] = f.instance.map(
            (f.set_index("instance").known_opt - rg[bound]) / f.set_index("instance").known_opt)
    ts = f.merge(gaps, left_on="instance", right_index=True)
    ts["family"] = "tsplib"
    ts["opt"] = ts.known_opt
    cols = ["instance", "family", "opt"] + FEATURES + ALVOS + ["root_gap_hk"]
    df = pd.concat([sy[cols + ["log_prova_1arvore"]], ts[cols]], ignore_index=True)
    return df


def vif(X: pd.DataFrame) -> pd.Series:
    Z = (X - X.mean()) / X.std(ddof=0)
    out = {}
    for c in Z.columns:
        others = Z.drop(columns=c)
        r2 = LinearRegression().fit(others, Z[c]).score(others, Z[c])
        out[c] = 1 / max(1 - r2, 1e-12)
    return pd.Series(out)


def seleciona_vif(X: pd.DataFrame, limite: float = 10.0) -> tuple[list[str], list[str]]:
    """Remove iterativamente a feature com maior VIF até todas ficarem < limite."""
    cols, removidas = list(X.columns), []
    while True:
        v = vif(X[cols])
        if v.max() < limite:
            return cols, removidas
        pior = v.idxmax()
        cols.remove(pior)
        removidas.append(pior)


def ols(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """MQO com variáveis padronizadas: coeficiente, erro padrão, t, p."""
    Z = (X - X.mean()) / X.std(ddof=0)
    A = np.column_stack([np.ones(len(Z)), Z.values])
    beta, *_ = np.linalg.lstsq(A, y.values, rcond=None)
    res = y.values - A @ beta
    dof = len(y) - A.shape[1]
    s2 = res @ res / dof
    cov = s2 * np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    return pd.DataFrame({"coef": beta[1:], "se": se[1:], "t": t[1:], "p": p[1:]},
                        index=X.columns)


def modelos(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    resumo, coefs, arvores = [], [], {}
    for alvo in ALVOS + ["log_prova_1arvore"]:
        d = df.dropna(subset=[alvo] + cols)
        if len(d) < 20:
            continue
        usadas = [c for c in cols if d[c].nunique() > 1]   # ex.: N é constante em N = 20
        X, y = d[usadas], d[alvo] * (100 if alvo.startswith(("gap", "root")) else 1)
        Xs = StandardScaler().fit_transform(X)
        lin = LinearRegression().fit(Xs, y)
        cv_lin = cross_val_score(LinearRegression(), Xs, y, cv=CV, scoring="r2").mean()
        tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=10, random_state=0)
        cv_tree = cross_val_score(tree, X, y, cv=CV, scoring="r2").mean()
        tree.fit(X, y)
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=4)
        cv_rf = cross_val_score(rf, X, y, cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=0),
                                scoring="r2", n_jobs=1).mean()
        rf.fit(X, y)
        pi = permutation_importance(rf, X, y, n_repeats=20, random_state=0, n_jobs=4)
        imp = pd.Series(pi.importances_mean, index=usadas).sort_values(ascending=False)
        o = ols(X, y)
        o["alvo"] = alvo
        coefs.append(o.reset_index().rename(columns={"index": "feature"}))
        top_lin = o.coef.abs().sort_values(ascending=False).index[:3]
        resumo.append({
            "alvo": alvo, "instancias": len(d), "r2_treino_linear": lin.score(Xs, y),
            "r2_cv_linear": cv_lin, "r2_cv_arvore": cv_tree, "r2_cv_floresta": cv_rf,
            "top_linear": "; ".join(f"{c} ({o.coef[c]:+.2f})" for c in top_lin),
            "top_floresta": "; ".join(f"{c} ({imp[c]:.2f})" for c in imp.index[:3]),
            **{f"imp_{c}": imp[c] for c in usadas},
        })
        arvores[alvo] = (tree, export_text(tree, feature_names=usadas, decimals=3))
    return pd.DataFrame(resumo), pd.concat(coefs, ignore_index=True), arvores


def fig_spearman(df: pd.DataFrame) -> pd.DataFrame:
    alvos = ALVOS + ["root_gap_hk", "log_prova_1arvore"]
    rho = pd.DataFrame(index=FEATURES, columns=alvos, dtype=float)
    pv = rho.copy()
    for f in FEATURES:
        for a in alvos:
            d = df[[f, a]].dropna()
            if d[f].nunique() > 1 and len(d) > 5:
                rho.loc[f, a], pv.loc[f, a] = stats.spearmanr(d[f], d[a])
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    im = ax.imshow(rho.values.astype(float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(len(FEATURES)):
        for j in range(len(alvos)):
            v = rho.values[i, j]
            if np.isfinite(v):
                forte = pv.values[i, j] < 0.001
                ax.text(j, i, f"{v:.2f}".replace(".", ","), ha="center", va="center",
                        fontsize=5.5, fontweight="bold" if forte else "normal",
                        color="w" if abs(v) > 0.6 else "k")
    ax.set_xticks(range(len(alvos)), [ALVO_ROTULO[a] for a in alvos], rotation=35, ha="right")
    ax.set_yticks(range(len(FEATURES)), [FEAT_ROTULO[f] for f in FEATURES])
    fig.colorbar(im, ax=ax, shrink=0.8, label="$\\rho$ de Spearman (negrito: $p<0{,}001$)")
    salva_fig(fig, "sensibilidade_spearman")
    out = rho.add_prefix("rho_").join(pv.add_prefix("p_"))
    out.to_csv(ANALISE / "sensibilidade_spearman.csv")
    return out


def fig_familias(df: pd.DataFrame) -> pd.DataFrame:
    alvos = ["gap_nn", "gap_hull", "gap_tatt", "gap_christofides_blossom", "gap_christofides_greedy"]
    fams = [f for f in FAMILIA_ROTULO if f in set(df.family)]
    fig, axes = plt.subplots(1, len(alvos), figsize=(6.6, 2.6), sharey=True)
    for ax, a in zip(axes, alvos):
        dados = [100 * df[df.family == f][a].dropna().values for f in fams]
        ax.boxplot(dados, vert=False, widths=0.6, flierprops={"markersize": 2})
        ax.set_title(ALVO_ROTULO[a], fontsize=7)
        ax.set_xlabel("gap (%)")
    axes[0].set_yticks(range(1, len(fams) + 1), [FAMILIA_ROTULO[f] for f in fams])
    salva_fig(fig, "sensibilidade_familias")
    med = df.groupby("family")[alvos + ["root_gap_onetree", "root_gap_hk"]].median() * 100
    med["instancias"] = df.groupby("family").size()
    med = med.loc[fams]
    med.to_csv(ANALISE / "sensibilidade_familias.csv")
    return med


def fig_arvores(arvores: dict):
    for alvo, nome in (("gap_christofides_blossom", "arvore_christofides"),
                       ("gap_nn", "arvore_nn"), ("root_gap_onetree", "arvore_raiz_1arvore")):
        if alvo not in arvores:
            continue
        tree = arvores[alvo][0]
        fig, ax = plt.subplots(figsize=(6.6, 3.0))
        plot_tree(tree, feature_names=[FEAT_ROTULO[c].replace("$", "") for c in tree.feature_names_in_],
                  filled=True, rounded=True, fontsize=6, impurity=False, precision=2, ax=ax)
        ax.set_title(f"Árvore de decisão: gap (%) de {ALVO_ROTULO[alvo]}", fontsize=7.5)
        salva_fig(fig, nome)


def fig_dispersao(df: pd.DataFrame):
    """As relações mais fortes em gráfico (cor = família)."""
    pares = [("hull_fraction", "gap_nn"), ("mst_odd_fraction", "gap_christofides_blossom"),
             ("clark_evans", "root_gap_onetree")]
    fams = [f for f in FAMILIA_ROTULO if f in set(df.family)]
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3))
    for ax, (x, y) in zip(axes, pares):
        for i, f in enumerate(fams):
            d = df[df.family == f]
            ax.scatter(d[x], 100 * d[y], s=6, color=cmap(i), label=FAMILIA_ROTULO[f],
                       marker="*" if f == "tsplib" else "o", alpha=0.8)
        ax.set_xlabel(FEAT_ROTULO[x])
        ax.set_ylabel(f"{ALVO_ROTULO[y]} (%)" if not y.startswith("gap") else f"gap {ALVO_ROTULO[y]} (%)")
    axes[-1].legend(frameon=False, fontsize=5, loc="upper right", markerscale=1.5)
    salva_fig(fig, "sensibilidade_dispersao")


def tabelas_latex(res: pd.DataFrame, coefs: pd.DataFrame, med: pd.DataFrame, cols: list[str]):
    t = pd.DataFrame({
        "Alvo": res.alvo.map(ALVO_ROTULO), "Inst.": res.instancias,
        "$R^2$ treino (lin.)": res.r2_treino_linear.map(lambda v: fmt_num(v, 2)),
        "$R^2_{CV}$ linear": res.r2_cv_linear.map(lambda v: fmt_num(v, 2)),
        "$R^2_{CV}$ árvore": res.r2_cv_arvore.map(lambda v: fmt_num(v, 2)),
        "$R^2_{CV}$ floresta": res.r2_cv_floresta.map(lambda v: fmt_num(v, 2)),
    })
    tabela_latex(t, "sensibilidade_modelos",
                 "Poder preditivo das features (validação cruzada 5-fold repetida). "
                 "Linear: regressão múltipla padronizada; árvore: profundidade 3; "
                 "floresta: 300 árvores.", "tab:modelos", "lrcccc")
    c = coefs.pivot(index="feature", columns="alvo", values="coef").reindex(cols)
    p = coefs.pivot(index="feature", columns="alvo", values="p").reindex(cols)
    alvos = [a for a in ALVOS if a in c.columns]
    rows = []
    for f in cols:
        r = {"Feature": FEAT_ROTULO[f]}
        for a in alvos:
            v, pv = c.loc[f, a], p.loc[f, a]
            if not np.isfinite(v):
                r[ALVO_ROTULO[a]] = "--"
                continue
            estrela = "$^{***}$" if pv < 0.001 else "$^{**}$" if pv < 0.01 else "$^{*}$" if pv < 0.05 else ""
            r[ALVO_ROTULO[a]] = fmt_num(v, 2) + estrela
        rows.append(r)
    tabela_latex(pd.DataFrame(rows), "sensibilidade_coeficientes",
                 "Coeficientes padronizados da regressão múltipla (variação do gap, em "
                 "pontos percentuais, por desvio padrão da feature). "
                 "$^{*}p<0{,}05$, $^{**}p<0{,}01$, $^{***}p<0{,}001$.",
                 "tab:coeficientes", "l" + "r" * len(alvos))
    m = med.reset_index()
    m = pd.DataFrame({
        "Família": m.family.map(FAMILIA_ROTULO), "Inst.": m.instancias,
        **{ALVO_ROTULO[a]: m[a].map(lambda v: fmt_num(v, 1)) for a in
           ["gap_nn", "gap_hull", "gap_tatt", "gap_christofides_blossom",
            "gap_christofides_greedy", "root_gap_onetree"]},
    })
    tabela_latex(m, "sensibilidade_familias",
                 "Mediana do gap (\\%) por família de instância. A última coluna é o gap "
                 "do limite de 1-árvore na raiz do B\\&B.", "tab:familias", "lrrrrrrr")


def roda():
    df = carrega()
    df.to_csv(ANALISE / "sensibilidade_base.csv", index=False)
    rho = fig_spearman(df)
    med = fig_familias(df)
    cols, removidas = seleciona_vif(df[FEATURES].dropna())
    v = vif(df[FEATURES].dropna()).rename("vif").to_frame()
    v["usada"] = v.index.isin(cols)
    v.to_csv(ANALISE / "sensibilidade_vif.csv")
    res, coefs, arvores = modelos(df, cols)
    res.to_csv(ANALISE / "sensibilidade_modelos.csv", index=False)
    coefs.to_csv(ANALISE / "sensibilidade_coeficientes.csv", index=False)
    (ANALISE / "sensibilidade_arvores.txt").write_text(
        "\n\n".join(f"== {a} ==\n{txt}" for a, (_, txt) in arvores.items()))
    fig_arvores(arvores)
    fig_dispersao(df)
    tabelas_latex(res, coefs, med, cols)
    numero("sens_instancias", int(len(df)))
    numero("sens_sinteticas", int((df.family != "tsplib").sum()))
    numero("sens_features_removidas_vif", removidas)
    for _, r in res.iterrows():
        numero(f"sens_{r.alvo}_r2cv_lin", r.r2_cv_linear)
        numero(f"sens_{r.alvo}_r2cv_arv", r.r2_cv_arvore)
        numero(f"sens_{r.alvo}_r2cv_rf", r.r2_cv_floresta)
    return df, rho, med, res, coefs, arvores, removidas
