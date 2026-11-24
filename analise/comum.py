"""Funções comuns da análise: figuras, tabelas LaTeX e ajustes de curvas."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
RES = RAIZ / "results"
ANALISE = RES / "analise"
FIG = RAIZ / "relatorio" / "figuras"
TAB = RAIZ / "relatorio" / "tabelas"
for d in (ANALISE, FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

STARTS = ["trivial", "nn", "hull", "tatt", "christofides"]
START_ROTULO = {"trivial": "Trivial", "nn": "Viz. próximo", "hull": "Fecho convexo",
                "tatt": "TATT", "christofides": "Christofides"}
CORES = {"trivial": "#7f7f7f", "nn": "#1f77b4", "hull": "#2ca02c",
         "tatt": "#ff7f0e", "christofides": "#d62728"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
    "font.family": "serif",
})

# números citados no artigo
_NUMEROS: dict = {}


def numero(chave: str, valor):
    """Guarda um número para o texto (results/analise/numeros.json)."""
    if isinstance(valor, (np.floating, np.integer)):
        valor = valor.item()
    _NUMEROS[chave] = valor
    return valor


def salva_numeros():
    arq = ANALISE / "numeros.json"
    antigos = json.loads(arq.read_text()) if arq.exists() else {}
    antigos.update(_NUMEROS)
    arq.write_text(json.dumps(antigos, indent=1, ensure_ascii=False, sort_keys=True))


def salva_fig(fig, nome: str):
    fig.savefig(FIG / f"{nome}.pdf")
    fig.savefig(FIG / f"{nome}.png", dpi=150)
    plt.close(fig)


def fmt_num(x, casas: int = 2) -> str:
    """Número com vírgula decimal para o LaTeX."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    s = f"{x:,.{casas}f}"
    return s.replace(",", "X").replace(".", "{,}").replace("X", ".")


def tabela_latex(df: pd.DataFrame, nome: str, legenda: str, rotulo: str,
                 alinhamento: str | None = None, pequena: bool = True):
    """Escreve relatorio/tabelas/<nome>.tex (booktabs)."""
    alinhamento = alinhamento or "l" + "r" * (df.shape[1] - 1)
    linhas = [r"\begin{table}[htb]", r"\centering",
              r"\caption{" + legenda + "}", r"\label{" + rotulo + "}"]
    if pequena:
        linhas.append(r"\small")
    linhas += [r"\begin{adjustbox}{max width=\linewidth}",
               r"\begin{tabular}{" + alinhamento + "}", r"\toprule",
               " & ".join(map(str, df.columns)) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        linhas.append(" & ".join(map(str, row.values)) + r" \\")
    linhas += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    (TAB / f"{nome}.tex").write_text("\n".join(linhas) + "\n")


MODELOS = {
    "n": lambda n: n,
    "n log n": lambda n: n * np.log(n),
    "n^2": lambda n: n ** 2,
    "n^2 log n": lambda n: n ** 2 * np.log(n),
    "n^3": lambda n: n ** 3,
}
MODELO_TEX = {"n": "$n$", "n log n": r"$n\log n$", "n^2": "$n^2$",
              "n^2 log n": r"$n^2\log n$", "n^3": "$n^3$"}


def r2(y, yhat) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def ajuste_potencia(n, y) -> dict:
    """log y = log a + b log n (mínimos quadrados em log-log).
    Devolve expoente b, IC 95% de b, R² em log e RMSE em log10."""
    from scipy import stats
    n, y = np.asarray(n, float), np.asarray(y, float)
    ok = (y > 0) & np.isfinite(y)
    x, z = np.log(n[ok]), np.log(y[ok])
    res = stats.linregress(x, z)
    tcrit = stats.t.ppf(0.975, max(len(x) - 2, 1))
    zhat = res.intercept + res.slope * x
    return {"b": res.slope, "b_ic": tcrit * res.stderr, "a": np.exp(res.intercept),
            "r2_log": res.rvalue ** 2,
            "rmse_log10": float(np.sqrt(np.mean((z - zhat) ** 2)) / np.log(10))}


def ajuste_modelo(n, y, modelo: str) -> dict:
    """y = c0 + c1 f(n) (c1 >= 0) em escala linear, com R² e RMSE; também a forma
    pura log y = log c + log f, para comparar modelos.
    """
    from scipy.optimize import nnls
    n, y = np.asarray(n, float), np.asarray(y, float)
    f = MODELOS[modelo](n)
    A = np.column_stack([np.ones_like(f), f])
    coef, _ = nnls(A, y)
    yhat = A @ coef
    ok = y > 0
    logc = np.mean(np.log(y[ok]) - np.log(f[ok]))
    zhat = logc + np.log(f[ok])
    return {"modelo": modelo, "c0": coef[0], "c1": coef[1], "r2": r2(y, yhat),
            "rmse": float(np.sqrt(np.mean((y - yhat) ** 2))),
            "r2_log_forma": r2(np.log(y[ok]), zhat),
            "rmse_log10_forma": float(np.sqrt(np.mean((np.log(y[ok]) - zhat) ** 2)) / np.log(10))}


def melhor_modelo(n, y) -> tuple[str, pd.DataFrame]:
    """Compara os modelos teóricos pela forma (RMSE em log10)."""
    tab = pd.DataFrame([ajuste_modelo(n, y, m) for m in MODELOS])
    best = tab.loc[tab.rmse_log10_forma.idxmin(), "modelo"]
    return best, tab
