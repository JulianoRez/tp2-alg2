"""Leitura de instâncias da TSPLIB e construção da matriz de distâncias.

Referência do formato: G. Reinelt, "TSPLIB 95" (documentação oficial).

Suporta os tipos EUC_2D, CEIL_2D, ATT, GEO e EXPLICIT (com todos os
EDGE_WEIGHT_FORMAT usuais). Na faixa N <= 75 a TSPLIB tem só três instâncias
EUC_2D (eil51, berlin52, st70); as demais citadas no enunciado (ulysses16,
gr24, fri26, bays29, swiss42, att48) são GEO, EXPLICIT ou ATT. Por isso o
parser segue a definição oficial de cada tipo: assim o custo de um tour bate
exatamente com o ótimo publicado pela TSPLIB, o que serve de validação.

Toda distância da TSPLIB é inteira. A matriz é guardada em float64 (inteiros
são representados exatamente até 2^53) porque o limite inferior Held-Karp do
Branch-and-Bound soma penalidades reais a ela.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Ótimos publicados pela TSPLIB (página de soluções do TSPLIB95). Servem para
# (1) validar o B&B e (2) calcular o gap quando o B&B estoura o tempo limite.
KNOWN_OPTIMA = {
    "burma14": 3323, "ulysses16": 6859, "gr17": 2085, "gr21": 2707,
    "ulysses22": 7013, "gr24": 1272, "fri26": 937, "bayg29": 1610,
    "bays29": 2020, "dantzig42": 699, "swiss42": 1273, "att48": 10628,
    "gr48": 5046, "hk48": 11461, "eil51": 426, "berlin52": 7542,
    "brazil58": 25395, "st70": 675, "eil76": 538, "pr76": 108159,
}


@dataclass
class Instance:
    name: str
    n: int
    edge_weight_type: str
    D: np.ndarray            # matriz n x n de distâncias inteiras (diagonal 0)
    coords: np.ndarray       # n x 2: coordenadas usadas só para geometria
    coords_source: str       # 'node_coord' | 'display_data' | 'mds'
    comment: str = ""

    @property
    def known_opt(self) -> int | None:
        return KNOWN_OPTIMA.get(self.name)

    def tour_cost(self, tour) -> int:
        return tour_cost(self.D, tour)


def tour_cost(D: np.ndarray, tour) -> int:
    """Custo do ciclo hamiltoniano `tour` (lista de vértices, sem repetir o 1º)."""
    t = np.asarray(tour)
    return int(round(D[t, np.roll(t, -1)].sum()))


# --------------------------------------------------------------------------
# Leitura do arquivo
# --------------------------------------------------------------------------

def _read_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return f.read()
    return path.read_text()


def _parse(path: Path) -> tuple[dict, dict]:
    """Separa o arquivo em cabeçalho (CHAVE: valor) e seções (lista de tokens)."""
    header: dict[str, str] = {}
    sections: dict[str, list[str]] = {}
    current = None
    for raw in _read_text(path).splitlines():
        line = raw.strip()
        if not line:
            continue
        first = line.split()[0].rstrip(":").upper()
        if first == "EOF":
            break
        if first.endswith("_SECTION"):
            current = first
            sections[current] = []
            continue
        if ":" in line and not _is_number(first):
            key, _, value = line.partition(":")
            header[key.strip().upper()] = value.strip()
            current = None
            continue
        if current is not None:
            sections[current].extend(line.split())
    return header, sections


def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _coords_from_section(tokens: list[str], n: int) -> np.ndarray:
    """Seções de coordenadas têm linhas 'id x y'; devolve array n x 2."""
    arr = np.array(tokens, dtype=float).reshape(-1, 3)
    out = np.zeros((n, 2))
    out[arr[:, 0].astype(int) - 1] = arr[:, 1:]
    return out


# --------------------------------------------------------------------------
# Funções de distância (definições da documentação TSPLIB95)
# --------------------------------------------------------------------------

def _nint(x: np.ndarray) -> np.ndarray:
    return np.floor(x + 0.5)


def _pairwise_deltas(c: np.ndarray):
    dx = c[:, 0][:, None] - c[:, 0][None, :]
    dy = c[:, 1][:, None] - c[:, 1][None, :]
    return dx, dy


def dist_euc_2d(c: np.ndarray) -> np.ndarray:
    dx, dy = _pairwise_deltas(c)
    return _nint(np.sqrt(dx * dx + dy * dy))


def dist_ceil_2d(c: np.ndarray) -> np.ndarray:
    dx, dy = _pairwise_deltas(c)
    return np.ceil(np.sqrt(dx * dx + dy * dy))


def dist_att(c: np.ndarray) -> np.ndarray:
    """Distância 'pseudo-euclidiana' (att48, att532)."""
    dx, dy = _pairwise_deltas(c)
    r = np.sqrt((dx * dx + dy * dy) / 10.0)
    t = _nint(r)
    return np.where(t < r, t + 1, t)


def _geo_radians(c: np.ndarray) -> np.ndarray:
    """Converte coordenadas GEO (formato GRAUS.MINUTOS) para radianos.

    A parte inteira é truncada (o "nint" da documentação, na prática, é
    truncamento: é assim que os ótimos publicados foram calculados, conferido
    em tests/test_all.py com ulysses16/ulysses22/burma14).
    """
    pi = 3.141592
    deg = np.trunc(c)
    minutes = c - deg
    return pi * (deg + 5.0 * minutes / 3.0) / 180.0


def dist_geo(c: np.ndarray) -> np.ndarray:
    rad = _geo_radians(c)
    lat, lon = rad[:, 0], rad[:, 1]
    rrr = 6378.388
    q1 = np.cos(lon[:, None] - lon[None, :])
    q2 = np.cos(lat[:, None] - lat[None, :])
    q3 = np.cos(lat[:, None] + lat[None, :])
    arg = np.clip(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3), -1.0, 1.0)
    d = np.floor(rrr * np.arccos(arg) + 1.0)
    np.fill_diagonal(d, 0)
    return d


def _explicit_matrix(n: int, fmt: str, w: list[float]) -> np.ndarray:
    """Monta a matriz completa a partir de EDGE_WEIGHT_SECTION."""
    fmt = fmt.upper()
    if fmt == "FULL_MATRIX":
        return np.array(w[: n * n], dtype=float).reshape(n, n)
    # Para matrizes simétricas, ler "coluna" de um triângulo equivale a ler
    # "linha" do triângulo oposto.
    ranges = {
        "UPPER_ROW": lambda i: range(i + 1, n),
        "LOWER_COL": lambda i: range(i + 1, n),
        "LOWER_ROW": lambda i: range(0, i),
        "UPPER_COL": lambda i: range(0, i),
        "UPPER_DIAG_ROW": lambda i: range(i, n),
        "LOWER_DIAG_COL": lambda i: range(i, n),
        "LOWER_DIAG_ROW": lambda i: range(0, i + 1),
        "UPPER_DIAG_COL": lambda i: range(0, i + 1),
    }
    if fmt not in ranges:
        raise ValueError(f"EDGE_WEIGHT_FORMAT não suportado: {fmt}")
    tri = np.zeros((n, n))
    it = iter(w)
    for i in range(n):
        for j in ranges[fmt](i):
            tri[i, j] = next(it)
    full = tri + tri.T
    np.fill_diagonal(full, 0)
    return full


# --------------------------------------------------------------------------
# API pública
# --------------------------------------------------------------------------

def instance_name(path: Path) -> str:
    name = path.name
    for suf in (".gz", ".tsp"):
        if name.endswith(suf):
            name = name[: -len(suf)]
    return name


def load_instance(path: str | Path) -> Instance:
    path = Path(path)
    header, sections = _parse(path)
    n = int(header["DIMENSION"])
    ewt = header.get("EDGE_WEIGHT_TYPE", "").upper()
    # alguns arquivos (ulysses16/22) trazem NAME com extensão: normalizamos
    name = instance_name(Path(header.get("NAME", path.name).strip()))

    node = sections.get("NODE_COORD_SECTION")
    node_c = _coords_from_section(node, n) if node else None

    if ewt == "EUC_2D":
        D = dist_euc_2d(node_c)
    elif ewt == "CEIL_2D":
        D = dist_ceil_2d(node_c)
    elif ewt == "ATT":
        D = dist_att(node_c)
    elif ewt == "GEO":
        D = dist_geo(node_c)
    elif ewt == "EXPLICIT":
        w = [float(x) for x in sections["EDGE_WEIGHT_SECTION"]]
        D = _explicit_matrix(n, header["EDGE_WEIGHT_FORMAT"], w)
    else:
        raise ValueError(f"EDGE_WEIGHT_TYPE não suportado: {ewt}")
    np.fill_diagonal(D, 0)
    if not np.allclose(D, D.T):
        raise ValueError(f"{name}: matriz assimétrica (não é TSP simétrico)")

    display = sections.get("DISPLAY_DATA_SECTION")
    if node_c is not None:
        coords, source = node_c, "node_coord"
    elif display:
        coords, source = _coords_from_section(display, n), "display_data"
    else:
        coords, source = np.zeros((n, 2)), "none"

    return Instance(name=name, n=n, edge_weight_type=ewt, D=D,
                    coords=coords, coords_source=source,
                    comment=header.get("COMMENT", ""))


def load_tour(path: str | Path) -> list[int]:
    """Lê um arquivo .opt.tour (vértices numerados a partir de 1)."""
    _, sections = _parse(Path(path))
    tour = []
    for tok in sections["TOUR_SECTION"]:
        v = int(tok)
        if v == -1:
            break
        tour.append(v - 1)
    return tour


def find_instance(name_or_path: str, data_dir: str | Path = "data") -> Path:
    p = Path(name_or_path)
    if p.exists():
        return p
    for cand in (Path(data_dir) / f"{name_or_path}.tsp.gz",
                 Path(data_dir) / f"{name_or_path}.tsp"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"instância não encontrada: {name_or_path} "
                            f"(rode download_instances.py)")
