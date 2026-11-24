"""Configuração padrão dos experimentos.

A TSPLIB só tem 3 instâncias EUC_2D com N em [10, 75] e 9 instâncias com
N <= 40; por isso o conjunto inclui GEO, EXPLICIT e ATT (distância oficial).
"""

# 11 com N <= 42 + 4 em [45, 75]
DEFAULT_INSTANCES = [
    "burma14", "ulysses16", "gr17", "gr21", "ulysses22", "gr24", "fri26",
    "bayg29", "bays29", "dantzig42", "swiss42",
    "att48", "eil51", "berlin52", "st70",
]

EXTRA_INSTANCES = ["gr48", "hk48", "brazil58", "eil76", "pr76"]

STARTS = ["trivial", "nn", "hull", "tatt", "christofides"]

# (limite inferior, ordem dos filhos na DFS)
BNB_CONFIGS = [
    ("hk", "index"),
    ("hk", "bound"),
    ("onetree", "index"),
    ("onetree", "bound"),
]

# (algoritmo, emparelhamento do Christofides)
HEURISTIC_RUNS = [
    ("trivial", "blossom"),
    ("nn", "blossom"),
    ("hull", "blossom"),
    ("tatt", "blossom"),
    ("christofides", "blossom"),
    ("christofides", "greedy"),
]

TIME_LIMIT = 1800.0   # 30 minutos por execução
