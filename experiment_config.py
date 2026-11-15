"""Configuração padrão dos experimentos (edite aqui para mudar o conjunto).

Seleção de instâncias
---------------------
O enunciado pede instâncias EUC_2D com N em [10, 75], 12 a 16 instâncias e
~80% com N <= 40. Na TSPLIB só existem 3 instâncias EUC_2D nessa faixa
(eil51, berlin52, st70), e os próprios exemplos do enunciado (ulysses16,
gr24, fri26, bays29, swiss42, att48) são GEO/EXPLICIT/ATT. Por isso usamos
todas as instâncias TSP simétricas da faixa, com a distância oficial de cada
tipo. Instâncias TSP da TSPLIB com N <= 40 são só 9, então a proporção
possível é 11/15 com N <= 42.
"""

# 11 instâncias com N <= 42 (faixa principal) + 4 em [45, 75] (estresse)
DEFAULT_INSTANCES = [
    "burma14", "ulysses16", "gr17", "gr21", "ulysses22", "gr24", "fri26",
    "bayg29", "bays29", "dantzig42", "swiss42",
    "att48", "eil51", "berlin52", "st70",
]

# Alternativas na mesma faixa (sem coordenadas, exceto eil76/pr76 com N=76)
EXTRA_INSTANCES = ["gr48", "hk48", "brazil58", "eil76", "pr76"]

# Estratégias de warm-start (limite superior inicial do B&B)
STARTS = ["trivial", "nn", "hull", "tatt", "christofides"]

# Configurações do B&B. Cada tupla = (limite inferior, ordem dos filhos).
#   hk/onetree: força do limite inferior (Held-Karp vs 1-árvore sem penalidades)
#   index/bound: ordem em que a DFS visita os filhos. Com 'bound' a primeira
#   descida já acha um tour bom e o warm-start pesa pouco; com 'index' a busca
#   é "cega" e o warm-start faz muita diferença.
BNB_CONFIGS = [
    ("hk", "index"),
    ("hk", "bound"),
    ("onetree", "index"),
    ("onetree", "bound"),
]

# Algoritmos rodados isoladamente (tempo por subrotina + memória + custo)
HEURISTIC_RUNS = [
    ("trivial", "blossom"),
    ("nn", "blossom"),
    ("hull", "blossom"),
    ("tatt", "blossom"),
    ("christofides", "blossom"),
    ("christofides", "greedy"),
]

TIME_LIMIT = 1800.0   # 30 minutos por execução
