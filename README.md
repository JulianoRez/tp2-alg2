# TP2 — Análise Empírica e Otimização para o Caixeiro Viajante (DCC207)

Branch-and-Bound exato (3 limites inferiores, DFS/best-first, 5 warm-starts),
Twice-around-the-tree e Christofides, com instrumentação por subrotina
(tempo e memória), extração de features das instâncias e execução
retomável dos experimentos.

> Artigo (modelo oficial da SBC): **[relatorio/main.pdf](relatorio/main.pdf)** (fonte em `relatorio/`; compilar com `cd relatorio && latexmk -pdf main.tex`).

## Instalação

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python download_instances.py        # baixa as 15 instâncias padrão para data/
python -m pytest -q                 # 103 testes (~5 s)
```

## Estrutura

| Arquivo | Conteúdo |
|---|---|
| `tsp/tsplib.py` | Parser TSPLIB (EUC_2D, CEIL_2D, ATT, GEO, EXPLICIT), ótimos conhecidos, MDS |
| `tsp/mst.py` | Prim O(n²) vetorizado; MST "sem cada vértice" em lote (usada no B&B) |
| `tsp/heuristics.py` | Trivial, vizinho mais próximo, inserção a partir do fecho convexo |
| `tsp/approx.py` | Twice-around-the-tree, Christofides (blossom ou guloso), Hierholzer |
| `tsp/bnb.py` | Branch-and-Bound: limites `twomin`/`onetree`/`hk`, busca `dfs`/`best` |
| `tsp/features.py` | Features geométricas/topológicas das instâncias |
| `tsp/profiler.py` | Profiling por subrotina (perf_counter + tracemalloc) |
| `tsp/naive.py` | Versões "brutas" usadas só como linha de base na auditoria |
| `run_one.py` | Executa uma configuração e imprime um JSON |
| `run_experiments.py` | Matriz completa de experimentos (paralela, retomável) |
| `experiment_config.py` | Instâncias, estratégias e configurações do B&B |
| `profile_scaling.py` | Profiling em instâncias aleatórias de N crescente (regressão) |
| `audit_naive.py` | Código bruto x refatorado (tempo, memória, nós) |
| `tsp/synthetic.py`, `run_synthetic.py` | Instâncias sintéticas adversariais (7 famílias) com ótimo exato |
| `analise/` | Análise completa: regressões, warm-start, sensibilidade, auditoria (`python -m analise.roda`) |
| `run_all.sh`, `espera_maquina.sh` | Pipeline completo dos experimentos (espera a máquina ficar livre) |
| `export_sheets.py` | Tabelas para a planilha (`results/planilha/`) |
| `relatorio/` | Artigo em LaTeX (padrão SBC), figuras e tabelas geradas pela análise |
| `tests/test_all.py` | Testes de corretude |

## Como rodar

```bash
# uma execução isolada
python run_one.py att48 --algo bnb --start hull --bound hk --child-order index
python run_one.py st70 --algo christofides --reps 5 --trace-memory

# experimentos principais (gera results/features.csv, runs.jsonl, runs.csv)
python run_experiments.py --dry-run                      # lista as execuções
python run_experiments.py --no-bnb --workers 1           # heurísticas/aproximativos (tempos limpos)
nohup python run_experiments.py --no-heuristics --workers 8 > results/bnb.log 2>&1 &

# complementares
python profile_scaling.py                 # results/scaling.csv (N = 25 a 6400)
python profile_scaling.py --no-gc --out results/scaling_nogc.csv
python profile_scaling.py --sizes --bnb-sizes 10 15 20 --bnb-bound hk --out results/scaling_bnb_hk.csv
python audit_naive.py                     # results/audit.csv
python run_synthetic.py --workers 8       # results/synthetic.csv (168 instâncias)

# tudo, na ordem certa (medições de tempo sozinhas na máquina):
setsid nohup ./run_all.sh > results/run_all.log 2>&1 &

# análise -> relatorio/figuras, relatorio/tabelas, results/analise
python -m analise.roda
cd relatorio && latexmk -pdf main.tex
```

A matriz padrão tem 300 execuções de B&B (15 instâncias × 5 warm-starts ×
4 configurações) e 90 de heurísticas. As execuções com 1-árvore (quase
metade estoura os 30 min) levaram ~5 h com 8 processos (lote v1, em
`results/v1/`, mantido porque o código da DFS não mudou). A coleta final
(`run_all.sh`: heurísticas, escala com e sem GC, auditoria, B&B Held-Karp
com e sem GC, escala do B&B e instâncias sintéticas) levou ~2 h. Se o
processo for interrompido, rode o mesmo comando de novo: o que já está em
`runs.jsonl` / `synthetic.jsonl` é pulado.

Ajustes: edite `experiment_config.py` ou use `--instances`, `--starts`,
`--configs hk:index onetree:bound`, `--time-limit`.

## Dados gerados

### `results/runs.csv` (uma linha por execução)

Comuns: `instance, n, edge_weight_type, algo, status, cost, known_opt, gap`
(= (cost − ótimo)/ótimo), `time_total` (s), `peak_rss_kb` (pico de memória
residente do processo), `peak_rss_before_run_kb` (linha de base:
interpretador + bibliotecas), `gc_collections`, `gc_enabled`.

Heurísticas/aproximativos (`algo` ∈ trivial, nn, hull, tatt, christofides):
- `time_total` = mediana de `reps` execuções; `time_total_all` = todas.
- `time_<subrotina>` (mediana): `mst`, `preorder`, `odd_vertices`,
  `matching_build`, `matching_solve`, `euler`, `shortcut`, `nn`, `hull`.
- `mem_peak_<subrotina>` (bytes, execução separada com tracemalloc): memória
  alocada a mais durante a subrotina; `mem_peak_total` = algoritmo inteiro.
- `matching` = `blossom` (ótimo, O(k³)) ou `greedy` (sem garantia).

Branch-and-Bound (`algo = bnb`):
- Configuração: `start` (warm-start), `bound`, `search`, `child_order`, `time_limit`.
- `status`: `optimal` | `timeout` (estourou 30 min → `cost` = NA) | `memout`
  (best-first passou do teto de nós abertos) | `killed` (processo morto).
- `best_found`: melhor tour encontrado (mesmo em timeout); `initial_ub`:
  custo do warm-start; `root_lb`: limite inferior na raiz.
- `nodes_generated`: nós criados (limite calculado), inclui a raiz.
  `nodes_expanded`: nós cujos filhos foram gerados. `nodes_pruned`: nós
  descartados porque ⌈LB⌉ ≥ UB. `leaves`: tours completos avaliados.
  Em execução completa: generated = pruned + expanded + leaves.
- `expanded_by_depth`, `pruned_by_depth`: listas (JSON) indexadas pela
  profundidade (raiz = 0). `max_depth`: maior profundidade expandida.
- `ub_updates`, `ub_history` (JSON de pares [tempo, custo]), `time_to_best`.
- Tempos: `time_heuristic` (warm-start), `time_root_bound` (Held-Karp/1-árvore
  na raiz), `time_child_bounds` (soma do cálculo dos limites dos filhos — a
  subrotina dominante), `time_search`, `time_bnb`, `time_total`.
- `hk_iterations`, `max_open`/`open_at_end` (best-first).

### `results/features.csv` (uma linha por instância)

`aspect_ratio` (lado menor/maior do bounding box), `hull_fraction` (fração de
pontos no fecho convexo), `spatial_entropy` (entropia normalizada numa grade
≈√(n/2)×√(n/2)), `clark_evans` (<1 agrupado, ≈1 aleatório, >1 regular),
`silhouette_best`/`kmeans_best_k` (clusterização), `nn_dist_cv` (CV das
distâncias ao vizinho mais próximo), `dist_cv`, `mst_max_degree` (hubs),
`mst_leaf_fraction`, `mst_odd_fraction` (tamanho do matching do
Christofides), `mst_edge_cv`, `mst_weight`, `triangle_violation_frac` e
`triangle_violation_max_rel` (violações da desigualdade triangular),
`opt_over_mst`, `coords_source` (`node_coord`, `display_data` ou `mds`) e
`mds_quality`.

### Demais resultados

- `results/scaling.csv` / `scaling_nogc.csv`: profiling por subrotina em instâncias uniformes (N = 25 a 6400), com e sem GC.
- `results/scaling_bnb_*.csv`: B&B em instâncias uniformes (10 sementes por N), com nós, tempo, tempo de CPU e memória (tracemalloc).
- `results/audit.csv`: código bruto × refatorado (ver `audit_naive.py`).
- `results/synthetic.csv`: 168 instâncias sintéticas com features, gaps exatos e gaps dos limites na raiz.
- `results/analise/`: tabelas da análise (ajustes, warm-start, sensibilidade, auditoria) e `numeros.json` com os números citados no artigo.
- `results/planilha/`: as mesmas tabelas no formato da planilha (`export_sheets.py`).
- `results/v1/`: o primeiro lote, antes das correções da revisão cruzada.

Campos novos em `runs.csv` (B&B): `cpu_time_total` (tempo de CPU do processo),
`mem_peak_bnb` e `mem_peak_root_bound` (bytes, execução curta sob tracemalloc).

## Observação sobre as instâncias

O enunciado pede instâncias EUC_2D com N ∈ [10, 75], mas a TSPLIB só tem
três nessa faixa (eil51, berlin52, st70), e vários exemplos do próprio
enunciado (ulysses16 = GEO, gr24/fri26/swiss42 = EXPLICIT, att48 = ATT) não
são EUC_2D. O código usa a distância oficial de cada tipo (validada contra
os tours ótimos publicados, ver testes). Instâncias sem coordenadas (gr17,
gr21, gr24, fri26, swiss42) recebem coordenadas 2D por MDS clássico, usadas
**apenas** nas features geométricas e na heurística do fecho convexo.
