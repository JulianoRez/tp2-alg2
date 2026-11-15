"""Branch-and-Bound exato para o TSP simétrico.

Espaço de busca
---------------
Cada nó da árvore é um caminho parcial que começa no vértice 0:
    P = (0, v1, ..., vk),   R = vértices ainda não visitados.
Os filhos de P são P + (c) para cada c em R. Uma folha (|R| = 1) fecha o tour.
Fixar o vértice inicial elimina as n rotações equivalentes de cada tour.

Limites inferiores (opção `bound`)
---------------------------------
Todos são calculados para TODOS os filhos de um nó de uma vez (vetorizado).

* 'twomin'  (limite clássico do livro do Levitin): cada vértice contribui com
  as duas arestas mais baratas incidentes a ele (ou as arestas já fixadas no
  caminho). Como cada aresta do tour é contada por suas duas pontas,
      custo(tour) >= (soma das contribuições) / 2.
  Custo O(m) por filho, mas é fraco -> árvore enorme.

* 'onetree' (limite de árvore geradora): o que falta para fechar o tour é um
  caminho hamiltoniano  c -> r1 -> ... -> rj -> 0  passando por todo R' = R - {c}.
  O trecho r1..rj é uma árvore geradora de R', então
      LB = custo(P) + d(vk, c) + MST(R') + min d(c, R') + min d(R', 0).
  Custo O(m^2) por filho (uma MST), mas bem mais forte.

* 'hk' (Held-Karp): o mesmo limite 'onetree', porém sobre os custos
  penalizados  w(i,j) = d(i,j) + pi_i + pi_j.  Em QUALQUER tour todo vértice
  tem grau 2, então  w(tour) = d(tour) + 2*sum(pi)  -> qualquer limite inferior
  para w vale para d subtraindo a constante 2*sum(pi). As penalidades pi são
  calculadas UMA vez na raiz por otimização por subgradiente da relaxação
  Lagrangiana da 1-árvore (Held & Karp, 1970/1971): elas "empurram" a 1-árvore
  mínima a ter grau 2 em todo vértice, deixando o limite muito mais apertado.

Como os custos da TSPLIB são inteiros, um nó só pode melhorar a incumbente se
ceil(LB) <= UB - 1; então podamos quando ceil(LB) >= UB.

Estratégias de busca (opção `search`)
------------------------------------
* 'dfs'  (padrão): busca em profundidade recursiva; filhos visitados em ordem
  crescente de limite inferior (`child_order='bound'`). Memória O(n^2)
  (pilha de recursão + vetor de limites por nível). Depende fortemente da
  qualidade do limite superior inicial (warm-start).
* 'best': best-first com heap pelo limite inferior. Expande o menor número de
  nós, mas guarda TODOS os nós abertos -> memória pode crescer
  exponencialmente. Há um teto de nós abertos (`max_open`).
"""
from __future__ import annotations

import heapq
import itertools
import math
from time import perf_counter

import numpy as np

from .heuristics import nearest_neighbor_tour
from .mst import mst_without_each_vertex, prim, tree_degrees
from .tsplib import tour_cost

BOUNDS = ("twomin", "onetree", "hk")
SEARCHES = ("dfs", "best")
CHILD_ORDERS = ("bound", "index")


class _Stop(Exception):
    """Interrompe a busca (tempo ou memória esgotados) preservando estatísticas."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# --------------------------------------------------------------------------
# Limite de Held-Karp na raiz (1-árvore + subgradiente)
# --------------------------------------------------------------------------

def one_tree(W: np.ndarray) -> tuple[float, np.ndarray]:
    """1-árvore mínima com vértice especial 0: MST(V - {0}) + 2 arestas mais
    baratas de 0. Todo tour é uma 1-árvore, então peso(1-árvore) <= OPT.
    Devolve (peso, graus)."""
    n = W.shape[0]
    parent, w = prim(W[1:, 1:])
    deg = np.zeros(n, dtype=np.int64)
    deg[1:] = tree_degrees(parent)
    row = W[0, 1:]
    two = np.argpartition(row, 1)[:2]
    deg[0] = 2
    deg[1 + two] += 1
    return w + float(row[two].sum()), deg


def held_karp_bound(D: np.ndarray, ub: float | None = None,
                    max_iter: int | None = None) -> tuple[np.ndarray, float, int]:
    """Otimização por subgradiente das penalidades pi (Held-Karp).

    L(pi) = peso_w(1-árvore mínima) - 2*sum(pi) é limite inferior para todo pi.
    Subgradiente: g_v = grau_v - 2 (grau na 1-árvore). Atualização:
        pi <- pi + t * g,    t = lam * (UB - L(pi)) / ||g||^2   (passo de Polyak)
    lam começa em 2 e cai pela metade quando L não melhora por `patience`
    iterações. Vértice com grau > 2 fica mais caro (pi sobe), folha fica mais
    barata (pi desce).

    O UB aqui só calibra o tamanho do passo. Por padrão usamos o vizinho mais
    próximo, calculado internamente da MESMA forma para todas as estratégias
    de warm-start -> o limite (e portanto a poda) não depende da estratégia;
    as diferenças entre estratégias vêm só da incumbente inicial do B&B.

    Devolve (pi, melhor L, iterações).
    """
    n = D.shape[0]
    Dinf = D.astype(float).copy()
    np.fill_diagonal(Dinf, np.inf)
    if n < 3:
        return np.zeros(n), 0.0, 0
    if ub is None:
        ub = float(tour_cost(D, nearest_neighbor_tour(D)))
    max_iter = max_iter or 50 * n
    patience = max(5, n // 5)
    pi = np.zeros(n)
    best_pi, best_L = pi.copy(), -np.inf
    lam, stall, it = 2.0, 0, 0
    for it in range(1, max_iter + 1):
        W = Dinf + pi[:, None] + pi[None, :]
        w, deg = one_tree(W)
        L = w - 2.0 * pi.sum()
        if L > best_L + 1e-9:
            best_L, best_pi, stall = L, pi.copy(), 0
        else:
            stall += 1
            if stall >= patience:
                lam, stall = lam / 2.0, 0
        g = deg - 2
        norm2 = float(g @ g)
        if norm2 == 0 or lam < 1e-4 or best_L >= ub - 1e-9:
            break   # 1-árvore virou tour (ótimo), passo minúsculo, ou provou UB
        pi = pi + lam * (ub - L) / norm2 * g
    return best_pi, float(best_L), it


# --------------------------------------------------------------------------
# Branch-and-Bound
# --------------------------------------------------------------------------

class BranchAndBound:
    def __init__(self, D: np.ndarray, bound: str = "hk", search: str = "dfs",
                 child_order: str = "bound", time_limit: float = 1800.0,
                 max_open: int = 2_000_000, hk_iters: int | None = None):
        if bound not in BOUNDS or search not in SEARCHES or child_order not in CHILD_ORDERS:
            raise ValueError("opção inválida")
        self.D = np.asarray(D, dtype=float)
        self.n = self.D.shape[0]
        self.bound, self.search, self.child_order = bound, search, child_order
        self.time_limit, self.max_open_cap, self.hk_iters = time_limit, max_open, hk_iters
        self.integral = bool(np.all(self.D == np.round(self.D)))
        self.Dinf = self.D.copy()
        np.fill_diagonal(self.Dinf, np.inf)

    # ---------------------------------------------------------------- poda
    def _prunable(self, lb: float) -> bool:
        if self.integral:
            return math.ceil(lb - 1e-6) >= self.ub
        return lb >= self.ub - 1e-9

    # ------------------------------------------------- pré-processamento
    def _prepare_bound(self):
        n = self.n
        self.pi = np.zeros(n)
        self.hk_iterations = 0
        if self.bound == "twomin":
            two = np.partition(self.Dinf, 1, axis=1)[:, :2]
            self.s2 = two.sum(axis=1)               # 2 menores arestas de cada vértice
            self.root_lb = float(self.s2.sum() / 2)
            self.W = self.Dinf
            self.const = 0.0
            return
        if self.bound == "hk":
            self.pi, self.root_lb, self.hk_iterations = held_karp_bound(
                self.D, max_iter=self.hk_iters)
        else:
            self.root_lb = one_tree(self.Dinf)[0] if n >= 3 else 0.0
        self.W = self.Dinf + self.pi[:, None] + self.pi[None, :]
        self.const = 2.0 * float(self.pi.sum())

    # ------------------------------------- limites de todos os filhos
    def _child_bounds(self, last: int, R: np.ndarray, cost_d: float,
                      cost_w: float) -> np.ndarray:
        """LB de cada filho P + (c), c em R (vetor alinhado com R). |R| >= 2."""
        t0 = perf_counter()
        m = R.size
        # aresta (r, 0) que fecha o tour: menor e 2ª menor para "excluir c"
        if self.bound == "twomin":
            to0 = self.D[R, 0]
            S = self.Dinf[R[:, None], R]       # submatriz m x m (mais barato que np.ix_)
        else:
            to0 = self.W[R, 0]
            S = self.W[R[:, None], R]
        a = int(np.argmin(to0))
        second = np.partition(to0, 1)[1]
        conn0 = np.full(m, to0[a])
        conn0[a] = second                   # se c é o mais próximo de 0, usa o 2º
        near_c = S.min(axis=1)              # aresta mais barata de c para R - {c}

        if self.bound == "twomin":
            # interior do caminho conta 2x as arestas fixadas; pontas (0 e c)
            # contam 1 aresta fixada + a mais barata livre; R' usa as 2 menores
            s2 = self.s2[R]
            lb = (2.0 * (cost_d + self.D[last, R]) + conn0 + near_c
                  + (s2.sum() - s2)) / 2.0
        else:
            mst = mst_without_each_vertex(S)
            lb = cost_w + self.W[last, R] + mst + near_c + conn0 - self.const
        self.time_bound += perf_counter() - t0
        return lb

    # ---------------------------------------------------- incumbente
    def _update_incumbent(self, total: float, tour: list[int]):
        if total < self.ub - 1e-9:
            self.ub = total
            self.best_tour = tour
            self.ub_updates += 1
            self.ub_history.append((perf_counter() - self.t_start, total))

    def _check_time(self):
        if perf_counter() > self.deadline:
            raise _Stop("timeout")

    # ------------------------------------------------------------ DFS
    def _dfs(self, last: int, depth: int, cost_d: float, cost_w: float):
        self._check_time()
        R = np.flatnonzero(~self.visited)
        if R.size == 1:                     # folha: fechar o tour é forçado
            c = int(R[0])
            self.leaves += 1
            self._update_incumbent(cost_d + self.D[last, c] + self.D[c, 0],
                                   self.path + [c])
            return
        self.expanded += 1
        self.expanded_by_depth[depth] += 1
        self.max_depth = max(self.max_depth, depth)
        lb = self._child_bounds(last, R, cost_d, cost_w)
        m = R.size
        self.generated += m
        order = np.argsort(lb, kind="stable") if self.child_order == "bound" else range(m)
        for pos, k in enumerate(order):
            if self._prunable(lb[k]):
                if self.child_order == "bound":
                    # ordenado: todos os irmãos restantes também são podados
                    self.pruned += m - pos
                    self.pruned_by_depth[depth + 1] += m - pos
                    return
                self.pruned += 1
                self.pruned_by_depth[depth + 1] += 1
                continue
            c = int(R[k])
            self.visited[c] = True
            self.path.append(c)
            self._dfs(c, depth + 1, cost_d + self.D[last, c], cost_w + self.W[last, c])
            self.path.pop()
            self.visited[c] = False

    # ----------------------------------------------------- best-first
    def _best_first(self):
        """Best-first: sempre expande o nó aberto de menor LB."""
        n = self.n
        tie = itertools.count()
        heap = [(self.root_lb, 0, next(tie), 0.0, 0.0, (0,))]
        visited = np.zeros(n, dtype=bool)
        while heap:
            lb, negdepth, _, cost_d, cost_w, path = heapq.heappop(heap)
            depth = -negdepth
            if self._prunable(lb):
                self.pruned += 1 + len(heap)
                self.pruned_by_depth[depth] += 1
                for item in heap:
                    self.pruned_by_depth[-item[1]] += 1
                heap.clear()
                break
            self._check_time()
            visited[:] = False
            visited[list(path)] = True
            R = np.flatnonzero(~visited)
            last = path[-1]
            if R.size == 1:
                c = int(R[0])
                self.leaves += 1
                self._update_incumbent(cost_d + self.D[last, c] + self.D[c, 0],
                                       list(path) + [c])
                continue
            self.expanded += 1
            self.expanded_by_depth[depth] += 1
            self.max_depth = max(self.max_depth, depth)
            lbs = self._child_bounds(last, R, cost_d, cost_w)
            self.generated += R.size
            for k, c in enumerate(R):
                c = int(c)
                if self._prunable(lbs[k]):
                    self.pruned += 1
                    self.pruned_by_depth[depth + 1] += 1
                    continue
                heapq.heappush(heap, (lbs[k], -(depth + 1), next(tie),
                                      cost_d + self.D[last, c],
                                      cost_w + self.W[last, c],
                                      path + (c,)))
            self.max_open = max(self.max_open, len(heap))
            if len(heap) > self.max_open_cap:
                raise _Stop("memout")
        self.open_at_end = len(heap)

    # ---------------------------------------------------------- solve
    def solve(self, initial_tour: list[int] | None = None) -> dict:
        n = self.n
        self.t_start = perf_counter()
        self.deadline = self.t_start + self.time_limit
        self.generated, self.expanded, self.pruned, self.leaves = 1, 0, 0, 0
        self.ub_updates, self.max_depth, self.max_open, self.open_at_end = 0, 0, 0, 0
        self.expanded_by_depth = [0] * (n + 1)
        self.pruned_by_depth = [0] * (n + 1)
        self.ub_history: list[tuple[float, float]] = []
        self.time_bound = 0.0

        self._prepare_bound()
        time_root = perf_counter() - self.t_start

        if initial_tour is not None:
            self.ub = float(tour_cost(self.D, initial_tour))
            self.best_tour = list(initial_tour)
        else:
            self.ub, self.best_tour = math.inf, None
        initial_ub = self.ub
        self.ub_history.append((0.0, initial_ub))

        status = "optimal"
        t_search = perf_counter()
        try:
            if n <= 3:
                if self.best_tour is None:
                    self._update_incumbent(float(tour_cost(self.D, list(range(n)))), list(range(n)))
            elif self._prunable(self.root_lb):
                self.pruned += 1
            elif self.search == "dfs":
                self.visited = np.zeros(n, dtype=bool)
                self.visited[0] = True
                self.path = [0]
                self._dfs(0, 0, 0.0, 0.0)
            else:
                self._best_first()
        except _Stop as stop:
            status = stop.reason
        now = perf_counter()
        last_improvement = self.ub_history[-1][0] if self.ub_updates else 0.0

        def out(x):
            return None if math.isinf(x) else int(round(x))

        return {
            "status": status,
            "cost": out(self.ub) if status == "optimal" else None,
            "best_found": out(self.ub),
            "tour": self.best_tour,
            "initial_ub": out(initial_ub),
            "root_lb": float(self.root_lb),
            "hk_iterations": self.hk_iterations,
            "nodes_generated": self.generated,
            "nodes_expanded": self.expanded,
            "nodes_pruned": self.pruned,
            "leaves": self.leaves,
            "ub_updates": self.ub_updates,
            "max_depth": self.max_depth,
            "max_open": self.max_open,
            "open_at_end": self.open_at_end,
            "expanded_by_depth": self.expanded_by_depth,
            "pruned_by_depth": self.pruned_by_depth,
            "ub_history": self.ub_history,
            "time_root_bound": time_root,
            "time_child_bounds": self.time_bound,
            "time_search": now - t_search,
            "time_to_best": last_improvement,
            "time_bnb": now - self.t_start,
        }
