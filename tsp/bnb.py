"""Branch-and-Bound exato para o TSP simétrico.

Um nó é um caminho parcial (0, v1, ..., vk); fixar o vértice 0 elimina as
rotações. Limites inferiores, calculados para todos os filhos de uma vez:
  twomin   duas menores arestas de cada vértice, somadas e divididas por 2;
  onetree  custo(P) + d(vk, c) + MST(R - c) + min d(c, R - c) + min d(R - c, 0);
  hk       o mesmo sobre w_ij = d_ij + pi_i + pi_j (Held-Karp), menos 2 sum(pi),
           com pi otimizado uma vez na raiz por subgradiente.
Custos inteiros: poda quando ceil(LB) >= UB. Busca em profundidade (memória
O(n^2)) ou best-first (memória exponencial, com teto de nós abertos).
"""
from __future__ import annotations

import heapq
import itertools
import math
from time import perf_counter

import numpy as np

from .heuristics import nearest_neighbor_tour
from .mst import mst_without_each_vertex, prim, tree_degrees


def _cost(D: np.ndarray, tour) -> float:
    t = np.asarray(tour)
    return float(D[t, np.roll(t, -1)].sum())


BOUNDS = ("twomin", "onetree", "hk")
SEARCHES = ("dfs", "best")
CHILD_ORDERS = ("bound", "index")


class _Stop(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def one_tree(W: np.ndarray) -> tuple[float, np.ndarray]:
    """1-árvore mínima com vértice especial 0. Devolve (peso, graus)."""
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
    """Penalidades pi de Held-Karp por subgradiente (passo de Polyak).

    g = grau - 2; pi += lam (UB - L) g / |g|^2; lam cai pela metade após
    `patience` iterações sem melhora. O UB só calibra o passo e é sempre o do
    vizinho mais próximo, para o limite não depender do warm-start.
    Devolve (pi, melhor L, iterações).
    """
    n = D.shape[0]
    Dinf = D.astype(float).copy()
    np.fill_diagonal(Dinf, np.inf)
    if n < 3:
        return np.zeros(n), 0.0, 0
    if ub is None:
        ub = _cost(D, nearest_neighbor_tour(D))
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
            break   # 1-árvore virou tour, passo minúsculo ou L alcançou o UB
        pi = pi + lam * (ub - L) / norm2 * g
    return best_pi, float(best_L), it


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

    def _prunable(self, lb: float) -> bool:
        if self.integral:
            # custos inteiros: só vale continuar se existir tour <= UB - 1;
            # a tolerância absorve o erro de ponto flutuante das penalidades pi
            return math.ceil(lb - 1e-6) >= self.ub
        return lb >= self.ub - 1e-9

    def _prepare_bound(self):
        n = self.n
        self.pi = np.zeros(n)
        self.hk_iterations = 0
        if self.bound == "twomin":
            two = np.partition(self.Dinf, 1, axis=1)[:, :2]
            self.s2 = two.sum(axis=1)
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

    def _child_bounds(self, last: int, R: np.ndarray, cost_d: float,
                      cost_w: float) -> np.ndarray:
        """LB de cada filho P + (c), c em R (|R| >= 2)."""
        t0 = perf_counter()
        m = R.size
        # aresta de volta ao 0: a 2ª menor substitui a menor quando ela é a do próprio c
        if self.bound == "twomin":
            to0 = self.D[R, 0]
            S = self.Dinf[R[:, None], R]
        else:
            to0 = self.W[R, 0]
            S = self.W[R[:, None], R]
        a = int(np.argmin(to0))
        second = np.partition(to0, 1)[1]
        conn0 = np.full(m, to0[a])
        conn0[a] = second
        near_c = S.min(axis=1)

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

    def _update_incumbent(self, total: float, tour: list[int]):
        if total < self.ub - 1e-9:
            self.ub = total
            self.best_tour = tour
            self.ub_updates += 1
            self.ub_history.append((perf_counter() - self.t_start, total))

    def _check_time(self):
        if perf_counter() > self.deadline:
            raise _Stop("timeout")

    def _dfs(self, last: int, depth: int, cost_d: float, cost_w: float):
        self._check_time()
        R = np.flatnonzero(~self.visited)
        if R.size == 1:
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
                    # filhos ordenados: os irmãos seguintes também caem
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

    def _best_first(self):
        """Best-first; o caminho de cada nó aberto é uma lista encadeada (c, pai)."""
        n = self.n
        tie = itertools.count()
        # chave: (LB, -profundidade, contador) -> empate favorece o nó mais fundo
        heap = [(float(self.root_lb), 0, next(tie), 0.0, 0.0, (0, None))]
        visited = np.zeros(n, dtype=bool)
        try:
            while heap:
                lb, negdepth, _, cost_d, cost_w, node = heapq.heappop(heap)
                depth = -negdepth
                if self._prunable(lb):
                    # heap ordenado por LB: todos os nós abertos restantes também caem
                    self.pruned += 1 + len(heap)
                    self.pruned_by_depth[depth] += 1
                    for item in heap:
                        self.pruned_by_depth[-item[1]] += 1
                    heap.clear()
                    break
                self._check_time()
                visited[:] = False
                v = node
                while v is not None:
                    visited[v[0]] = True
                    v = v[1]
                R = np.flatnonzero(~visited)
                last = node[0]
                if R.size == 1:
                    c = int(R[0])
                    self.leaves += 1
                    path, v = [], node
                    while v is not None:
                        path.append(v[0])
                        v = v[1]
                    self._update_incumbent(cost_d + self.D[last, c] + self.D[c, 0],
                                           path[::-1] + [c])
                    continue
                self.expanded += 1
                self.expanded_by_depth[depth] += 1
                self.max_depth = max(self.max_depth, depth)
                lbs = self._child_bounds(last, R, cost_d, cost_w).tolist()
                self.generated += R.size
                Dl, Wl = self.D[last], self.W[last]
                usa_w = self.bound != "twomin"      # no twomin W = D
                for k, c in enumerate(R.tolist()):
                    if self._prunable(lbs[k]):
                        self.pruned += 1
                        self.pruned_by_depth[depth + 1] += 1
                        continue
                    heapq.heappush(heap, (lbs[k], -(depth + 1), next(tie),
                                          cost_d + float(Dl[c]),
                                          cost_w + float(Wl[c]) if usa_w else 0.0,
                                          (c, node)))
                self.max_open = max(self.max_open, len(heap))
                if len(heap) > self.max_open_cap:
                    raise _Stop("memout")
        finally:
            self.open_at_end = len(heap)

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
            self.ub = _cost(self.D, initial_tour)
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
                    self._update_incumbent(_cost(self.D, list(range(n))), list(range(n)))
            elif self._prunable(self.root_lb):
                self.pruned += 1            # a raiz já prova a incumbente
                self.pruned_by_depth[0] += 1
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
            if math.isinf(x):
                return None
            return int(round(x)) if self.integral else float(x)

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
