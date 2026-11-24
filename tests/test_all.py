"""Testes de corretude (python -m pytest -q): ótimos e tours publicados da
TSPLIB, força bruta em instâncias pequenas e SciPy como referência.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tsp.approx import (blossom_matching, christofides, eulerian_circuit,  # noqa: E402
                        greedy_matching, odd_degree_vertices, twice_around_the_tree)
from tsp.bnb import BranchAndBound, held_karp_bound  # noqa: E402
from tsp.features import instance_features  # noqa: E402
from tsp.heuristics import (convex_hull, convex_hull_insertion_tour,  # noqa: E402
                            nearest_neighbor_tour, trivial_tour)
from tsp.mst import mst_without_each_vertex, prim  # noqa: E402
from tsp.tsplib import (KNOWN_OPTIMA, Instance, _explicit_matrix,  # noqa: E402
                        load_instance, load_tour, tour_cost)

DATA = ROOT / "data"


def euclid(n, seed, integer=True):
    P = np.random.default_rng(seed).uniform(0, 100, (n, 2))
    D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
    return P, (np.floor(D + 0.5) if integer else D)


def brute_force(D):
    n = len(D)
    return min(tour_cost(D, (0,) + p) for p in itertools.permutations(range(1, n)))


def is_tour(t, n):
    return sorted(t) == list(range(n))

@pytest.mark.parametrize("tour_file", sorted(DATA.glob("*.opt.tour.gz")),
                         ids=lambda p: p.name.split(".")[0])
def test_opt_tour_matches_published_optimum(tour_file):
    """Valida a função de distância de cada tipo (GEO, ATT, EXPLICIT, EUC_2D)."""
    name = tour_file.name.split(".")[0]
    inst = load_instance(DATA / f"{name}.tsp.gz")
    assert inst.tour_cost(load_tour(tour_file)) == KNOWN_OPTIMA[name]


@pytest.mark.parametrize("fmt", ["UPPER_ROW", "LOWER_ROW", "UPPER_DIAG_ROW",
                                 "LOWER_DIAG_ROW", "UPPER_COL", "LOWER_COL",
                                 "FULL_MATRIX"])
def test_explicit_formats(fmt):
    _, M = euclid(6, 0)
    n = 6
    order = {
        "UPPER_ROW": [(i, j) for i in range(n) for j in range(i + 1, n)],
        "LOWER_ROW": [(i, j) for i in range(n) for j in range(i)],
        "UPPER_DIAG_ROW": [(i, j) for i in range(n) for j in range(i, n)],
        "LOWER_DIAG_ROW": [(i, j) for i in range(n) for j in range(i + 1)],
        "UPPER_COL": [(i, j) for j in range(n) for i in range(j)],
        "LOWER_COL": [(i, j) for j in range(n) for i in range(j + 1, n)],
        "FULL_MATRIX": [(i, j) for i in range(n) for j in range(n)],
    }[fmt]
    w = [M[i, j] for i, j in order]
    assert np.array_equal(_explicit_matrix(n, fmt, w), M)

@pytest.mark.parametrize("seed", range(5))
def test_prim_matches_scipy(seed):
    _, D = euclid(40, seed, integer=False)
    parent, w = prim(D)
    assert w == pytest.approx(minimum_spanning_tree(D).sum())
    assert (parent >= 0).sum() == 39


@pytest.mark.parametrize("seed", range(6))
def test_mst_leaf_shortcut_matches_batched(seed):
    from tsp.naive import sequential_children_mst
    rng = np.random.default_rng(seed)
    for m in (3, 4, 7, 15, 30):
        if seed % 2:                                   # muitos empates: pontos numa grade
            P = rng.integers(0, 4, (m, 2)).astype(float) + rng.integers(0, 2, (m, 1)) * 1e-9
        else:
            P = rng.uniform(0, 1, (m, 2))
        S = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
        np.fill_diagonal(S, np.inf)
        a = mst_without_each_vertex(S, leaf_shortcut=True)
        b = mst_without_each_vertex(S, leaf_shortcut=False)
        c = np.array(sequential_children_mst(S))
        assert np.allclose(a, b) and np.allclose(b, c)


def test_mst_without_each_vertex():
    for m in range(2, 15):
        _, S = euclid(m, m, integer=False)
        np.fill_diagonal(S, np.inf)
        got = mst_without_each_vertex(S)
        for c in range(m):
            idx = [i for i in range(m) if i != c]
            exp = prim(S[np.ix_(idx, idx)])[1] if len(idx) > 1 else 0.0
            assert got[c] == pytest.approx(exp)

@pytest.mark.parametrize("seed", range(5))
def test_convex_hull_matches_scipy(seed):
    P, _ = euclid(60, seed, integer=False)
    assert set(convex_hull(P)) == set(ConvexHull(P).vertices)


@pytest.mark.parametrize("seed", range(5))
def test_heuristics_return_valid_tours(seed):
    P, D = euclid(30, seed)
    for t in (trivial_tour(30), nearest_neighbor_tour(D),
              convex_hull_insertion_tour(D, P),
              twice_around_the_tree(D), christofides(D),
              christofides(D, matching="greedy")):
        assert is_tour(t, 30) and t[0] == 0


def test_nearest_neighbor_first_step():
    _, D = euclid(20, 3)
    Dinf = D.copy()
    np.fill_diagonal(Dinf, np.inf)
    t = nearest_neighbor_tour(D)
    assert D[0, t[1]] == Dinf[0].min() or D[0, t[-1]] == Dinf[0].min()

@pytest.mark.parametrize("seed", range(15))
def test_approximation_guarantees(seed):
    """Em métrica exata: TATT <= 2 OPT e Christofides <= 1,5 OPT."""
    _, D = euclid(8, seed, integer=False)
    opt = brute_force(D)
    assert tour_cost(D, twice_around_the_tree(D)) <= 2 * opt + 1e-6
    assert tour_cost(D, christofides(D)) <= 1.5 * opt + 1e-6


@pytest.mark.parametrize("seed", range(10))
def test_blossom_is_min_weight_perfect_matching(seed):
    _, D = euclid(8, seed, integer=False)
    odd = np.arange(8)

    def best(vs):
        if not vs:
            return 0.0
        a, rest = vs[0], vs[1:]
        return min(D[a, b] + best(rest[:i] + rest[i + 1:]) for i, b in enumerate(rest))

    pairs = blossom_matching(D, odd)
    assert sorted(v for p in pairs for v in p) == list(range(8))
    assert sum(D[u, v] for u, v in pairs) == pytest.approx(best(list(range(8))))
    g = greedy_matching(D, odd)
    assert sorted(v for p in g for v in p) == list(range(8))


def test_eulerian_circuit_uses_each_edge_once():
    _, D = euclid(25, 1)
    parent, _ = prim(D)
    odd = odd_degree_vertices(parent)
    edges = [(int(parent[v]), v) for v in range(25) if parent[v] >= 0]
    edges += blossom_matching(D, odd)
    circ = eulerian_circuit(25, edges, 0)
    assert len(circ) == len(edges) + 1 and circ[0] == circ[-1] == 0
    used = sorted(tuple(sorted(e)) for e in zip(circ, circ[1:]))
    assert used == sorted(tuple(sorted(e)) for e in edges)

@pytest.mark.parametrize("seed", range(12))
def test_bnb_matches_brute_force(seed):
    n = 4 + seed % 5
    _, D = euclid(n, seed)
    opt = brute_force(D)
    for bound in ("twomin", "onetree", "hk"):
        for search in ("dfs", "best"):
            for order in ("bound", "index"):
                for init in (None, trivial_tour(n)):
                    r = BranchAndBound(D, bound, search, order).solve(init)
                    assert r["status"] == "optimal" and r["cost"] == opt
                    assert tour_cost(D, r["tour"]) == opt
                    assert r["root_lb"] <= opt + 1e-6
                    # todo nó gerado foi podado, expandido ou é folha
                    assert r["nodes_generated"] == (r["nodes_pruned"] + r["nodes_expanded"]
                                                    + r["leaves"])
                    # contagens por profundidade batem com os totais
                    assert sum(r["pruned_by_depth"]) == r["nodes_pruned"]
                    assert sum(r["expanded_by_depth"]) == r["nodes_expanded"]


@pytest.mark.parametrize("seed", range(6))
def test_bnb_non_integer_distances(seed):
    """Com distâncias reais, o B&B não pode arredondar a incumbente."""
    n = 5 + seed % 4
    _, D = euclid(n, seed, integer=False)
    opt = min(float(D[np.array((0,) + p), np.roll(np.array((0,) + p), -1)].sum())
              for p in itertools.permutations(range(1, n)))      # sem arredondar
    for bound in ("twomin", "onetree", "hk"):
        for search in ("dfs", "best"):
            r = BranchAndBound(D, bound, search).solve(nearest_neighbor_tour(D))
            assert r["status"] == "optimal"
            assert abs(r["cost"] - opt) < 1e-6


def test_root_pruned_when_warm_start_is_optimal():
    inst = load_instance(DATA / "ulysses16.tsp.gz")
    opt_tour = load_tour(DATA / "ulysses16.opt.tour.gz")
    r = BranchAndBound(inst.D, "hk").solve(opt_tour)
    assert r["cost"] == inst.known_opt
    if r["nodes_expanded"] == 0:                     # a raiz já provou o ótimo
        assert r["nodes_pruned"] == 1 and r["pruned_by_depth"][0] == 1


@pytest.mark.parametrize("name", ["burma14", "ulysses16", "gr17", "gr21",
                                  "ulysses22", "gr24", "fri26", "bays29"])
def test_bnb_finds_published_optimum(name):
    inst = load_instance(DATA / f"{name}.tsp.gz")
    r = BranchAndBound(inst.D, "hk", time_limit=120).solve(nearest_neighbor_tour(inst.D))
    assert r["status"] == "optimal" and r["cost"] == inst.known_opt


def test_held_karp_is_lower_bound():
    for name in ("gr24", "att48", "st70"):
        inst = load_instance(DATA / f"{name}.tsp.gz")
        _, L, _ = held_karp_bound(inst.D)
        assert L <= inst.known_opt + 1e-6


def test_bnb_timeout_keeps_partial_stats():
    inst = load_instance(DATA / "att48.tsp.gz")
    r = BranchAndBound(inst.D, "onetree", time_limit=1.0).solve(trivial_tour(inst.n))
    assert r["status"] == "timeout" and r["cost"] is None
    assert r["nodes_expanded"] > 0 and r["best_found"] is not None
    assert r["time_bnb"] < 3.0


def test_best_first_memory_cap():
    inst = load_instance(DATA / "gr17.tsp.gz")
    r = BranchAndBound(inst.D, "twomin", "best", max_open=1000).solve(trivial_tour(17))
    assert r["status"] == "memout"
    assert r["open_at_end"] > 1000                  # nós abertos no momento da parada


def test_naive_bnb_matches_optimum():
    from tsp.naive import naive_bnb
    for seed in range(4):
        _, D = euclid(7, seed)
        opt = brute_force(D)
        assert naive_bnb(D)["cost"] == opt
        assert naive_bnb(D, initial_tour=nearest_neighbor_tour(D))["cost"] == opt

def test_features_ranges():
    P, D = euclid(30, 0, integer=False)
    inst = Instance("rnd", 30, "EUC_2D", D, P, "node_coord")
    f = instance_features(inst)
    assert 0 < f["aspect_ratio"] <= 1
    assert 0 < f["hull_fraction"] <= 1
    assert 0 <= f["spatial_entropy"] <= 1
    assert f["triangle_violation_frac"] == 0.0
    assert f["mst_max_degree"] >= 2


def test_clark_evans_detects_patterns():
    """~1 em pontos uniformes, > 1 em grade, < 1 em aglomerados."""
    from tsp.synthetic import make_instance
    def ce(fam, seeds=range(4), n=50):
        return np.mean([instance_features(make_instance(fam, n, s))["clark_evans"] for s in seeds])
    assert 0.9 < ce("uniform") < 1.2
    assert ce("grid") > 1.4
    assert ce("cluster") < 0.85


def test_heuristic_gc_count_excludes_forced_collections():
    import run_one
    rec = run_one.main(["burma14", "--algo", "trivial", "--reps", "3", "--data-dir", str(DATA)])
    assert rec["gc_collections"] == 0
