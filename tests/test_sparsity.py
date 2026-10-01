import numpy as np
import pytest

from sos_portfolio import ChordalExtension, clique_tree, verify_rip
from sos_portfolio.graph_sparsity import build_block_adjacency, is_chordal
from util import cliques_of, graph_edges


def adj_from(kind, n, seed=0):
    a = np.zeros((n, n), dtype=bool)
    for i, j in graph_edges(kind, n, seed):
        a[i, j] = a[j, i] = True
    return a


def test_disjoint_blocks():
    ce = ChordalExtension(build_block_adjacency(12, 4)).build()
    assert sorted(map(tuple, ce.maximal_cliques)) == [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)]
    assert ce.fill_in_count() == 0 and ce.verify_rip()


@pytest.mark.parametrize("kind,n", [("chain", 9), ("star", 7), ("random", 14), ("random", 20)])
def test_clique_tree_order_satisfies_rip_and_covers_edges(kind, n):
    a = adj_from(kind, n, seed=3)
    ce = ChordalExtension(a).build()
    assert is_chordal(ce.chordal_adjacency, ce.perfect_elimination_ordering)
    cl = ce.maximal_cliques
    assert verify_rip(cl)
    assert set(v for c in cl for v in c) == set(range(n))
    for i, j in zip(*np.where(np.triu(a, 1))):
        assert any(i in c and j in c for c in cl)


def test_chain_cliques_overlap():
    assert sorted(map(tuple, cliques_of("chain", 7))) == [(0, 1, 2), (2, 3, 4), (4, 5, 6)]


def test_verify_rip_rejects_bad_order_and_non_chordal_family():
    # order matters: {0,1},{2,3},{1,2}: intersection of the third with the union is {1,2}, in no single clique
    assert not verify_rip([[0, 1], [2, 3], [1, 2]])
    # 4-cycle cliques (non-chordal family) violate RIP in any order
    assert not verify_rip([[0, 1], [1, 2], [2, 3], [3, 0]])


def test_clique_tree_reorders_to_rip():
    bad = [[0, 1], [2, 3], [1, 2]]
    ordered, parent = clique_tree(bad)
    assert verify_rip(ordered) and parent[0] == -1
    assert sorted(map(tuple, ordered)) == sorted(map(tuple, bad))


def test_mdo_adds_fill_in_for_cycle():
    a = np.zeros((4, 4), dtype=bool)
    for i in range(4):
        a[i, (i + 1) % 4] = a[(i + 1) % 4, i] = True
    ce = ChordalExtension(a).build()
    assert ce.fill_in_count() == 1 and is_chordal(ce.chordal_adjacency, ce.perfect_elimination_ordering)
