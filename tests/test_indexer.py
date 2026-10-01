from math import comb

import numpy as np

from sos_portfolio import SparseIndexer


def test_dense_counts():
    ix = SparseIndexer(4, 2, [[0, 1, 2, 3]])
    assert ix.n_moments == comb(8, 4)
    assert ix.pair_index(0, 2).shape == (15, 15)


def test_disjoint_cliques_share_only_constant():
    ix = SparseIndexer(6, 2, [[0, 1, 2], [3, 4, 5]])
    assert ix.n_moments == 2 * (comb(7, 4) - 1) + 1


def test_overlapping_cliques_share_moments():
    ix = SparseIndexer(5, 2, [[0, 1, 2], [2, 3, 4]])
    # moments of x2 (the shared variable) up to degree 4 are common
    assert ix.n_moments == 2 * comb(7, 4) - 5
    a = ix.index((0, 0, 3, 0, 0))
    I0 = ix.pair_index(0, 2)
    I1 = ix.pair_index(1, 2)
    assert a in I0 and a in I1
    assert not ix.has_moment((1, 0, 0, 1, 0))


def test_hankel_structure_and_symmetry():
    ix = SparseIndexer(3, 2, [[0, 1, 2]])
    I = ix.pair_index(0, 2)
    assert (I == I.T).all()
    mons = ix.local_monomials(0, 2)
    for a, m in enumerate(mons):
        for b, q in enumerate(mons):
            assert I[a, b] == ix.index(tuple(u + v for u, v in zip(m, q)))


def test_basis_restriction_keeps_full_key_set():
    ix = SparseIndexer(3, 2, [[0, 1, 2]], basis=[[0, 1]])
    assert ix.pair_index(0, 2).shape == (6, 6)
    assert ix.n_moments == comb(7, 4)
