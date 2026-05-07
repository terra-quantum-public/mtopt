import numpy as np
import pytest

from tq_mtopt.tensor import Tensor, tensordot


def test_tensor_creation_normalizes_edges():
    """Tensor should normalize edges into sorted tuples."""
    arr = np.zeros((2, 3))
    # Non-canonical order in edges:
    t = Tensor(arr, edges=[(1, 0), (2, 1)])

    assert isinstance(t, Tensor)
    assert t.shape == (2, 3)
    # Edges should be sorted
    assert t.edges == [(0, 1), (1, 2)]


def test_tensor_transpose_default_reverses_axes_and_edges():
    """Default transpose should reverse axes and corresponding edges."""
    arr = np.arange(6).reshape(2, 3)
    t = Tensor(arr, edges=["i", "j"])

    # Default transpose reverses axes
    tr = t.transpose()
    assert isinstance(tr, Tensor)
    assert tr.shape == (3, 2)
    # Edges associated with axes should be reversed
    assert tr.edges == [("j",), ("i",)]
    # Numerical content matches numpy
    np.testing.assert_array_equal(tr, arr.T)


def test_tensor_transpose_custom_axes():
    """Transpose with explicit axes should permute edges accordingly."""
    arr = np.zeros((2, 3, 4))
    t = Tensor(arr, edges=["i", "j", "k"])

    tr = t.transpose((1, 2, 0))

    assert tr.shape == (3, 4, 2)
    # Edges should follow the same permutation
    assert tr.edges == [("j",), ("k",), ("i",)]
    np.testing.assert_array_equal(tr, np.transpose(arr, (1, 2, 0)))


def test_tensordot_contracts_along_edge_and_propagates_edges():
    """tensordot must contract along the named edge and combine edges."""
    rng = np.random.default_rng(123)
    A_data = rng.standard_normal((2, 3))
    B_data = rng.standard_normal((3, 4))

    A = Tensor(A_data, edges=["i", "k"])
    B = Tensor(B_data, edges=["k", "j"])

    C = tensordot(A, B, edge="k")

    # Shape should match numpy.tensordot result
    expected = np.tensordot(A_data, B_data, axes=(1, 0))
    assert C.shape == expected.shape
    np.testing.assert_allclose(C, expected)

    # Edges should be A.edges without 'k' + B.edges without 'k'
    assert C.edges == [("i",), ("j",)]


def test_tensordot_raises_if_edge_missing():
    """tensordot should raise a ValueError if the edge is not found."""
    A = Tensor(np.zeros((2, 3)), edges=["i", "k"])
    B = Tensor(np.zeros((3, 4)), edges=["k", "j"])

    with pytest.raises(ValueError):
        tensordot(A, B, edge="z")  # edge 'z' not present in either tensor


def test_tensordot_raises_on_dimension_mismatch():
    """tensordot should raise if dimensions along the contracted edge mismatch."""
    A = Tensor(np.zeros((3, 2)), edges=["k", "i"])
    B = Tensor(np.zeros((4, 2)), edges=["k", "j"])  # dim along 'k' differs (3 vs 4)

    with pytest.raises(ValueError):
        tensordot(A, B, edge="k")


def test_flatten_with_monkeypatched_back_permutation(monkeypatch):
    """
    flatten should produce a 2D Tensor and preserve edge metadata
    according to the permutation returned by back_permutation.

    We monkeypatch back_permutation to have a simple, deterministic
    behavior for the test: move the requested edge to the last axis
    while keeping the others in their original order.
    """
    import tq_mtopt.tensor as tensor_mod

    def fake_back_permutation(edges, edge):
        # edges: list of edge labels; edge: canonical edge to keep last
        idx = edges.index(edge)
        perm = list(range(len(edges)))
        perm.pop(idx)
        perm.append(idx)
        return perm

    monkeypatch.setattr(tensor_mod, "back_permutation", fake_back_permutation)

    arr = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    t = Tensor(arr, edges=["i", "j", "k"])

    flat = t.flatten("k")

    # Numerics: flatten all but last axis
    expected = arr.reshape(2 * 3, 5)
    assert flat.shape == expected.shape
    np.testing.assert_array_equal(flat, expected)

    # Metadata:
    #  - second edge should be the preserved 'k'
    assert flat.edges[1] == ("k",)
    #  - we expect flattened_to to record the logical edge
    assert flat.flattened_to == ("k",)
    #  - expanded_shape should reflect the original tensor shape
    assert flat.expanded_shape == t.expanded_shape == (2, 3, 5)
