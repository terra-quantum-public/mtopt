import numpy as np
import pytest

from mtopt.maxvol import maxvol, maxvol_rect


# ----------------------------------------------------------------------
# maxvol basic behavior
# ----------------------------------------------------------------------


def test_maxvol_on_square_matrix_returns_identity_like_coeff():
    """If matrix is square, maxvol should just return all rows and the matrix itself."""
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    row_indices, coeff_matrix = maxvol(matrix)

    # For n == r, we expect row_indices = [0, 1], coeff_matrix = matrix
    assert row_indices.shape == (2,)
    np.testing.assert_array_equal(row_indices, np.array([0, 1]))
    np.testing.assert_allclose(coeff_matrix, matrix)


def test_maxvol_requires_tall_matrix():
    """maxvol should raise on non-tall matrices (n < r)."""
    matrix = np.ones((2, 3), dtype=float)  # n=2, r=3 -> not tall
    with pytest.raises(ValueError):
        _ = maxvol(matrix)


def test_maxvol_shapes_and_reconstruction_quality():
    """maxvol should produce a good approximate factorization A ≈ B @ A[I, :]."""
    rng = np.random.default_rng(123)
    num_rows, rank = 20, 5
    matrix = rng.standard_normal((num_rows, rank))

    row_indices, coeff_matrix = maxvol(matrix, accuracy=1.05, max_iters=100)

    # Shape checks
    assert row_indices.shape == (rank,)
    assert coeff_matrix.shape == (num_rows, rank)

    # Indices must be valid and unique
    assert np.all((row_indices >= 0) & (row_indices < num_rows))
    assert len(np.unique(row_indices)) == rank

    # Reconstruction quality
    reconstructed = coeff_matrix @ matrix[row_indices, :]
    num = np.linalg.norm(matrix - reconstructed)
    den = np.linalg.norm(matrix)
    rel_err = num / den if den > 0 else num

    # Use a reasonably loose tolerance to avoid flakiness
    assert rel_err < 1e-5


# ----------------------------------------------------------------------
# maxvol_rect basic behavior
# ----------------------------------------------------------------------


def test_maxvol_rect_reduces_to_maxvol_when_no_extra_rows():
    """maxvol_rect with zero extra rows should match maxvol output up to numerical precision."""
    rng = np.random.default_rng(42)
    num_rows, rank = 15, 4
    matrix = rng.standard_normal((num_rows, rank))

    row_indices_sq, coeff_sq = maxvol(matrix, accuracy=1.02, max_iters=50)
    row_indices_rect, coeff_rect = maxvol_rect(
        matrix,
        accuracy=1.1,
        min_extra_rows=0,
        max_extra_rows=0,
        base_accuracy=1.02,
        base_max_iters=50,
    )

    # Row selection must be identical
    np.testing.assert_array_equal(row_indices_rect, row_indices_sq)

    # Coefficients should be numerically very close (differences ~1e-16)
    np.testing.assert_allclose(
        coeff_rect,
        coeff_sq,
        rtol=1e-7,
        atol=1e-12,  # allow tiny absolute differences
    )


def test_maxvol_rect_shapes_and_constraints():
    """maxvol_rect should respect min/max extra rows and produce a good factorization."""
    rng = np.random.default_rng(321)
    num_rows, rank = 25, 6
    matrix = rng.standard_normal((num_rows, rank))

    min_extra_rows = 2
    max_extra_rows = 4

    row_indices, coeff_matrix = maxvol_rect(
        matrix,
        accuracy=1.1,
        min_extra_rows=min_extra_rows,
        max_extra_rows=max_extra_rows,
        base_accuracy=1.05,
        base_max_iters=20,
    )

    # Check sizes
    total_rows = row_indices.shape[0]
    assert rank + min_extra_rows <= total_rows <= rank + max_extra_rows
    assert coeff_matrix.shape == (num_rows, total_rows)

    # Indices should be valid and unique
    assert np.all((row_indices >= 0) & (row_indices < num_rows))
    assert len(np.unique(row_indices)) == total_rows

    # Reconstruction quality
    reconstructed = coeff_matrix @ matrix[row_indices, :]
    num = np.linalg.norm(matrix - reconstructed)
    den = np.linalg.norm(matrix)
    rel_err = num / den if den > 0 else num
    assert rel_err < 1e-5


def test_maxvol_rect_invalid_bounds_raise():
    """Invalid min/max extra row settings should raise ValueError."""
    matrix = np.random.randn(10, 4)

    # min_extra_rows < 0
    with pytest.raises(ValueError):
        _ = maxvol_rect(matrix, min_extra_rows=-1, max_extra_rows=2)

    # min_extra_rows > max_extra_rows
    with pytest.raises(ValueError):
        _ = maxvol_rect(matrix, min_extra_rows=3, max_extra_rows=1)

    # max_extra_rows too large (but note: implementation clips r+max_extra_rows to n,
    # so to actually trigger ValueError we need an inconsistent min rank)
    with pytest.raises(ValueError):
        _ = maxvol_rect(matrix, min_extra_rows=20, max_extra_rows=30)


def test_maxvol_rect_no_max_extra_rows_argument():
    """maxvol_rect should work when max_extra_rows is None."""
    rng = np.random.default_rng(777)
    num_rows, rank = 12, 5
    matrix = rng.standard_normal((num_rows, rank))

    row_indices, coeff_matrix = maxvol_rect(
        matrix,
        accuracy=1.5,         # looser accuracy to stop early
        min_extra_rows=0,
        max_extra_rows=None,  # no explicit upper bound
        base_accuracy=1.05,
        base_max_iters=10,
    )

    # Shapes must be consistent
    assert coeff_matrix.shape[0] == num_rows
    assert coeff_matrix.shape[1] == row_indices.shape[0]

    # Indices valid
    assert np.all((row_indices >= 0) & (row_indices < num_rows))

    # Basic reconstruction sanity (looser tolerance because we may select fewer rows)
    reconstructed = coeff_matrix @ matrix[row_indices, :]
    num = np.linalg.norm(matrix - reconstructed)
    den = np.linalg.norm(matrix)
    rel_err = num / den if den > 0 else num
    assert rel_err < 1e-4
