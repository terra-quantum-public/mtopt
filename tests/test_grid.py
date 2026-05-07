import numpy as np
import pytest

from tq_mtopt.grid import (
    Grid,
    cartesian_product,
    direct_sum,
    regularized_inverse,
)


# ----------------------------------------------------------------------
# Grid construction and basic properties
# ----------------------------------------------------------------------


def test_grid_init_and_shape():
    arr = [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
    coords = [10, 20]

    g = Grid(arr, coords)

    assert isinstance(g, Grid)
    assert g.shape() == (3, 2)
    assert g.num_points() == 3
    assert g.num_coords() == 2
    assert np.array_equal(g.coords, np.array(coords))


def test_grid_init_1d_grid_and_scalar_coord():
    arr = [0.0, 1.0, 2.0, 3.0]
    coord = 7

    g = Grid(arr, coord)

    # grid should be (4, 1)
    assert g.shape() == (4, 1)
    assert np.array_equal(g.coords, np.array([coord]))


# ----------------------------------------------------------------------
# Cartesian product and permutation
# ----------------------------------------------------------------------


def test_grid_matmul_cartesian_product():
    # First grid: 2 points in 1D
    g1 = Grid([[0.0], [1.0]], coords=[0])
    # Second grid: 3 points in 1D
    g2 = Grid([[10.0], [20.0], [30.0]], coords=[1])

    g = g1 @ g2

    # Cartesian product: 2 * 3 = 6 points, 2 coordinates
    assert g.shape() == (6, 2)
    assert np.array_equal(g.coords, np.array([0, 1]))

    # Check the actual combinations (manual enumeration)
    expected = np.array(
        [
            [0.0, 10.0],
            [0.0, 20.0],
            [0.0, 30.0],
            [1.0, 10.0],
            [1.0, 20.0],
            [1.0, 30.0],
        ]
    )
    np.testing.assert_allclose(g.grid, expected)


def test_grid_permute_sorts_coords_and_tracks_inverse():
    rng = np.random.default_rng(42)
    arr = rng.random((4, 3))
    coords = np.array([5, 1, 7])

    g = Grid(arr, coords)
    assert np.array_equal(g.permutation, np.argsort(coords))

    gp = g.permute()

    # Coordinates should now be sorted
    assert np.array_equal(gp.coords, np.sort(coords))

    # Permutation of gp should map back to original coords
    recovered_coords = gp.coords[gp.permutation]
    assert np.array_equal(recovered_coords, coords)

    # Grid columns must match the permutation of the original
    np.testing.assert_allclose(gp.grid, arr[:, np.argsort(coords)])


# ----------------------------------------------------------------------
# Random subset
# ----------------------------------------------------------------------


def test_grid_random_subset_respects_size_and_rows():
    np.random.seed(0)  # For reproducibility of np.random.choice

    arr = np.arange(10).reshape(5, 2).astype(float)
    coords = [0, 1]
    g = Grid(arr, coords)

    # Request fewer points than available
    subset = g.random_subset(3)
    assert subset.num_points() == 3
    assert subset.num_coords() == g.num_coords()
    assert np.array_equal(subset.coords, g.coords)

    # Every row of subset must be a row of the original grid
    for row in subset.grid:
        assert any(np.array_equal(row, orig) for orig in g.grid)

    # Request more points than available → we should get all points
    subset_all = g.random_subset(10)
    assert subset_all.num_points() == g.num_points()
    assert subset_all.num_coords() == g.num_coords()


# ----------------------------------------------------------------------
# Indexing behavior
# ----------------------------------------------------------------------


def test_grid_getitem_row_int():
    arr = np.arange(6).reshape(3, 2).astype(float)
    coords = [0, 1]
    g = Grid(arr, coords)

    g_row = g[1]

    assert isinstance(g_row, Grid)
    # One row, all coordinates
    assert g_row.shape() == (1, 2)
    np.testing.assert_allclose(g_row.grid[0], arr[1])
    assert np.array_equal(g_row.coords, g.coords)


def test_grid_getitem_columns_slice_and_equivalence():
    arr = np.arange(12).reshape(3, 4).astype(float)
    coords = [0, 1, 2, 3]
    g = Grid(arr, coords)

    # Select last two columns via full 2D slicing
    g_cols_2d = g[:, 2:]
    # Select last two columns via 1D indexing
    g_cols_1d = g[2:]

    assert g_cols_2d.shape() == (3, 2)
    assert g_cols_1d.shape() == (3, 2)

    np.testing.assert_allclose(g_cols_2d.grid, arr[:, 2:])
    np.testing.assert_allclose(g_cols_1d.grid, arr[:, 2:])

    assert np.array_equal(g_cols_2d.coords, np.array(coords[2:]))
    assert np.array_equal(g_cols_1d.coords, np.array(coords[2:]))


def test_grid_getitem_rows_and_columns_tuple():
    arr = np.arange(12).reshape(3, 4).astype(float)
    coords = [10, 11, 12, 13]
    g = Grid(arr, coords)

    # Select rows 1:3, column 0
    g_sub = g[1:3, 0]

    assert g_sub.shape() == (2, 1)  # 2 rows, 1 coord
    np.testing.assert_allclose(g_sub.grid, arr[1:3, [0]])
    assert np.array_equal(g_sub.coords, np.array([coords[0]]))

    # Select a single scalar (1, 0) → normalized to (1, 1)
    g_scalar = g[1, 0]
    assert g_scalar.shape() == (1, 1)
    np.testing.assert_allclose(g_scalar.grid, arr[1:2, 0:1])
    assert np.array_equal(g_scalar.coords, np.array([coords[0]]))


# ----------------------------------------------------------------------
# Evaluate and transform
# ----------------------------------------------------------------------


def test_grid_evaluate_scalar_function():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    coords = [0, 1]
    g = Grid(arr, coords)

    def f(point):
        return np.sum(point)

    values = g.evaluate(f)
    np.testing.assert_allclose(values, np.array([3.0, 7.0]))


def test_grid_transform_vector_function():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    coords = [0, 1]
    g = Grid(arr, coords)

    def square(point):
        return point**2

    gt = g.transform(square)

    assert isinstance(gt, Grid)
    assert gt.shape() == g.shape()
    np.testing.assert_allclose(gt.grid, arr**2)
    assert np.array_equal(gt.coords, g.coords)


# ----------------------------------------------------------------------
# Grid addition and direct_sum/cartesian_product helpers
# ----------------------------------------------------------------------


def test_grid_add_concatenates_rows_when_coords_match():
    arr1 = np.array([[0.0, 1.0], [2.0, 3.0]])
    arr2 = np.array([[10.0, 11.0]])
    coords = [0, 1]

    g1 = Grid(arr1, coords)
    g2 = Grid(arr2, coords)

    g = g1 + g2

    assert g.shape() == (3, 2)
    expected = np.vstack([arr1, arr2])
    np.testing.assert_allclose(g.grid, expected)
    assert np.array_equal(g.coords, np.array(coords))


def test_grid_add_raises_on_coord_mismatch():
    arr = np.array([[0.0, 1.0], [2.0, 3.0]])
    g1 = Grid(arr, [0, 1])
    g2 = Grid(arr, [1, 0])  # same length, but different order

    with pytest.raises(ValueError):
        _ = g1 + g2


def test_cartesian_product_multiple_grids():
    g1 = Grid([[0.0], [1.0]], coords=[0])
    g2 = Grid([[10.0], [20.0]], coords=[1])
    g3 = Grid([[100.0]], coords=[2])

    gc = cartesian_product([g1, g2, g3])

    # 2 * 2 * 1 = 4 points, 3 coordinates
    assert gc.shape() == (4, 3)
    assert np.array_equal(gc.coords, np.array([0, 1, 2]))

    expected = np.array(
        [
            [0.0, 10.0, 100.0],
            [0.0, 20.0, 100.0],
            [1.0, 10.0, 100.0],
            [1.0, 20.0, 100.0],
        ]
    )
    np.testing.assert_allclose(gc.grid, expected)


def test_direct_sum_multiple_grids():
    arr = np.array([[0.0, 1.0], [2.0, 3.0]])
    coords = [0, 1]
    g1 = Grid(arr, coords)
    g2 = Grid(arr + 10.0, coords)
    g3 = Grid(arr + 20.0, coords)

    gd = direct_sum([g1, g2, g3])

    # 3 copies stacked
    assert gd.shape() == (6, 2)
    expected = np.vstack([arr, arr + 10.0, arr + 20.0])
    np.testing.assert_allclose(gd.grid, expected)
    assert np.array_equal(gd.coords, np.array(coords))


# ----------------------------------------------------------------------
# regularized_inverse tests
# ----------------------------------------------------------------------


def test_regularized_inverse_approximate_inverse_on_diagonal():
    # Simple diagonal matrix; well-conditioned
    A = np.diag([1.0, 2.0, 3.0])
    lam = 1e-3

    A_inv_reg = regularized_inverse(A, lambda_reg=lam)
    # For a diagonal A, A_inv_reg @ A should be close to identity
    I_est = A_inv_reg @ A

    np.testing.assert_allclose(I_est, np.eye(3), rtol=1e-5, atol=1e-5)


def test_regularized_inverse_zero_lambda_and_empty_matrix():
    # Zero lambda_reg should be internally clamped to eps
    A = np.array([[1.0, 0.0], [0.0, 2.0]])
    A_inv_reg = regularized_inverse(A, lambda_reg=0.0)
    # Just check that result is finite and shape is correct
    assert A_inv_reg.shape == (2, 2)
    assert np.all(np.isfinite(A_inv_reg))

    # Empty matrix: (0, 0)
    A_empty = np.zeros((0, 0))
    inv_empty = regularized_inverse(A_empty, lambda_reg=1e-3)
    assert inv_empty.shape == (0, 0)

    # Rectangular with zero rows
    A_rect = np.zeros((0, 3))
    inv_rect = regularized_inverse(A_rect, lambda_reg=1e-3)
    # Should be zeros with transposed shape (3, 0)
    assert inv_rect.shape == (3, 0)
    assert np.all(inv_rect == 0.0)
