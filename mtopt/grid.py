r"""
Grid utilities for tensor rank cross and matrix train optimizers.

This module defines a small :class:`Grid` class and helper routines to build
and manipulate product grids associated with tensor-network structures.

A :class:`Grid` is represented by:

* a 2D array ``grid`` of shape ``(n_points, n_coords)``, where each row is a
  point in coordinate space;
* a 1D array ``coords`` of length ``n_coords`` describing which logical
  coordinates (dimensions) each column corresponds to.

The main operations provided are:

* Cartesian products of grids via the matrix-multiplication operator
  (``grid_a @ grid_b``),
* direct sums of grids,
* propagation of grids on a tensor network graph (node and edge grids),
* selection of subgrids and random subsets,
* a numerically stable Tikhonov-regularized inverse used in TT-cross-like
  algorithms, and
* a ``maxvol_grids`` helper combining grids with a max-volume submatrix
  selection.

The core numerical work relies on :mod:`numpy`. Graph operations assume a
NetworkX-like interface but do not depend on a specific graph implementation.

Provenance
----------
This module contains code adapted from the pyQuTree package by Roman Ellerbrock
(see Ref. [2]). The underlying library and concepts are described in the QuTree
paper (Ref. [1]).

References
----------
.. [1] R. Ellerbrock, K. G. Johnson, S. Seritan, H. Hoppe, J. H. Zhang,
       T. Lenzen, T. Weike, U. Manthe, and T. J. Martínez,
       "QuTree: A tree tensor network package",
       *J. Chem. Phys.* **160**(11), 112501 (2024).
       doi:10.1063/5.0180233

.. [2] R. Ellerbrock, *pyQuTree* (software), Python package ``pyqutree``.
       Source: https://github.com/roman-ellerbrock/pyQuTree
"""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple, Union

import numpy as np

from mtopt.maxvol import maxvol
from mtopt.tensor import Tensor
from mtopt.network import (
    is_leaf_node,
    collect,
    up_leaves,
    sweep,
    pre_edges,
    flip,
)


__all__ = [
    "Grid",
    "cartesian_product",
    "direct_sum",
    "build_node_grid",
    "tensor_network_grid",
    "transform_node_grid",
    "regularized_inverse",
    "maxvol_grids",
]


def _cartesian_product(grid_1: np.ndarray, grid_2: np.ndarray) -> np.ndarray:
    r"""
    Compute the Cartesian product of two 2D grids.

    Given two grids ``grid_1`` and ``grid_2`` with shapes ``(n_1, d_1)`` and
    ``(n_2, d_2)``, this returns a grid ``product`` with shape
    ``(n_1 * n_2, d_1 + d_2)`` whose rows are all concatenated pairs
    of rows from ``grid_1`` and ``grid_2``::

        product = [ [grid_1[0], grid_2[0]],
                    [grid_1[0], grid_2[1]],
                    ...
                    [grid_1[n_1-1], grid_2[n_2-1]] ]

    implemented in a fully vectorized way.

    Parameters
    ----------
    grid_1, grid_2 :
        Input grids as 2D arrays.

    Returns
    -------
    ndarray
        Cartesian product grid as a 2D array.

    Raises
    ------
    ValueError
        If ``grid_1`` or ``grid_2`` is not 2-dimensional.
    """
    grid_1 = np.asarray(grid_1)
    grid_2 = np.asarray(grid_2)

    if grid_1.ndim != 2 or grid_2.ndim != 2:
        raise ValueError(
            "_cartesian_product expects 2D arrays, got shapes "
            f"{grid_1.shape} and {grid_2.shape}."
        )

    n_1, d_1 = grid_1.shape
    n_2, d_2 = grid_2.shape

    # Broadcast to (n_1, n_2, d_1) and (n_1, n_2, d_2)
    grid_1_expanded = grid_1[:, np.newaxis, :]  # (n_1, 1, d_1)
    grid_2_expanded = grid_2[np.newaxis, :, :]  # (1, n_2, d_2)

    grid_1_tiled = np.broadcast_to(grid_1_expanded, (n_1, n_2, d_1))
    grid_2_tiled = np.broadcast_to(grid_2_expanded, (n_1, n_2, d_2))

    product = np.concatenate(
        (grid_1_tiled, grid_2_tiled), axis=2
    )  # (n_1, n_2, d_1 + d_2)
    return product.reshape(n_1 * n_2, d_1 + d_2)


class Grid:
    r"""
    Finite product grid over a set of coordinates.

    A :class:`Grid` stores a collection of points in a coordinate space
    together with the logical coordinate indices/labels for each column.

    Parameters
    ----------
    grid :
        Array-like object describing the grid points. If 1D, it is reshaped
        to ``(n_points, 1)``. If 2D, the shape is interpreted as
        ``(n_points, n_coords)``.
    coords :
        Coordinate indices or labels corresponding to the columns of ``grid``.
        If an integer is given, it is converted to a one-element array. If an
        array-like, it is converted to a 1D :class:`numpy.ndarray`.

    Attributes
    ----------
    grid : ndarray
        Grid points as a 2D array of shape ``(n_points, n_coords)``.
    coords : ndarray
        Coordinate labels as a 1D array of length ``n_coords``.
    permutation : ndarray
        A permutation of ``range(n_coords)`` indicating how coordinates
        should be reordered to achieve a canonical ordering (e.g. sorted).

    Raises
    ------
    ValueError
        If ``grid`` is not 1D or 2D, or if the number of columns in ``grid``
        does not match the length of ``coords``.
    """

    def __init__(
        self,
        grid: Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]],
        coords: Union[int, Sequence[int], np.ndarray],
    ) -> None:
        # Normalize grid to a 2D numpy array.
        if not isinstance(grid, np.ndarray):
            grid = np.array(grid)

        if grid.ndim == 1:
            grid = grid.reshape(-1, 1)
        if grid.ndim != 2:
            raise ValueError("Grid must be a 2D array.")

        self.grid: np.ndarray = np.array(grid)

        # Normalize coords to a 1D numpy array.
        if isinstance(coords, int):
            coords = [coords]
        if not isinstance(coords, np.ndarray):
            coords = np.array(coords)
        self.coords: np.ndarray = coords

        if self.grid.shape[1] != self.coords.shape[0]:
            raise ValueError(
                "Number of columns in grid must match the length of coords "
                f"(got {self.grid.shape[1]} vs {self.coords.shape[0]})."
            )

        # Permutation that sorts coords (used by permute()).
        self.permutation: np.ndarray = np.argsort(self.coords)

    # ------------------------------------------------------------------
    # Basic properties and representations
    # ------------------------------------------------------------------
    def __str__(self) -> str:  # pragma: no cover - cosmetic
        r"""
        String representation listing coordinates and underlying grid.
        """
        return f"coords: {self.coords}\ngrid:\n{self.grid}"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Grid(shape={self.grid.shape}, coords={self.coords})"

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def __matmul__(self, other: "Grid") -> "Grid":
        r"""
        Cartesian product of two grids via the ``@`` operator.

        Parameters
        ----------
        other :
            Another :class:`Grid` instance.

        Returns
        -------
        Grid
            New :class:`Grid` whose points are the Cartesian product of
            ``self.grid`` and ``other.grid`` and whose coordinates are the
            concatenation of ``self.coords`` and ``other.coords``.

        Raises
        ------
        TypeError
            If ``other`` is not a :class:`Grid` instance.
        """
        if not isinstance(other, Grid):
            raise TypeError(f"Grid.__matmul__ expects Grid, got {type(other)!r}.")

        new_grid = _cartesian_product(self.grid, other.grid)
        new_coords = np.concatenate((self.coords, other.coords))
        return Grid(new_grid, new_coords)

    def permute(self) -> "Grid":
        r"""
        Return a new grid with coordinates sorted according to ``coords``.

        This method reorders the columns of :attr:`grid` and the entries
        of :attr:`coords` according to :attr:`permutation`. The resulting
        grid has coordinates in sorted order. Its :attr:`permutation`
        attribute is set to the inverse permutation, which can be used
        elsewhere to restore the original order if needed.

        Returns
        -------
        Grid
            New :class:`Grid` instance with permuted columns and coordinates.
        """
        new_grid = self.grid[:, self.permutation]
        new_coords = self.coords[self.permutation]
        grid = Grid(new_grid, new_coords)
        grid.permutation = np.argsort(self.permutation)
        return grid

    def random_subset(self, n: int) -> "Grid":
        r"""
        Return a random subset of grid points.

        Parameters
        ----------
        n :
            Desired number of points in the subset. If ``n`` exceeds the
            total number of points, all points are returned.

        Returns
        -------
        Grid
            New :class:`Grid` containing at most ``n`` randomly chosen
            rows of :attr:`grid`, with the same :attr:`coords`.
        """
        m = min(n, self.grid.shape[0])
        idx = np.random.choice(self.grid.shape[0], m, replace=False)
        return Grid(self.grid[idx], self.coords)

    def __getitem__(
        self,
        idx: Union[int, slice, np.ndarray, List[int], Tuple[Any, Any]],
    ) -> "Grid":
        r"""
        Extract a subgrid by indexing.

        Indexing behavior is tailored to grid semantics:

        * ``grid[i]`` (integer) selects row ``i`` (a single point) and
          returns a grid with shape ``(1, n_coords)``.
        * ``grid[:, columns]`` or ``grid[columns]`` (non-integer) selects
          a subset of coordinates (columns).
        * ``grid[rows, columns]`` selects both a subset of rows and a
          subset of columns.

        Parameters
        ----------
        idx :
            Index or index tuple. If a tuple, it must be of length 2 and
            interpreted as ``(rows, columns)`` for standard 2D indexing.

        Returns
        -------
        Grid
            New :class:`Grid` containing the selected subgrid.

        Raises
        ------
        ValueError
            If a tuple index does not have length 2.
        """
        # 2D indexing: (rows, cols)
        if isinstance(idx, tuple):
            if len(idx) != 2:
                raise ValueError("Index must be a tuple of two indices.")
            row_idx, col_idx = idx
            grid = self.grid[row_idx, col_idx]
            coords = self.coords[col_idx]

            grid = np.array(grid)
            coords = np.array(coords)

            # Ensure coords is 1D
            if coords.ndim == 0:
                coords = coords.reshape(1)

            # Normalize grid to 2D:
            if grid.ndim == 0:
                # Single scalar -> 1x1
                grid = grid.reshape(1, 1)
            elif grid.ndim == 1:
                # One dimension collapsed: decide whether it's a row or column
                if isinstance(col_idx, (int, np.integer)):
                    # One column, multiple rows -> column vector
                    grid = grid.reshape(-1, 1)
                else:
                    # One row, multiple columns -> row vector
                    grid = grid.reshape(1, -1)

            return Grid(grid, coords)

        # 1D indexing:
        if isinstance(idx, (int, np.integer)):
            # Select a single row (point), keep all coordinates
            grid = np.array(self.grid[idx])
            if grid.ndim == 1:
                grid = grid.reshape(1, -1)
            return Grid(grid, self.coords)

        # Non-integer 1D index: select columns (coordinates)
        grid = np.array(self.grid[:, idx])
        coords = np.array(self.coords[idx])

        if grid.ndim == 1:
            grid = grid.reshape(-1, 1)
        if coords.ndim == 0:
            coords = coords.reshape(1)

        return Grid(grid, coords)

    def evaluate(self, func, **kwargs) -> np.ndarray:
        r"""
        Evaluate a function on each grid point.

        Parameters
        ----------
        func :
            Callable with signature ``f(point, *args, **kwargs)`` taking a
            1D array (a row of :attr:`grid`) and returning a scalar or
            array-like value.
        **kwargs :
            Additional keyword arguments forwarded to ``func`` via
            :func:`numpy.apply_along_axis`.

        Returns
        -------
        ndarray
            Array of function values, with shape determined by
            :func:`numpy.apply_along_axis`.
        """
        return np.apply_along_axis(func, 1, self.grid, **kwargs)

    def transform(self, func) -> "Grid":
        r"""
        Apply a pointwise transformation to the grid.

        Parameters
        ----------
        func :
            Callable with signature ``f(point)`` taking a 1D array and
            returning a transformed 1D array of the same length.

        Returns
        -------
        Grid
            New :class:`Grid` whose points are obtained by applying
            ``func`` to each row of :attr:`grid`, with the same
            :attr:`coords`.
        """
        new_grid = np.apply_along_axis(func, 1, self.grid)
        return Grid(new_grid, self.coords)

    def __add__(self, other: "Grid") -> "Grid":
        r"""
        Direct sum (row-wise concatenation) of two grids.

        Parameters
        ----------
        other :
            Another :class:`Grid` with the same coordinates.

        Returns
        -------
        Grid
            New :class:`Grid` containing all points from both grids.

        Raises
        ------
        ValueError
            If the coordinate sets differ.
        """
        if len(self.coords) != len(other.coords):
            raise ValueError("Number of coordinates does not match.")

        if not np.array_equal(self.coords, other.coords):
            raise ValueError(
                f"Coordinates do not match: {self.coords} | {other.coords}."
            )

        combined = np.concatenate((self.grid, other.grid), axis=0)
        return Grid(combined, self.coords)

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def shape(self) -> Tuple[int, int]:
        r"""
        Return the shape of the underlying grid array.

        Returns
        -------
        tuple of int
            ``(n_points, n_coords)``.
        """
        return self.grid.shape

    def num_points(self) -> int:
        r"""
        Number of points (rows) in the grid.

        Returns
        -------
        int
            Number of rows in :attr:`grid`.
        """
        return self.shape()[0]

    def num_coords(self) -> int:
        r"""
        Number of coordinates (columns) in the grid.

        Returns
        -------
        int
            Number of columns in :attr:`grid`.
        """
        return self.shape()[1]


# ----------------------------------------------------------------------
# Grid combinations and TN grid construction
# ----------------------------------------------------------------------
def cartesian_product(grids: Sequence[Grid]) -> Grid:
    r"""
    Compute the Cartesian product of a sequence of grids.

    This is defined recursively via the ``@`` operator on :class:`Grid`
    instances and is equivalent to applying :func:`_cartesian_product`
    pairwise.

    Parameters
    ----------
    grids :
        Sequence of :class:`Grid` objects.

    Returns
    -------
    Grid
        Single :class:`Grid` representing the Cartesian product of all
        input grids.
    """
    if len(grids) == 1:
        return grids[0]
    return grids[0] @ cartesian_product(grids[1:])


def direct_sum(grids: Sequence[Grid]) -> Grid:
    r"""
    Compute the direct sum (row-wise concatenation) of a sequence of grids.

    Parameters
    ----------
    grids :
        Sequence of :class:`Grid` objects.

    Returns
    -------
    Grid
        Single :class:`Grid` containing all points from all input grids.
    """
    if len(grids) == 1:
        return grids[0]
    return grids[0] + direct_sum(grids[1:])


def build_node_grid(graph: Any) -> None:
    r"""
    Build and attach node grids based on incident edge grids.

    For each non-leaf node in the tensor network graph ``G``, this function
    collects the grids associated with its incoming edges, forms their
    Cartesian product, and stores the resulting grid under
    ``G.nodes[node]["grid"]``.

    Parameters
    ----------
    graph :
        Tensor-network-like graph object. It is expected to provide:

        * ``graph.nodes`` (node attribute mapping),
        * ``graph.in_edges(node)`` (incoming edges iterator),
        * ``is_leaf_node(node, graph)`` helper, and
        * :func:`collect` to gather edge attributes.
    """
    for node in graph.nodes:
        if is_leaf_node(node, graph):
            continue
        edges = graph.in_edges(node)
        pre_grids: List[Grid] = collect(graph, edges, "grid")
        grid = cartesian_product(pre_grids).permute()
        graph.nodes[node]["grid"] = grid


def tensor_network_grid(
    graph: Any,
    primitive_grid: Sequence[np.ndarray],
    start_grid: np.ndarray | None = None,
) -> Any:
    r"""
    Initialize edge and node grids on a tensor network graph.

    This routine attaches :class:`Grid` objects to all edges of a
    tensor-network-like graph and then builds node grids from incident
    edge grids:

    * For each leaf edge, a :class:`Grid` is built directly from the
      corresponding 1D primitive grid. The edge's rank ``"r"`` is
      **synchronized** to the number of points in this primitive grid,
      i.e. ``edge["r"] = edge["grid"].num_points()``.
    * For each internal edge, a grid is constructed as a random
      subset of the Cartesian product of its predecessor edges' grids,
      with the number of points equal to the edge rank ``"r"``.
      Optionally, points can be taken from a global ``start_grid``
      instead of sampled randomly.
    * Finally, for each non-leaf node, a node grid is built as the
      Cartesian product of its incoming edge grids via
      :func:`build_node_grid`.

    Parameters
    ----------
    graph :
        Tensor-network-like graph object. Expected to support the helpers
        :func:`up_leaves`, :func:`sweep`, :func:`pre_edges`, and
        :func:`collect`, and to carry edge attributes:

        * ``"coordinate"`` on leaf edges (integer index into
          ``primitive_grid``),
        * ``"r"`` on internal edges (desired rank / number of grid points).
    primitive_grid :
        Sequence of primitive 1D grids, one per coordinate. The element
        ``primitive_grid[k]`` must be indexable and suitable as input to
        :class:`Grid` for coordinate ``k``.
    start_grid :
        Optional 2D array providing an initial global grid to be sampled
        instead of random selection for internal edges. For every internal
        edge with rank ``r``, it must hold that
        ``start_grid.shape[0] >= r``. If provided, the first ``r`` rows of
        ``start_grid`` are used, reordered according to the coordinate
        ordering of each internal edge grid.

    Returns
    -------
    graph :
        The same graph object with additional ``"grid"`` attributes attached
        to edges and nodes. Leaf edges also have their rank ``"r"``
        synchronized with the number of primitive grid points.

    Notes
    -----
    This function overwrites the ``r`` attribute of leaf edges to
    ensure consistency between the leaf rank and the size of the attached
    primitive grid. Internal edge ranks are left unchanged.
    """
    # ------------------------------------------------------------------
    # Leaf edge grids: attach primitive grids and synchronize ranks
    # ------------------------------------------------------------------
    for leaf in sorted(up_leaves(graph)):
        coord = graph.edges[leaf]["coordinate"]

        # Build the leaf grid from the corresponding primitive 1D grid
        leaf_grid = Grid(primitive_grid[coord], coord)
        graph.edges[leaf]["grid"] = leaf_grid

        # IMPORTANT: keep the leaf rank consistent with the number of points
        graph.edges[leaf]["r"] = leaf_grid.num_points()

    # ------------------------------------------------------------------
    # Internal edge grids: random subsets / start_grid-based subsets
    # ------------------------------------------------------------------
    for edge in sweep(graph, include_leaves=False):
        rank = graph.edges[edge]["r"]
        predecessor_edges = pre_edges(graph, edge, remove_flipped=True)
        predecessor_grids: List[Grid] = collect(graph, predecessor_edges, "grid")

        next_grid = cartesian_product(predecessor_grids).random_subset(rank)

        if start_grid is not None:
            if start_grid.shape[0] < rank:
                raise ValueError(
                    f"start_grid must have at least r={rank} rows, "
                    f"got {start_grid.shape[0]}."
                )
            # Align columns with the coordinate order of next_grid
            next_grid.grid = start_grid[:rank, next_grid.coords]

        graph.edges[edge]["grid"] = next_grid

    # ------------------------------------------------------------------
    # Node grids from incident edge grids
    # ------------------------------------------------------------------
    build_node_grid(graph)
    return graph


def transform_node_grid(graph: Any, q_to_x) -> Any:
    r"""
    Apply a pointwise transformation to node grids.

    Parameters
    ----------
    graph :
        Tensor-network-like graph object with node attribute ``"grid"``.
    q_to_x :
        Callable mapping a grid point (1D array) to another 1D array of
        the same length. It is passed to :meth:`Grid.transform`.

    Returns
    -------
    graph :
        The same graph object with transformed node grids.
    """
    for node in graph.nodes:
        if node < 0:
            # Conventionally skip "virtual" / auxiliary nodes
            continue
        graph.nodes[node]["grid"] = graph.nodes[node]["grid"].transform(q_to_x)
    return graph


# ----------------------------------------------------------------------
# Linear algebra helpers
# ----------------------------------------------------------------------
def regularized_inverse(
    matrix: np.ndarray,
    lambda_reg: float,
    eps: float = 1e-15,
) -> np.ndarray:
    r"""
    Tikhonov-regularized inverse of a matrix via SVD.

    We compute a stabilized pseudo-inverse of ``matrix`` using

    .. math::

        \sigma_\text{inv} = \frac{\sigma}{\sigma^2 + \alpha},

    where :math:`\sigma` are the singular values of ``matrix`` and

    .. math::

        \alpha = (\lambda_\text{eff} \cdot \sigma_{\max})^2, \quad
        \lambda_\text{eff} = \max(\lambda_\text{reg}, \varepsilon).

    The denominator is clamped to be at least ``eps`` to avoid
    division by zero or overflow, and all computations are carried
    out in double precision.

    Parameters
    ----------
    matrix :
        Input matrix of shape ``(m, n)``.
    lambda_reg :
        Regularization parameter :math:`\lambda_\text{reg}`. If non-finite
        or non-positive, it is replaced by ``eps``.
    eps :
        Small positive number used as a lower bound for both
        :math:`\lambda_\text{eff}` and the denominator.

    Returns
    -------
    ndarray
        Regularized inverse of shape ``(n, m)``. If ``matrix`` has no singular
        values (empty spectrum), a zero matrix of the appropriate shape
        is returned.
    """
    u_matrix, singular_values, vt_matrix = np.linalg.svd(matrix, full_matrices=False)

    # Ensure float64 for stability
    u_matrix = u_matrix.astype(np.float64, copy=False)
    singular_values = singular_values.astype(np.float64, copy=False)
    vt_matrix = vt_matrix.astype(np.float64, copy=False)

    if singular_values.size == 0:
        # Empty spectrum: return shape-consistent zero "inverse"
        return np.zeros_like(matrix.T, dtype=np.float64)

    sigma_max = float(np.max(singular_values))
    lambda_eff = float(lambda_reg)

    # Enforce strictly positive, finite regularization
    if not np.isfinite(lambda_eff) or lambda_eff <= 0.0:
        lambda_eff = eps

    if not np.isfinite(sigma_max) or sigma_max <= 0.0:
        alpha_reg = lambda_eff * lambda_eff
    else:
        alpha_reg = (lambda_eff * sigma_max) ** 2

    denominator = singular_values * singular_values + alpha_reg
    denominator = np.where(denominator <= eps, eps, denominator)

    singular_values_inv = singular_values / denominator

    # matrix_inv = V * diag(sigma_inv) * U^T
    return (vt_matrix.T * singular_values_inv) @ u_matrix.T


def maxvol_grids(
    tensor: "Tensor",
    graph: Any,
    edge: Iterable[Any],
) -> Tuple[Grid, Any]:
    r"""
    Perform a max-volume selection on a tensor flattened along an edge.

    This helper:

    1. Builds the left grid by taking the Cartesian product of the grids
       on the predecessor edges of ``edge``.
    2. Flattens the tensor ``tensor`` along ``edge`` (using its
       :meth:`flatten` method).
    3. Runs :func:`maxvol` on the resulting matrix to select a subset of
       rows with (approximately) maximal volume.
    4. Computes a Tikhonov-regularized inverse of the selected cross
       matrix and wraps it as a tensor via :class:`Tensor`.

    Parameters
    ----------
    tensor :
        Tensor-like object supporting ``tensor.flatten(edge)`` and resulting
        in a 2D array whose rows correspond to the grid points in
        the Cartesian product of predecessor grids.
    graph :
        Tensor-network-like graph object. Expected to provide the
        helper :func:`pre_edges` and to store grids under edge attribute
        ``"grid"``.
    edge :
        Edge label (or descriptor) along which ``tensor`` is flattened and
        contracted.

    Returns
    -------
    grid_L : Grid
        Subgrid corresponding to the selected max-volume rows.
    cross_inv : Tensor-like
        Regularized inverse of the selected cross matrix, wrapped as a
        tensor.

    Notes
    -----
    The regularization strength in :func:`regularized_inverse` is
    currently hard-coded to ``1e-12`` for numerical stability.
    """
    pre = pre_edges(graph, edge, remove_flipped=True)
    grid_l_list: List[Grid] = collect(graph, pre, "grid")
    grid_l = cartesian_product(grid_l_list)

    mat = tensor.flatten(edge)

    rows, _ = maxvol(mat)

    # Compute cross matrix inverse
    cross_inv_mat = regularized_inverse(mat[rows, :], lambda_reg=1e-12)
    cross_inv = Tensor(cross_inv_mat, [edge, flip(edge)])

    return grid_l[rows, :], cross_inv
