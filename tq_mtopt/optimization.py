"""
Optimization routines for grid-based tensor approximations in PyQuTree.

This module provides:
  - A base Model class defining the optimization API (sweep/optimize).
  - Utilities to evaluate grids of function values and select subsets via max-volume,
    greedy, or linear assignment methods (grid → matrix → grid).
  - Functions to generate candidate grids by "cross"-sampling one or multiple legs.
  - Two model implementations:
      * TensorRankOptimization: CP/Tensor-Rank optimizer.
      * MatrixTrainOptimization: General N-site Matrix-Train optimizer.

Provenance
----------
This module contains code adapted from the pyQuTree package by Roman Ellerbrock
(see Ref. [2]). The underlying library and concepts are described in the QuTree
paper (Ref. [1]). Co-authored by Aleksandr Berezutskii.

References
----------
.. [1] R. Ellerbrock, K. G. Johnson, S. Seritan, H. Hoppe, J. H. Zhang,
       T. Lenzen, T. Weike, U. Manthe, and T. J. Martínez,
       "QuTree: A tree tensor network package",
       *J. Chem. Phys.* **160**(11), 112501 (2024).
       doi:10.1063/5.0180233

.. [2] R. Ellerbrock, *pyQuTree* (software), Python package ``pyQuTree``.
       Source: https://github.com/roman-ellerbrock/pyQuTree
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from tq_mtopt.grid import (
    Grid,
    cartesian_product,
    maxvol_grids,
    regularized_inverse,
    tensor_network_grid,
)
from tq_mtopt.maxvol import maxvol
from tq_mtopt.network import (
    collect,
    flip,
    pre_edges,
    star_sweep,
    sweep,
)
from tq_mtopt.tensor import Tensor


class Model:
    """
    Base class for optimization models. Subclass must implement:
      - sweep(grid, function, epoch) -> (new_grid, aux_data)
      - optionally override optimize(...) to customize training loop.
    Provides a default optimize() that runs sweep() n_epoch times.
    """

    def __init__(self):
        pass

    def sweep(self, grid: Grid, function: Callable, epoch: int):
        """
        Perform one pass of the optimization.
        Should be implemented by subclasses.

        Args:
            grid:    Current skeleton/Grid of points.
            function: User-supplied scalar objective V(x) -> float.
            epoch:   Current epoch index (0-based).

        Returns:
            new_grid: Updated Grid.
            aux_data: Optional auxiliary output (e.g. evaluation matrix).
        """

    def optimize(self, grid: Grid, function: Callable, num_epochs: int) -> Grid:
        """
        Run `sweep` for `n_epochs` iterations starting from `grid`.

        Args:
            grid: Initial Grid.
            function: Objective callable.
            num_epochs: Number of sweeps to perform.

        Returns:
            Final updated Grid after n_epoch sweeps.
        """
        for epoch in range(num_epochs):
            grid, _ = self.sweep(grid, function, epoch)
        return grid


def evaluate_grid(grid: Grid, function: Callable, dim2: int, **kwargs) -> np.ndarray:
    """
    Evaluate `function` on each point in `grid`, then reshape to a (dim2 x dim1) matrix.

    Args:
        grid: Grid of shape (r*dim2 x f).
        function: Callable; grid.evaluate(V) returns flat array length r*dim2.
        dim2: Number of rows in the resulting matrix.

    Returns:
        vmat:   Numpy array shape (dim2, dim1).
    """
    vmat = grid.evaluate(function, **kwargs)
    dim1 = int(vmat.size / dim2)
    return vmat.reshape(dim1, dim2).T


def random_points(primitive_grid: list[Grid], r: int, seed: int = 42) -> np.ndarray:
    """
    Sample r random points from each 1D primitive, returning an (r x f) array.
    """
    random.seed(seed)
    x = []
    for g in primitive_grid:
        pts = random.sample(list(g.grid.flatten()), r)
        x.append(pts)
    return np.array(x).T


def random_grid_points(
    primitive_grids: list[Grid], n_samples: int, seed: int = 42
) -> Grid:
    """
    Sample n_samples unique points from the Cartesian product of the primitive grids

    Returns:
      Grid of shape (n_samples x n_primitives) with coords [0..n_primitives-1].
    """

    def sample_unique_indices(
        grid_sizes: list[int], n_samples: int, seed: int
    ) -> set[tuple[int, ...]]:
        rng = np.random.default_rng(seed)
        sampled_indices = set()
        while len(sampled_indices) < n_samples:
            indices = tuple(rng.integers(grid_sizes))
            sampled_indices.add(indices)
        return sampled_indices

    def indices_to_grid_points(indices, grids):
        return np.array(
            [[grids[d].grid[i, 0] for d, i in enumerate(pt)] for pt in indices]
        )

    grid_sizes = [int(grid.grid.shape[0]) for grid in primitive_grids]

    # NOTE: for QTT, the number of primitives can be large (e.g., D*L=160),
    # and the total number of grid points is base^(D*L), which overflows
    # int64 if computed via numpy. We only need to know whether
    # total_combinations is smaller than n_samples (typically tiny), so we
    # compute the product in Python integers and stop early once it is clear.
    total_combinations = 1
    product_complete = True
    for s in grid_sizes:
        total_combinations *= s
        if total_combinations > n_samples:
            # We already know the full product exceeds n_samples.
            product_complete = False
            break

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if product_complete:
        # Exact total_combinations is known (and <= n_samples).
        if n_samples > total_combinations:
            raise ValueError(
                f"Cannot sample {n_samples} unique points from {total_combinations} total combinations"
            )
        if n_samples == total_combinations:
            return cartesian_product(primitive_grids)

    sampled_index_tuples = sample_unique_indices(grid_sizes, n_samples, seed)
    coords = indices_to_grid_points(sampled_index_tuples, primitive_grids)
    return Grid(coords, list(range(len(grid_sizes))))


def maxvol_selection(grid: Grid, function: Callable, dim2: int, **kwargs):
    """
    Select rows of the (dim2 x dim1) evaluation matrix that maximize volume.

    Uses the `maxvol` algorithm.

    Returns:
        grid: Grid reduced to the selected rows.
        vmat:  The full evaluation matrix before selection.
    """
    vmat = evaluate_grid(grid, function, dim2, **kwargs)
    nidx, R = maxvol(vmat)
    grid.grid = grid.grid[nidx, :]
    return grid, vmat


def assignment_selection(grid: Grid, function: Callable, dim2: int, **kwargs):
    """
    Select rows by solving a linear assignment problem on the cost matrix.

    Args:
        grid: Grid of candidate points.
        function: Objective callable.
        dim2: Number of rows in cost matrix.

    Returns:
        grid: Updated Grid with selected points.
        vmat:  Cost matrix used for assignment.
    """
    vmat = evaluate_grid(grid, function, dim2, **kwargs)
    rows, cols = linear_sum_assignment(vmat)
    # map (row_i, col_j) back to flat index
    idcs = np.ravel_multi_index(
        (cols, rows), vmat.T.shape
    )  # NOTE: evaluate_grid transposes
    grid.grid = grid.grid[idcs, :]
    return grid, vmat


def rest(idx: list[int], grid: Grid) -> list[int]:
    """
    Given a list of column indices `idx`, return the complement in [0..D-1].
    """
    all_cols = set(range(grid.coords.shape[0]))
    return sorted(all_cols - set(idx))


def greedy_column_min(matrix: np.ndarray) -> tuple[list[int], list[int]]:
    """
    Greedily select one minimum per column.

    Returns (rows, cols).
    """
    cols = matrix.argmin(axis=1)
    rows = list(range(matrix.shape[0]))
    return rows, cols


def greedy_selection(grid: Grid, function: Callable, r: int, **kwargs):
    """
    Select grid points by choosing minimum in each column.

    Returns:
        grid: Updated Grid.
        vmat: Evaluation matrix used.
    """
    vmat = evaluate_grid(grid, function, r, **kwargs)
    rows, cols = greedy_column_min(vmat)
    idcs = np.ravel_multi_index(
        (cols, rows), vmat.T.shape
    )  # NOTE: evaluate_grid transposes
    grid.grid = grid.grid[idcs, :]
    return grid, vmat


def recombination(grid: Grid, idxs: list[int]) -> Grid:
    """
    Recombine mutually exclusive grid components via cartesian product.

    Args:
        grid: Original Grid.
        idxs: Indices to keep.

    Returns:
        New permuted Grid of candidates.
    """
    a = grid[idxs]
    b = grid[rest(idxs, grid)]
    return cartesian_product([a, b]).permute()


def create_mutations(grid: Grid, replacement_grid: Grid) -> tuple[Grid, Grid]:
    """
    Cross-sample one leg: for each point in replacement_grid,
    hold the other coordinates fixed.

    Returns:
      candidates: Grid of shape (r*Nr x f).
      kept:       Grid of shape (r x f-1).
    """
    b = replacement_grid
    a = grid[rest(b.coords, grid)]
    c = cartesian_product([a, b]).permute()
    return c, a


def create_mutations_multi(skel: Grid, leg_grids: list[Grid]) -> tuple[Grid, Grid]:
    """
    Cross-sample all k legs simultaneously:
    - skel has shape (r x k)
    - leg_grids is list of k Grids (Ni x 1)

    Returns:
      candidates: Grid (r * ∏ Ni x k)
      kept:       Empty Grid (r x 0) as placeholder.
    """
    kept = Grid(np.zeros((skel.num_points(), 0)), np.array([], dtype=int))
    C = cartesian_product([kept] + leg_grids).permute()
    return C, kept


def column_labels(matrix: np.ndarray) -> np.ndarray:
    """
    Compute integer labels for unique columns of `matrix`.
    """
    _, labels = np.unique(matrix, axis=1, return_inverse=True)
    return labels


def greedy_with_group_assignment(
    matrix: np.ndarray, groups: np.ndarray
) -> tuple[list[int], list[int]]:
    """
    Run a per-group linear assignment. If a group has more columns than there are
    available rows (i.e. matrix is rectangular with R < C_group), fall back to a
    per-column argmin within that group (allowing row reuse) so that every column
    receives a valid row index.

    Args:
      matrix: (R x C) cost matrix.
      groups: length-C array assigning each column to a group.

    Returns:
      rows: selected row indices per column (length C).
      cols: column indices (0..C-1) in order.
    """
    matrix = np.asarray(matrix)
    R, C = matrix.shape
    selected_rows = np.full(C, -1, dtype=int)

    for g in np.unique(groups):
        cols_g = np.flatnonzero(groups == g)
        sub = matrix[:, cols_g]

        if sub.shape[0] >= sub.shape[1]:
            # Standard one-to-one assignment inside the group.
            rows_g, cols_sub = linear_sum_assignment(sub)
            selected_rows[cols_g[cols_sub]] = rows_g
        else:
            # Not enough rows to assign uniquely: pick best row per column.
            selected_rows[cols_g] = np.argmin(sub, axis=0)

    return list(selected_rows), list(range(C))


def group_assignment(
    grid: Grid, function: Callable, groups: np.ndarray, r: int, **kwargs
):
    """
    Grouped selection: select one row per column group via greedy_with_group_assignment.

    Returns:
      grid: Updated Grid.
      vmat: Evaluation matrix used.
    """
    vmat = evaluate_grid(grid, function, r, **kwargs)
    rows, cols = greedy_with_group_assignment(vmat, groups)
    idcs = np.ravel_multi_index((cols, rows), vmat.T.shape)
    grid.grid = grid.grid[idcs, :]
    return grid, vmat


def variation_update(
    grid: Grid, replacement_grid: Grid, function: Callable, **kwargs
) -> tuple[Grid, np.ndarray]:
    """
    One cross-update on a single physical leg:
      grid --create_mutations--> candidates
           --group_assignment--> new grid
    """
    ngrid, a = create_mutations(grid, replacement_grid)
    groups = column_labels(a.grid.T)
    return group_assignment(
        ngrid, function, groups, replacement_grid.num_points(), **kwargs
    )


def recombination_update(
    grid: Grid, idxs: list[int], function: Callable, **kwargs
) -> tuple[Grid, np.ndarray]:
    """
    Recombine two subsets via max-volume selection.
    """
    ngrid = recombination(grid, idxs)
    return maxvol_selection(ngrid, function, grid.num_points(), **kwargs)


def recombination_update_assignment(
    grid: Grid, left_block_cols: list[int], function: Callable, **kwargs
):
    """
    Recombine two blocks A (columns=left_block_cols) and B (the rest).
    - cartesian_product(A,B) -> candidate grid with r*r rows
    - evaluate -> r x r cost matrix (rows=r 'slots', cols=r candidates)
    - Hungarian assignment -> pick r rows for the next grid

    Returns:
      new_grid, vmat  with vmat.shape == (r, r)
    """
    ngrid = recombination(grid, left_block_cols)  # r^2 x f
    new_grid, vmat = assignment_selection(
        ngrid, function, dim2=grid.num_points(), **kwargs
    )
    return new_grid, vmat


class TensorRankOptimization(Model):
    """
    Tensor-Rank (PARAFAC) optimizer: performs one-leg cross-updates
    in a CP-format tree tensor network (TTNcross).

    Args:
      primitive_grids: list of f Grids, one per each dimension.

    Notes
    -----
    The effective "rank" is defined by the current skeleton grid:
        r := grid.num_points()
    and is preserved by the update rules.
    """

    def __init__(self, primitive_grids: list[Grid]):
        self.data(primitive_grids)

    def data(self, primitive_grids: list[Grid]):
        """
        Args:
          primitive_grids: list of f one-dimensional Grids (Ni x 1)
        """
        self.primitive_grids = primitive_grids

    def sweep(self, grid: Grid, function: Callable, epoch: int):
        """
        One sweep: for each dimension k, perform `variation_update`.

        Returns:
          (new_grid, last_vmat)
        """
        vmat = None
        f = grid.num_coords()
        for k in range(f):
            grid, vmat = variation_update(
                grid, self.primitive_grids[k], function, epoch=epoch
            )
        return grid, vmat


class MatrixTrainOptimization(Model):
    """
    Optimizer for an N-site Matrix Train.

    Args:
        primitive_grids: list of N one-dimensional Grids (Ni x 1).

    Notes
    -----
    The effective "rank" is defined by the current skeleton grid:
        r := grid.num_points()
    and is preserved by the update rules.
    """

    def __init__(self, primitive_grids: list[Grid]):
        self.data(primitive_grids)

    def data(self, primitive_grids: list[Grid]):
        """
        Args:
          primitive_grids: list of N one-dimensional Grids (Ni x 1)
        """
        self.primitive_grids = primitive_grids
        self.N = len(primitive_grids)

    def sweep(self, grid: Grid, function: Callable, epoch: int):
        vmat = None
        f = grid.num_coords()

        for k in range(min(2, f)):
            grid, _ = variation_update(
                grid, self.primitive_grids[k], function, epoch=epoch
            )

        for k in range(2, f - 1):
            left_block = list(range(k))
            grid, vmat = recombination_update_assignment(
                grid, left_block, function, epoch=epoch
            )
            grid, vmat = variation_update(
                grid, self.primitive_grids[k], function, epoch=epoch
            )

        if f > 3:
            for k in range(f - 1, f):
                grid, vmat = variation_update(
                    grid, self.primitive_grids[k], function, epoch=epoch
                )

        return grid, vmat


class OptimizationLogger:
    r"""
    Simple pandas-based logger for optimization iterations.

    The logger stores each record (a mapping of column names to values)
    in an internal :class:`pandas.DataFrame`. It additionally provides a
    string representation that highlights the row with minimal objective
    value in the column ``"f"``.

    Attributes
    ----------
    dataframe : pandas.DataFrame
        Accumulated log of all records.
    """

    def __init__(self) -> None:
        self._records: list = []
        self._dataframe_cache: Optional[pd.DataFrame] = None

    @property
    def dataframe(self) -> pd.DataFrame:
        if self._dataframe_cache is None:
            if self._records:
                self._dataframe_cache = pd.DataFrame(self._records)
            else:
                self._dataframe_cache = pd.DataFrame()
        return self._dataframe_cache

    def __call__(self, record: Mapping[str, Any]) -> None:
        r"""
        Append a single record to the internal dataframe.

        Parameters
        ----------
        record :
            Mapping from column name to value. Typical keys include
            ``"x1"``, ``"x2"``, ..., and ``"f"`` for the objective.
        """
        self._records.append(dict(record))
        self._dataframe_cache = None

    def __str__(self) -> str:  # pragma: no cover - mostly formatting
        df = self.dataframe
        if "f" not in df.columns or df.empty:
            return "No logged records."

        best_idx = df["f"].idxmin()
        best_row = df.loc[best_idx]
        return f"Optimal value:\n{best_row}"


def numpy_array_to_tuple(
    array: np.ndarray,
    precision: int = 8,
) -> Tuple[float, ...]:
    r"""
    Convert a NumPy array to a rounded tuple for use as a cache key.

    Parameters
    ----------
    array :
        Input array-like object. It is flattened before rounding.
    precision :
        Number of decimal places to keep when rounding.

    Returns
    -------
    tuple of float
        Rounded entries of ``array`` as a 1D tuple.
    """
    arr = np.asarray(array, dtype=float).ravel()
    rounded = np.round(arr, precision)
    return tuple(float(x) for x in rounded)


class Objective:
    r"""
    Cached and logged objective function wrapper.

    This class wraps a scalar objective function and provides:

    * memoization via a dictionary cache keyed by rounded input points,
    * a simple logging facility (via :class:`OptimizationLogger`) that
      stores function values and arbitrary metadata,
    * a post-processing ``transformer`` applied to raw objective values.

    Parameters
    ----------
    error_fn :
        Callable ``error_fn(x)`` that evaluates the raw objective value
        at a point ``x`` (1D array-like).
    transformer :
        Optional callable ``transformer(f)`` applied to the raw objective
        value before returning. If ``None``, the identity transform is
        used.

    Attributes
    ----------
    error_fn : callable
        Underlying raw objective function.
    transformer : callable
        Transformation applied to the raw objective values.
    logger : OptimizationLogger
        Logger storing evaluation history.
    cache : dict
        Maps rounded-point tuples to raw objective values.
    cache_hits : int
        Number of times a cached value was reused.
    function_calls : int
        Number of actual calls to ``error_fn``.
    """

    def __init__(
        self,
        error_fn: Callable[[np.ndarray], float],
        transformer: Optional[Callable[[float], float]] = None,
    ) -> None:
        self.error_fn = error_fn
        self.transformer = transformer or (lambda value: value)
        self.logger = OptimizationLogger()
        self.cache: Dict[Tuple[float, ...], float] = {}
        self.cache_hits: int = 0
        self.function_calls: int = 0

    def __call__(self, x: np.ndarray, **kwargs: Any) -> Any:
        r"""
        Evaluate the objective at a point OR a batch of points, with caching/logging.

        If ``x`` is 2D of shape (n_points, dim), returns a 1D array of length n_points.
        If ``x`` is 1D, returns a scalar.
        """
        x_arr = np.asarray(x, dtype=float)

        # Batched path
        if x_arr.ndim == 2:
            return self.evaluate_batch(x_arr, **kwargs)

        return self.evaluate_point(x_arr, **kwargs)

    def __str__(self) -> str:  # pragma: no cover - mostly formatting
        num_function = self.function_calls
        num_cache = self.cache_hits
        total_calls = num_function + num_cache

        return (
            f"{self.logger}\n\n"
            f"Number of objective function calls: {num_function}\n"
            f"Number of cached function accesses: {num_cache}\n"
            f"Total number of calls: {total_calls}"
        )

    def evaluate_point(self, point: np.ndarray, **kwargs: Any) -> float:
        """Evaluate the objective on a single point, with caching/logging."""
        point = point.ravel()
        key = numpy_array_to_tuple(point)

        if key in self.cache:
            self.cache_hits += 1
            raw_value = self.cache[key]
            return self.transformer(raw_value)

        raw_value = self.error_fn(point)
        transformed_value = self.transformer(raw_value)

        self.cache[key] = raw_value
        self.function_calls += 1

        if kwargs:
            coord_dict = {f"x{i + 1}": key[i] for i in range(len(key))}
            self.logger({**coord_dict, "f": raw_value, **kwargs})

        return transformed_value

    def evaluate_batch(self, points: np.ndarray, **kwargs: Any) -> np.ndarray:
        r"""
        Evaluate the objective on a batch of points (n_points, dim), with caching/logging.

        Deduplicates repeated (rounded) points inside the batch, tries a single vectorized
        call to ``error_fn`` for the remaining unique uncached points, then falls back
        to pointwise evaluation if needed.
        """
        X = np.asarray(points, dtype=float)
        if X.ndim != 2:
            raise ValueError("Batch evaluation expects a 2D array of points.")

        n_points = X.shape[0]
        keys = [numpy_array_to_tuple(X[i]) for i in range(n_points)]
        values = np.empty(n_points, dtype=float)

        # Group indices by key to avoid duplicate evaluations in the same batch
        uncached_groups: Dict[Tuple[float, ...], list[int]] = {}

        # Fill cached values + collect uncached
        for i, key in enumerate(keys):
            if key in self.cache:
                self.cache_hits += 1
                values[i] = self.transformer(self.cache[key])
            else:
                uncached_groups.setdefault(key, []).append(i)

        if uncached_groups:
            unique_keys = list(uncached_groups.keys())
            unique_rep_indices = [uncached_groups[k][0] for k in unique_keys]
            X_unique = X[unique_rep_indices]

            # Try vectorized error_fn on the unique uncached points
            raw_unique = None
            try:
                raw_unique = np.asarray(self.error_fn(X_unique), dtype=float).reshape(
                    -1
                )
                if raw_unique.shape[0] != len(unique_keys):
                    raw_unique = None
            except Exception:
                raw_unique = None

            # Fallback: pointwise error_fn
            if raw_unique is None:
                raw_unique = np.array(
                    [self.error_fn(X[i].ravel()) for i in unique_rep_indices],
                    dtype=float,
                )

            # Cache + broadcast duplicates
            for j, key in enumerate(unique_keys):
                raw_value = float(raw_unique[j])
                transformed_value = self.transformer(raw_value)

                self.cache[key] = raw_value
                self.function_calls += 1

                for i in uncached_groups[key]:
                    values[i] = transformed_value

                if kwargs:
                    coord_dict = {f"x{i + 1}": key[i] for i in range(len(key))}
                    self.logger({**coord_dict, "f": raw_value, **kwargs})

        return values


def tree_tensor_network_optimizer_step(
    graph: nx.DiGraph,
    objective: Objective,
    sweep_id: int,
) -> nx.DiGraph:
    r"""
    Perform a single TTN optimizer sweep over all internal edges.

    For each internal edge (as given by :func:`star_sweep`), this step:

    1. Collects incoming edge grids and forms their Cartesian product.
    2. Evaluates the objective on this grid (with logging metadata
       ``sweep`` and ``node``).
    3. Reshapes values into a tensor with ranks from edge attributes
       ``"r"`` and wraps it as :class:`Tensor`.
    4. Runs :func:`maxvol_grids` to pick a subset of rows and compute
       the corresponding cross inverse.
    5. Stores the updated grids and tensors on the graph.

    Parameters
    ----------
    graph :
        Tensor-network-like directed graph with edge attributes
        ``"grid"`` and ``"r"`` and node attributes used to store
        tensors/grids.
    objective :
        :class:`Objective` wrapper for the scalar function to be
        optimized.
    sweep_id :
        Integer label for the current sweep, forwarded to the logger.

    Returns
    -------
    DiGraph
        A **new** graph instance (copy of ``graph``) with updated
        grids and tensors.
    """
    graph = graph.copy()

    for edge in star_sweep(graph, exclude_leaves=True):
        incoming_edges = list(pre_edges(graph, edge))
        edge_grids = collect(graph, incoming_edges, "grid")
        combined_grid = cartesian_product(edge_grids).permute()
        ranks = collect(graph, incoming_edges, "r")

        kwargs = {"sweep": sweep_id, "node": edge[0]}
        values = combined_grid.evaluate(objective, **kwargs)
        tensor_values = values.reshape(ranks)

        node_tensor = Tensor(tensor_values, incoming_edges)
        next_grid, cross_inv = maxvol_grids(node_tensor, graph, edge)

        # Save results on edge and node
        graph.edges[edge]["grid"] = next_grid
        graph.edges[edge]["A"] = cross_inv
        graph.nodes[edge[0]]["grid"] = combined_grid
        graph.nodes[edge[0]]["A"] = node_tensor

    return graph


def tree_tensor_network_optimize(
    graph: nx.DiGraph,
    objective: Objective,
    num_sweeps: int = 6,
    primitive_grid: Optional[Sequence[np.ndarray]] = None,
    start_grid: Optional[np.ndarray] = None,
) -> nx.DiGraph:
    r"""
    Run tensor-network optimization (TTN-style) over several sweeps.

    Parameters
    ----------
    graph :
        Tensor-network graph (e.g. from
        :func:`tq_mtopt.network.tensor_train_network` or
        :func:`tq_mtopt.network.balanced_tree`), equipped at least with
        edge ranks ``"r"`` and leaf coordinates if
        ``primitive_grid`` is provided.
    objective :
        :class:`Objective` wrapper for the scalar function to be
        optimized.
    num_sweeps :
        Number of global sweeps to perform.
    primitive_grid :
        Optional sequence of 1D primitive grids, one per coordinate.
        If provided, :func:`tn_grid` is called once at the beginning to
        initialize edge and node grids.
    start_grid :
        Optional 2D array used by :func:`tn_grid` as a deterministic
        source of grid points instead of random subsets. See
        :func:`tq_mtopt.grid.tn_grid` for details.

    Returns
    -------
    DiGraph
        Graph with updated grids and tensors after all sweeps.
    """
    graph = graph.copy()

    if primitive_grid is not None:
        graph = tensor_network_grid(graph, primitive_grid, start_grid=start_grid)

    for sweep_index in range(num_sweeps):
        graph = tree_tensor_network_optimizer_step(graph, objective, sweep_index)

    return graph


def tree_tensor_network_cross(
    graph: nx.DiGraph,
    objective: Objective,
) -> nx.DiGraph:
    r"""
    Build CUR-like cross approximation tensors on a tree tensor network.

    This routine assumes that a tree tensor network graph already carries
    edge and node grids (e.g. after :func:`tn_grid` / `tensor_network_grid`)
    and uses them to construct *consistent* tensors that approximate
    the objective function in a TTN/Tree-TN fashion:

    1. For each internal node, evaluate the objective on the Cartesian
       product of its incident edge grids and store the resulting tensor
       under ``node["A"]``.
    2. For each internal edge, do the same for its predecessor edges and
       store the tensor under ``edge["A"]``.
    3. For each internal edge again, take the grids of both directions
       (edge and flipped edge), evaluate the objective on their product,
       and compute a Tikhonov-regularized inverse via
       :func:`regularized_inverse`. This inverse is stored as a tensor
       under ``edge["A"]``.

    Parameters
    ----------
    graph :
        Tensor-network-like :class:`networkx.DiGraph` with edge and node
        grids already attached (e.g. via :func:`tn_grid`).
    objective :
        :class:`Objective` wrapper for the scalar function whose CUR-like
        approximation is being constructed.

    Returns
    -------
    DiGraph
        A **new** graph with tensors ``"A"`` stored both on nodes and
        edges, representing CUR-like cross approximation factors.
    """
    graph = graph.copy()

    # 1. Node tensors
    for node in graph.nodes():
        if node < 0:
            # Skip virtual / auxiliary nodes
            continue

        incoming_edges = list(graph.in_edges(node))
        edge_grids = collect(graph, incoming_edges, "grid")
        combined_grid = cartesian_product(edge_grids).permute()
        ranks = collect(graph, incoming_edges, "r")

        values = combined_grid.evaluate(objective)
        tensor_values = values.reshape(ranks)

        node_tensor = Tensor(tensor_values, incoming_edges)
        graph.nodes[node]["A"] = node_tensor

    # 2. Edge tensors based on predecessors
    for edge in sweep(graph, include_leaves=False):
        incoming_edges = list(pre_edges(graph, edge))
        edge_grids = collect(graph, incoming_edges, "grid")
        combined_grid = cartesian_product(edge_grids).permute()
        ranks = collect(graph, incoming_edges, "r")

        values = combined_grid.evaluate(objective)
        tensor_values = values.reshape(ranks)

        edge_tensor = Tensor(tensor_values, incoming_edges)
        graph.edges[edge]["A"] = edge_tensor

    # 3. Cross tensors as regularized inverses for each edge pair
    for edge in sweep(graph, include_leaves=False):
        edge_pair = [edge, flip(edge)]
        pair_grids = collect(graph, edge_pair, "grid")
        combined_grid = cartesian_product(pair_grids).permute()
        ranks = collect(graph, edge_pair, "r")

        cross_values = combined_grid.evaluate(objective).reshape(ranks)
        cross_inverse = regularized_inverse(cross_values, lambda_reg=1e-12)
        cross_tensor = Tensor(cross_inverse, edge_pair)

        graph.edges[edge]["A"] = cross_tensor

    return graph
