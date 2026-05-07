"""Benchmarking routines for optimization algorithms.

This module runs mtopt-based optimizers (TRC/MTC/TTOpt) and SciPy baselines
(DE/DA) under a shared evaluation-counting protocol.

Numerical-stability choice (TTOpt-style):
- For tensor-cross optimizers, we use a bounded, monotone score mapping based
  on a running best value. This avoids exponential under/overflow and works
  for objectives with negative minima.
- For SciPy baselines, we minimize the *raw* objective directly (no score
  transform), matching the common benchmarking practice in TTOpt.

All routines return: (objective_calls, best_f_raw, hits_dict, objective_instance).

hits_dict maps each threshold tau to the *true* first-hit call count (1-based)
for eps_best <= tau, with eps_best = |f_best - f_opt| and f_best the running
minimum of raw objective values.
"""

from __future__ import annotations

from typing import Callable, List, Tuple
import random as pyrandom
import numpy as np
from scipy.optimize import differential_evolution, dual_annealing

from threshold_logger import RawEvalLogger, first_hit_calls

from tq_mtopt.network import tensor_train_graph
from tq_mtopt.grid import Grid
from tq_mtopt.optimization import (
    tree_tensor_network_optimize,
    MatrixTrainOptimization,
    TensorRankOptimization,
    random_grid_points,
    Objective,
)

from tq_mtopt.qtt import (
    QTTDecoder,
    qtt_primitive_arrays,
    qtt_primitive_grids,
    qtt_total_cores,
    qtt_z_permuted_coordinate_map,
    make_qtt_objective,
)


def _seed_all(seed: int) -> None:
    """Seed all RNGs for reproducibility."""
    np.random.seed(seed)
    pyrandom.seed(seed)


def _dim(bounds: List[Tuple[float, float]]) -> int:
    """Problem dimension from bounds."""
    return len(bounds)


def _budget(
    num_grid_points: int,
    rank: int,
    num_sweeps: int,
    D: int,
    *,
    grid_type: str = "plain",
    qtt_levels: int = 16,
    qtt_base: int = 2,
) -> int:
    """Heuristic evaluation budget for SciPy baselines.

    We aim for the same order of magnitude of objective calls as TRC/MTC.

    - plain: budget ~ r * N * sweeps * D
    - qtt  : budget ~ r * base * sweeps * (D*L)   (effective dimension D_eff = D*L)
    """
    if grid_type == "plain":
        eff_n = int(num_grid_points)
        eff_D = int(D)
    elif grid_type == "qtt":
        eff_n = int(qtt_base)
        eff_D = int(D) * int(qtt_levels)
    else:
        raise ValueError("grid_type must be 'plain' or 'qtt'")

    return max(100, int(rank) * eff_n * int(num_sweeps) * eff_D)


def _snap_to_grid(xi: float, g: np.ndarray) -> float:
    """Snap a scalar coordinate to the nearest point in a sorted 1-D grid array."""
    if len(g) == 1 or g[-1] == g[0]:
        return float(g[0])
    idx = int(np.round((xi - g[0]) / (g[-1] - g[0]) * (len(g) - 1)))
    idx = max(0, min(len(g) - 1, idx))
    return float(g[idx])


def _make_snap_fn(
    bounds: List[Tuple[float, float]],
    num_grid_points: int,
    *,
    grid_type: str = "plain",
    qtt_levels: int = 16,
    qtt_base: int = 2,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function that snaps a continuous point to the nearest discrete grid point.

    For 'plain': each coordinate is snapped to the nearest point in
        linspace(lo, hi, num_grid_points).
    For 'qtt': each coordinate is snapped to the nearest point in
        linspace(lo, hi, qtt_base**qtt_levels), which is the physical grid
        implied by the QTT discretization.

    The returned snap function is kept separate from the objective so that
    Objective's cache can key on the already-snapped coordinates. This ensures
    that two different continuous proposals mapping to the same grid point are
    treated as one evaluation, matching the deduplication that tensor methods
    get for free by only ever proposing exact grid points.
    """
    if grid_type == "plain":
        grids = [
            np.linspace(lo, hi, num_grid_points, endpoint=True) for (lo, hi) in bounds
        ]

        def snap(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            return np.array([_snap_to_grid(xi, g) for xi, g in zip(x, grids)])

        return snap

    if grid_type == "qtt":
        # Avoid materializing base**levels points per dimension.
        # For a uniform grid with n_pts points on [lo, hi], the nearest grid
        # point to xi is lo + idx*(hi-lo)/(n_pts-1) where
        # idx = round(clip((xi-lo)/(hi-lo) * (n_pts-1), 0, n_pts-1)).
        n_pts = int(qtt_base) ** int(qtt_levels)

        def snap(x: np.ndarray) -> np.ndarray:  # type: ignore[misc]
            x = np.asarray(x, dtype=float)
            out = np.empty_like(x)
            for i, (xi, (lo, hi)) in enumerate(zip(x, bounds)):
                if n_pts == 1 or hi == lo:
                    out[i] = lo
                else:
                    idx = int(
                        np.round(
                            np.clip((xi - lo) / (hi - lo) * (n_pts - 1), 0, n_pts - 1)
                        )
                    )
                    out[i] = lo + idx * (hi - lo) / (n_pts - 1)
            return out

        return snap

    raise ValueError("grid_type must be 'plain' or 'qtt'")


def make_primitives(
    bounds: List[Tuple[float, float]],
    num_grid_points: int,
    *,
    grid_type: str = "plain",
    qtt_levels: int = 16,
    qtt_base: int = 2,
) -> list[Grid]:
    """Create list of 1D grids (primitives) for given bounds.

    - grid_type='plain': D primitives, each linspace over bounds.
    - grid_type='qtt'  : D*qtt_levels digit primitives, each {0,...,qtt_base-1}.
    """
    if grid_type == "plain":
        return [
            Grid(np.linspace(lo, hi, num_grid_points, endpoint=True), [k])
            for k, (lo, hi) in enumerate(bounds)
        ]
    if grid_type == "qtt":
        D = len(bounds)
        return qtt_primitive_grids(num_vars=D, levels=qtt_levels, base=qtt_base)
    raise ValueError("grid_type must be 'plain' or 'qtt'")


def _make_ttopt_score_transform(mode: str) -> Callable[[float], float]:
    """Return a TTOpt-style bounded monotone score mapping.

    TTOpt uses a running best value J_min and maps raw objective y to
        z = pi/2 - arctan(y - J_min),
    which is bounded and numerically stable.

    - mode='max': return z  (larger is better)  -> suitable for maximizers.
    - mode='min': return -z (smaller is better) -> suitable for minimizers.

    The transform is stateful and must be re-created per independent run.
    """
    if mode not in {"min", "max"}:
        raise ValueError("mode must be 'min' or 'max'")

    jmin = {"val": float("inf")}

    if mode == "max":

        def score(y: float) -> float:
            yy = float(y)
            if yy < jmin["val"]:
                jmin["val"] = yy
            return float(np.pi / 2.0 - np.arctan(yy - jmin["val"]))

    else:

        def score(y: float) -> float:
            yy = float(y)
            if yy < jmin["val"]:
                jmin["val"] = yy
            return float(-(np.pi / 2.0 - np.arctan(yy - jmin["val"])))

    return score


def run_trc(
    func,
    bounds,
    num_grid_points,
    rank,
    num_sweeps,
    seed: int = 42,
    *,
    grid_type: str = "plain",
    qtt_levels: int = 16,
    qtt_base: int = 2,
    qtt_z: int = 3,
    f_opt: float | None = None,
    thresholds: tuple[float, ...] = (1e-2, 1e-3, 1e-4, 1e-5),
):
    """Run Tensor-Rank Cross (TRC)."""
    _seed_all(seed)
    primitives = make_primitives(
        bounds,
        num_grid_points,
        grid_type=grid_type,
        qtt_levels=qtt_levels,
        qtt_base=qtt_base,
    )

    if grid_type == "qtt":
        decoder = QTTDecoder(
            num_vars=len(bounds), levels=qtt_levels, base=qtt_base, bounds=bounds
        )
        func = make_qtt_objective(func, decoder)

    traced = RawEvalLogger(func)

    # TRC is implemented as a minimizer -> use 'min' score.
    obj = Objective(traced, _make_ttopt_score_transform(mode="min"))

    model = TensorRankOptimization(primitives)
    grid_start = random_grid_points(primitives, rank, seed)
    _ = model.optimize(grid_start, obj, num_sweeps)

    calls = traced.calls
    f_min = float(np.min(traced.values)) if traced.values else np.nan
    hits = first_hit_calls(
        traced.values, float(f_opt) if f_opt is not None else np.nan, thresholds
    )
    return calls, f_min, hits, obj


def run_trc_z(
    func,
    bounds,
    num_grid_points,
    rank,
    num_sweeps,
    seed: int = 42,
    *,
    grid_type: str = "plain",
    qtt_levels: int = 16,
    qtt_base: int = 2,
    qtt_z: int = 3,
    f_opt: float | None = None,
    thresholds: tuple[float, ...] = (1e-2, 1e-3, 1e-4, 1e-5),
):
    """Run Tensor-Rank Cross with z-permuted QTT digit ordering (TRC-z).

    Identical to TRC but the QTT digit coordinates are interleaved in groups
    of ``qtt_z`` variables (z-curve / Morton ordering). Only active when
    ``grid_type='qtt'``; falls back to standard TRC for ``grid_type='plain'``.
    """
    _seed_all(seed)
    primitives = make_primitives(
        bounds,
        num_grid_points,
        grid_type=grid_type,
        qtt_levels=qtt_levels,
        qtt_base=qtt_base,
    )

    if grid_type == "qtt":
        perm = qtt_z_permuted_coordinate_map(
            num_vars=len(bounds), levels=qtt_levels, z=qtt_z
        )
        decoder = QTTDecoder(
            num_vars=len(bounds),
            levels=qtt_levels,
            base=qtt_base,
            bounds=bounds,
            permutation=perm,
        )
        func = make_qtt_objective(func, decoder)

    traced = RawEvalLogger(func)
    obj = Objective(traced, _make_ttopt_score_transform(mode="min"))

    model = TensorRankOptimization(primitives)
    grid_start = random_grid_points(primitives, rank, seed)
    _ = model.optimize(grid_start, obj, num_sweeps)

    calls = traced.calls
    f_min = float(np.min(traced.values)) if traced.values else np.nan
    hits = first_hit_calls(
        traced.values, float(f_opt) if f_opt is not None else np.nan, thresholds
    )
    return calls, f_min, hits, obj


def run_mtc(
    func,
    bounds,
    num_grid_points,
    rank,
    num_sweeps,
    seed: int = 42,
    *,
    grid_type: str = "plain",
    qtt_levels: int = 16,
    qtt_base: int = 2,
    qtt_z: int = 3,
    f_opt: float | None = None,
    thresholds: tuple[float, ...] = (1e-2, 1e-3, 1e-4, 1e-5),
):
    """Run Matrix-Train Cross (MTC)."""
    _seed_all(seed)
    primitives = make_primitives(
        bounds,
        num_grid_points=num_grid_points,
        grid_type=grid_type,
        qtt_levels=qtt_levels,
        qtt_base=qtt_base,
    )

    if grid_type == "qtt":
        decoder = QTTDecoder(
            num_vars=len(bounds), levels=qtt_levels, base=qtt_base, bounds=bounds
        )
        func = make_qtt_objective(func, decoder)

    traced = RawEvalLogger(func)

    # MTC is implemented as a minimizer -> use 'min' score.
    obj = Objective(traced, _make_ttopt_score_transform(mode="min"))

    model = MatrixTrainOptimization(primitives)
    grid_start = random_grid_points(primitives, rank, seed)
    _ = model.optimize(grid_start, obj, num_sweeps)

    calls = traced.calls
    f_min = float(np.min(traced.values)) if traced.values else np.nan
    hits = first_hit_calls(
        traced.values, float(f_opt) if f_opt is not None else np.nan, thresholds
    )
    return calls, f_min, hits, obj


def run_ttopt(
    func,
    bounds,
    num_grid_points,
    rank,
    num_sweeps,
    seed: int = 42,
    *,
    grid_type: str = "plain",
    qtt_levels: int = 16,
    qtt_base: int = 2,
    qtt_z: int = 3,
    f_opt: float | None = None,
    thresholds: tuple[float, ...] = (1e-2, 1e-3, 1e-4, 1e-5),
):
    """Run TTOpt (tree tensor network optimizer)."""
    _seed_all(seed)
    D = len(bounds)

    if grid_type == "qtt":
        # Digit space: f_eff = sum(levels) dimensions, each digit in {0,...,base-1}
        decoder = QTTDecoder(
            num_vars=D, levels=qtt_levels, base=qtt_base, bounds=bounds
        )
        func_eff = make_qtt_objective(func, decoder)
        f_eff = qtt_total_cores(D, qtt_levels)
        primitive_grid = [
            arr.astype(float) for arr in qtt_primitive_arrays(D, qtt_levels, qtt_base)
        ]
    else:
        func_eff = func
        f_eff = D
        primitive_grid = [
            np.linspace(lo, hi, num_grid_points, endpoint=True) for (lo, hi) in bounds
        ]

    traced = RawEvalLogger(func_eff)

    # TTOpt is implemented as a maximizer -> use 'max' score.
    obj = Objective(traced, _make_ttopt_score_transform(mode="max"))

    ttgraph = tensor_train_graph(f_eff, rank, primitive_grid)
    _ = tree_tensor_network_optimize(ttgraph, obj, num_sweeps, primitive_grid)

    calls = traced.calls
    f_min = float(np.min(traced.values)) if traced.values else np.nan
    hits = first_hit_calls(
        traced.values, float(f_opt) if f_opt is not None else np.nan, thresholds
    )
    return calls, f_min, hits, obj


def run_de(
    func: Callable,
    bounds: List[Tuple[float, float]],
    num_grid_points: int,
    rank: int,
    num_sweeps: int,
    seed: int = 42,
    *,
    grid_type: str = "plain",
    qtt_levels: int = 16,
    qtt_base: int = 2,
    qtt_z: int = 3,
    f_opt: float | None = None,
    thresholds: tuple[float, ...] = (1e-2, 1e-3, 1e-4, 1e-5),
):
    """SciPy Differential Evolution baseline.

    We minimize the *raw* objective f(x) directly (no score transform).
    """
    _seed_all(seed)
    num_dims = _dim(bounds)
    budget = _budget(
        num_grid_points,
        rank,
        num_sweeps,
        num_dims,
        grid_type=grid_type,
        qtt_levels=qtt_levels,
        qtt_base=qtt_base,
    )

    snap = _make_snap_fn(
        bounds,
        num_grid_points,
        grid_type=grid_type,
        qtt_levels=qtt_levels,
        qtt_base=qtt_base,
    )
    traced = RawEvalLogger(func)
    obj = Objective(traced)  # identity mapping; cache keys on snapped coordinates

    def f(x):
        return float(obj(snap(np.asarray(x))))

    popsize = 10
    maxiter = max(1, budget // max(popsize * num_dims, 1))

    _ = differential_evolution(
        f,
        bounds,
        seed=seed,
        popsize=popsize,
        maxiter=maxiter,
        polish=False,
        updating="deferred",
        workers=1,
        tol=0.0,
    )

    calls = traced.calls
    f_min = float(np.min(traced.values)) if traced.values else np.nan
    hits = first_hit_calls(
        traced.values, float(f_opt) if f_opt is not None else np.nan, thresholds
    )
    return calls, f_min, hits, obj


def run_da(
    func: Callable,
    bounds: List[Tuple[float, float]],
    num_grid_points: int,
    rank: int,
    num_sweeps: int,
    seed: int = 42,
    *,
    grid_type: str = "plain",
    qtt_levels: int = 16,
    qtt_base: int = 2,
    qtt_z: int = 3,
    f_opt: float | None = None,
    thresholds: tuple[float, ...] = (1e-2, 1e-3, 1e-4, 1e-5),
):
    """SciPy Dual Annealing baseline.

    We minimize the *raw* objective f(x) directly (no score transform).
    """
    _seed_all(seed)
    num_dims = _dim(bounds)
    budget = _budget(
        num_grid_points,
        rank,
        num_sweeps,
        num_dims,
        grid_type=grid_type,
        qtt_levels=qtt_levels,
        qtt_base=qtt_base,
    )

    snap = _make_snap_fn(
        bounds,
        num_grid_points,
        grid_type=grid_type,
        qtt_levels=qtt_levels,
        qtt_base=qtt_base,
    )
    traced = RawEvalLogger(func)
    obj = Objective(traced)  # identity mapping; cache keys on snapped coordinates

    def f(x):
        return float(obj(snap(np.asarray(x))))

    _ = dual_annealing(f, bounds=bounds, seed=seed, maxfun=budget, no_local_search=True)

    calls = traced.calls
    f_min = float(np.min(traced.values)) if traced.values else np.nan
    hits = first_hit_calls(
        traced.values, float(f_opt) if f_opt is not None else np.nan, thresholds
    )
    return calls, f_min, hits, obj
