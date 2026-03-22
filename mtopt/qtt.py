"""
Quantized Tensor-Train (QTT) utilities.

This module provides a lightweight "grid quantization" layer for mtopt.

Background
----------
In QTT, each *original* variable x_k is discretized on N_k = base**L_k points,
but represented as L_k "virtual" coordinates (digits) each of size `base`
(typically base=2). This turns a d-dimensional grid into a higher-dimensional
(digital) grid with small mode sizes, enabling TT-cross-like methods to work
with extremely fine discretizations without explicitly materializing them.

The mtopt optimizers operate on explicit point sets (skeleton grids). QTT is
implemented here by:

1) representing each digit coordinate as a primitive grid {0,1,...,base-1},
2) running the optimizer on these digit coordinates, and
3) decoding digit points to physical coordinates before calling the user's
   objective function.

Key idea: the optimizer never needs the full base**L grid; it only sees sampled
points in digit space.

Conventions
-----------
* Coordinates are indexed from 0.
* For a variable k with L_k digits, digit positions are ordered by default
  MSB->LSB (most significant digit first). This matches lexicographic ordering
  when coordinate ids increase with digit position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from mtopt.grid import Grid


ArrayLike = Union[np.ndarray, Sequence[float], Sequence[int]]


def _as_levels(num_vars: int, levels: Union[int, Sequence[int]]) -> List[int]:
    if isinstance(levels, int):
        if levels <= 0:
            raise ValueError("levels must be a positive integer.")
        return [levels] * num_vars
    lv = list(levels)
    if len(lv) != num_vars:
        raise ValueError(f"levels must have length num_vars={num_vars}, got {len(lv)}.")
    if any(int(L) <= 0 for L in lv):
        raise ValueError("All levels must be positive integers.")
    return [int(L) for L in lv]


def qtt_total_cores(num_vars: int, levels: Union[int, Sequence[int]]) -> int:
    """Total number of QTT digit coordinates (TT cores in digit space)."""
    return int(sum(_as_levels(num_vars, levels)))


def qtt_coordinate_map(
    num_vars: int, levels: Union[int, Sequence[int]]
) -> List[Tuple[int, int]]:
    """
    Map digit-coordinate id -> (var_id, digit_id).

    Digit ids are 0..L_k-1 within each variable k.
    Variables are laid out in var-major order: all digits of variable 0,
    then all digits of variable 1, etc.
    """
    lv = _as_levels(num_vars, levels)
    mapping: List[Tuple[int, int]] = []
    for k, L in enumerate(lv):
        for ell in range(L):
            mapping.append((k, ell))
    return mapping


def qtt_z_permuted_coordinate_map(
    num_vars: int,
    levels: Union[int, Sequence[int]],
    z: int,
) -> List[Tuple[int, int]]:
    """
    Coordinate map for z-permuted QTT.

    Variables are divided into consecutive groups of size z. Within each
    group, digits are interleaved by level: all level-0 digits first (one
    per variable in the group), then all level-1 digits, and so on.

    This places digits that correspond to the same resolution level of
    nearby variables adjacent in the TT chain, which can improve the
    rank structure for functions with spatial locality.

    Parameters
    ----------
    num_vars:
        Number of original (physical) variables.
    levels:
        Levels per variable -- int (same for all) or list [L_0,...,L_{K-1}].
    z:
        Group size for interleaving.  z=1 recovers standard var-major order;
        z >= num_vars gives full interleaving across all variables.

    Returns
    -------
    List of (var_id, digit_id) tuples, one per TT core position.

    Examples
    --------
    num_vars=6, levels=2, z=3 produces:
      group 0 (vars 0,1,2): (0,0),(1,0),(2,0),(0,1),(1,1),(2,1)
      group 1 (vars 3,4,5): (3,0),(4,0),(5,0),(3,1),(4,1),(5,1)
    """
    if z < 1:
        raise ValueError("z must be >= 1.")
    lv = _as_levels(num_vars, levels)
    mapping: List[Tuple[int, int]] = []
    for g_start in range(0, num_vars, z):
        group = list(range(g_start, min(g_start + z, num_vars)))
        max_L = max(lv[k] for k in group)
        for ell in range(max_L):
            for k in group:
                if ell < lv[k]:
                    mapping.append((k, ell))
    return mapping


def qtt_primitive_arrays(
    num_vars: int,
    levels: Union[int, Sequence[int]],
    base: int = 2,
    dtype: type = int,
) -> List[np.ndarray]:
    """
    Build primitive 1D grids for each QTT digit coordinate.

    Returns a list of length sum(levels), where each element is
    np.arange(base), i.e. [0,1,...,base-1].
    """
    if base < 2:
        raise ValueError("base must be >= 2.")
    total = qtt_total_cores(num_vars, levels)
    prim = np.arange(base, dtype=dtype)
    return [prim.copy() for _ in range(total)]


def qtt_primitive_grids(
    num_vars: int,
    levels: Union[int, Sequence[int]],
    base: int = 2,
) -> List["Grid"]:
    """
    Convenience: return mtopt.grid.Grid objects for each digit coordinate.

    This is the natural input format for mtopt.optimization models that expect
    a list of 1D :class:`~mtopt.grid.Grid` primitives.

    Notes
    -----
    Requires :class:`~mtopt.grid.Grid` to be available from :mod:`mtopt.grid`.
    """

    prim_arrays = qtt_primitive_arrays(num_vars=num_vars, levels=levels, base=base)
    return [Grid(arr, coords=i) for i, arr in enumerate(prim_arrays)]


def qtt_digits_to_index(
    digits: np.ndarray,
    base: int = 2,
    msb_first: bool = True,
) -> np.ndarray:
    """
    Convert base-`base` digits to integer indices.

    Parameters
    ----------
    digits:
        Array of digits of shape (..., L).
    base:
        Base for the digit expansion.
    msb_first:
        If True, digits[...,0] is the most significant digit.

    Returns
    -------
    indices:
        Integer array of shape (...,).
    """
    d = np.asarray(digits)
    if d.ndim < 1:
        raise ValueError("digits must have at least 1 dimension.")
    L = d.shape[-1]
    if L == 0:
        raise ValueError("digits must have length L>0 on the last axis.")
    if base < 2:
        raise ValueError("base must be >= 2.")

    # Robustly round to nearest integer digit, then clip
    dd = np.rint(d).astype(np.int64)
    if np.any(dd < 0) or np.any(dd >= base):
        raise ValueError(f"digits must be in [0, {base - 1}].")

    if msb_first:
        powers = base ** np.arange(L - 1, -1, -1, dtype=np.int64)
    else:
        powers = base ** np.arange(0, L, dtype=np.int64)

    return np.tensordot(dd, powers, axes=([-1], [0])).astype(np.int64)


def qtt_index_to_unit_interval(
    index: np.ndarray,
    num_points: int,
    endpoint: bool = False,
) -> np.ndarray:
    """
    Map integer indices to [0,1] (float).

    If endpoint=True, maps 0 -> 0 and (num_points-1) -> 1.
    If endpoint=False, maps i -> i/num_points (right endpoint excluded).
    """
    idx = np.asarray(index, dtype=np.float64)
    if num_points <= 0:
        raise ValueError("num_points must be positive.")
    if endpoint:
        if num_points == 1:
            return np.zeros_like(idx)
        return idx / (num_points - 1)
    return idx / num_points


@dataclass(frozen=True)
class QTTDecoder:
    """
    Decode QTT digit points into physical coordinates.

    Typical usage: build a digit-space objective for mtopt optimizers.

    Parameters
    ----------
    num_vars:
        Number of original (physical) variables.
    levels:
        Either a single int L (same for all vars) or a list [L_0,...,L_{K-1}]
        where K = num_vars.
    base:
        Digit base (2 for binary QTT).
    bounds:
        Sequence of (low, high) for each physical variable. If provided,
        decoded indices are mapped uniformly into these intervals.
        If None, decode() returns integer indices as float.
    msb_first:
        Digit order convention within each variable.
    endpoint:
        Mapping convention for uniform intervals; see qtt_index_to_unit_interval.
    permutation:
        Optional list of (var_id, digit_id) tuples of length sum(levels),
        specifying the ordering of digit coordinates in the flat digit vector.
        Must be a permutation of all (k, ell) pairs for k in 0..num_vars-1
        and ell in 0..L_k-1.  If None, var-major order is used (all digits of
        variable 0 first, then variable 1, etc.).  Use
        :func:`qtt_z_permuted_coordinate_map` to build a z-permuted ordering.

    Notes
    -----
    The ``permutation`` parameter only affects how the decoder interprets the
    flat digit vector.  The primitive grids passed to the optimizer are always
    ``[Grid(arange(base), coords=i) for i in range(total_cores)]`` regardless
    of ordering; use :func:`qtt_primitive_grids` to generate them.
    """

    num_vars: int
    levels: Union[int, Sequence[int]]
    base: int = 2
    bounds: Optional[Sequence[Tuple[float, float]]] = None
    msb_first: bool = True
    endpoint: bool = True
    permutation: Optional[Sequence[Tuple[int, int]]] = None

    def __post_init__(self) -> None:
        if self.num_vars <= 0:
            raise ValueError("num_vars must be positive.")
        if self.base < 2:
            raise ValueError("base must be >= 2.")
        lv = _as_levels(self.num_vars, self.levels)
        object.__setattr__(self, "_levels_list", lv)

        # Build coordinate map: flat position -> (var_id, digit_id)
        total = sum(lv)
        if self.permutation is not None:
            coord_map = list(self.permutation)
            if len(coord_map) != total:
                raise ValueError(
                    f"permutation must have length {total} (sum of levels), "
                    f"got {len(coord_map)}."
                )
            expected = {(k, ell) for k, L in enumerate(lv) for ell in range(L)}
            if set(coord_map) != expected:
                raise ValueError(
                    "permutation must be a permutation of all (var_id, digit_id) pairs."
                )
        else:
            coord_map = qtt_coordinate_map(self.num_vars, lv)

        # Inverse map: (var_id, digit_id) -> flat position
        inv_coord_map = {(k, ell): pos for pos, (k, ell) in enumerate(coord_map)}
        object.__setattr__(self, "_inv_coord_map", inv_coord_map)

        if self.bounds is not None:
            if len(self.bounds) != self.num_vars:
                raise ValueError(f"bounds must have length num_vars={self.num_vars}.")
            # Validate bounds
            for a, b in self.bounds:
                if not np.isfinite(a) or not np.isfinite(b):
                    raise ValueError("bounds entries must be finite.")
                if b <= a:
                    raise ValueError("Each bound must satisfy high > low.")

    @property
    def levels_list(self) -> List[int]:
        return list(self._levels_list)

    @property
    def total_cores(self) -> int:
        return int(sum(self._levels_list))

    def split_digits(self, q: np.ndarray) -> List[np.ndarray]:
        """
        Split digit coordinates into per-variable digit blocks.

        Respects the ordering defined by ``permutation`` (or var-major if None).
        For each variable k, returns its L_k digits in level order (ell=0..L_k-1),
        gathered from their positions in the flat digit vector.
        """
        Q = np.asarray(q)
        if Q.shape[-1] != self.total_cores:
            raise ValueError(
                f"Expected last axis size {self.total_cores}, got {Q.shape[-1]}."
            )
        blocks: List[np.ndarray] = []
        for k, L in enumerate(self._levels_list):
            positions = [self._inv_coord_map[(k, ell)] for ell in range(L)]
            blocks.append(Q[..., positions])
        return blocks

    def decode_indices(self, q: ArrayLike) -> np.ndarray:
        """
        Decode digit points to integer grid indices.

        Returns array of shape (..., num_vars) with dtype int64.
        """
        Q = np.asarray(q)
        if Q.ndim == 1:
            Q = Q.reshape(1, -1)
        blocks = self.split_digits(Q)
        inds = []
        for blk, L in zip(blocks, self._levels_list):
            idx = qtt_digits_to_index(blk, base=self.base, msb_first=self.msb_first)
            inds.append(idx)
        out = np.stack(inds, axis=-1).astype(np.int64)
        if np.asarray(q).ndim == 1:
            return out.reshape(-1)
        return out

    def decode(self, q: ArrayLike) -> np.ndarray:
        """
        Decode digit points to physical coordinates.

        If bounds is provided, outputs floats in those intervals.
        Otherwise, outputs indices as float.
        """
        inds = self.decode_indices(q)
        if self.bounds is None:
            return inds.astype(np.float64)

        # Uniform mapping index -> x in [a,b]
        X = np.asarray(inds, dtype=np.float64)
        for k, L in enumerate(self._levels_list):
            Nk = self.base**L
            u = qtt_index_to_unit_interval(X[..., k], Nk, endpoint=self.endpoint)
            a, b = self.bounds[k]
            X[..., k] = a + (b - a) * u
        return X


def make_qtt_objective(
    physical_objective: Callable[[np.ndarray], Union[float, np.ndarray]],
    decoder: QTTDecoder,
) -> Callable[[np.ndarray], Union[float, np.ndarray]]:
    """
    Wrap a physical objective f(x) to accept QTT digit points q.

    The resulting callable supports both:
      - q.shape == (dim_q,) -> scalar
      - q.shape == (n, dim_q) -> array of length n (if physical_objective supports it)

    If the physical objective is not vectorized, this wrapper falls back to
    pointwise evaluation on the batch.
    """

    def qtt_objective(q: np.ndarray) -> Union[float, np.ndarray]:
        Q = np.asarray(q)

        if Q.ndim == 1:
            x = decoder.decode(Q)
            return physical_objective(np.asarray(x, dtype=float))

        if Q.ndim != 2:
            raise ValueError("QTT objective expects a 1D or 2D array.")

        X = decoder.decode(Q)
        # Try vectorized evaluation first
        try:
            out = physical_objective(np.asarray(X, dtype=float))
            out_arr = np.asarray(out)
            if out_arr.shape != () and out_arr.shape[0] == X.shape[0]:
                return out_arr
        except (TypeError, ValueError):
            # Assume this indicates lack of vectorization support; fall back to pointwise.
            pass

        # Fallback: pointwise
        vals = np.empty(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            vals[i] = float(physical_objective(np.asarray(X[i], dtype=float)))
        return vals

    return qtt_objective
