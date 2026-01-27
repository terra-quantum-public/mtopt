r"""
Maximal-volume submatrix selection.

This module provides implementations of the *maxvol* and *rectangular maxvol*
algorithms for tall matrices:

* :func:`maxvol` finds a square submatrix of approximately maximal volume
  (i.e., with near-maximal absolute determinant).
* :func:`maxvol_rectangular` extends this to rectangular submatrices by greedily
  adding rows to the initial square maxvol selection.

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

.. [2] R. Ellerbrock, *pyQuTree* (software), Python package ``pyQuTree``.
       Source: https://github.com/roman-ellerbrock/pyQuTree
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.linalg import lu, solve_triangular


__all__ = ["maxvol", "maxvol_rectangular"]


def maxvol(
    matrix: np.ndarray,
    accuracy: float = 1.05,
    max_iters: int = 100,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Compute a square maximal-volume submatrix of a tall matrix.

    Given a tall matrix :math:`A \in \mathbb{R}^{n \times r}` with
    :math:`n \ge r`, this function finds a set of row indices ``row_indices``
    (of length ``r``) and a coefficient matrix ``coeff_matrix`` of shape
    ``(n, r)`` such that

    .. math::

        A \approx B \, A[\mathrm{row\_indices}, :],

    where ``B`` is ``coeff_matrix``. For the exact algorithm description,
    see the reference below.

    Parameters
    ----------
    matrix :
        Tall input matrix of shape ``(n, r)`` with ``n >= r``.
    accuracy :
        Accuracy parameter :math:`e \ge 1`. If ``accuracy == 1``, the
        algorithm iterates until true convergence (up to ``max_iters``).
        For ``accuracy > 1``, the algorithm may terminate earlier with
        slightly reduced accuracy; typical values are in the range
        ``[1.01, 1.1]``.
    max_iters :
        Maximum number of refinement iterations (must be >= 1).
    eps :
        Small non-negative value added to the diagonal of the ``U`` factor
        in the LU decomposition in case of singularity. This avoids division
        by zero when solving triangular systems. If too large, the accuracy
        of the algorithm may degrade; the default ``1e-8`` is usually
        sufficient.

    Returns
    -------
    row_indices : ndarray
        1D integer array of length ``r`` containing the indices of the
        selected rows that form the square maximal-volume submatrix.
    coeff_matrix : ndarray
        2D array of shape ``(n, r)`` such that approximately
        ``matrix ≈ coeff_matrix @ matrix[row_indices, :]``. In the ideal
        case, :math:`A (A[\text{row\_indices}, :])^{-1} = B`.

    Raises
    ------
    ValueError
        If the input matrix is not tall (i.e. ``n < r``).

    Notes
    -----
    Basic algorithm reference:

    * S. A. Goreinov, I. V. Oseledets, D. V. Savostyanov,
      E. E. Tyrtyshnikov, N. L. Zamarashkin,
      "How to find a good submatrix",
      *Matrix Methods: Theory, Algorithms And Applications*,
      Dedicated to the Memory of Gene Golub (2010), 247-256.
    """
    num_rows, rank = matrix.shape

    if num_rows == rank:
        row_indices = np.arange(rank, dtype=int)
        return row_indices, matrix

    if num_rows < rank:
        raise ValueError('Input matrix should be "tall" (n >= r).')

    # LU decomposition without pivoting checks for speed
    p_matrix, lower, upper = lu(matrix, check_finite=False, permute_l=False)  # pylint: disable=W0632

    try:
        # Initial row indices from LU permutation
        row_indices = p_matrix[:, :rank].argmax(axis=0)

        # Solve U^T * Q = A  =>  Q = (U^T)^{-1} A
        q_matrix = solve_triangular(
            upper,
            matrix.T,
            trans=1,
            check_finite=False,
        )

        # Solve L^T * B^T = Q  =>  B = ((L^T)^{-1} Q)^T
        coeff_matrix = solve_triangular(
            lower[:rank, :],
            q_matrix,
            trans=1,
            check_finite=False,
            unit_diagonal=True,
            lower=True,
        ).T
    except np.linalg.LinAlgError:
        # Regularize diagonal of U in case of singularity
        upper[np.diag_indices_from(upper)] += eps
        row_indices = p_matrix[:, :rank].argmax(axis=0)

        q_matrix = solve_triangular(
            upper,
            matrix.T,
            trans=1,
            check_finite=False,
        )
        coeff_matrix = solve_triangular(
            lower[:rank, :],
            q_matrix,
            trans=1,
            check_finite=False,
            unit_diagonal=True,
            lower=True,
        ).T

    # Greedy refinement iterations
    for _ in range(max_iters):
        flat_idx = np.abs(coeff_matrix).argmax()
        row_idx, col_idx = divmod(flat_idx, rank)

        if np.abs(coeff_matrix[row_idx, col_idx]) <= accuracy:
            break

        row_indices[col_idx] = row_idx

        col_vector = coeff_matrix[:, col_idx]
        row_vector = coeff_matrix[row_idx, :].copy()
        row_vector[col_idx] -= 1.0

        coeff_matrix -= np.outer(
            col_vector, row_vector / coeff_matrix[row_idx, col_idx]
        )

    return row_indices, coeff_matrix


def maxvol_rectangular(
    matrix: np.ndarray,
    accuracy: float = 1.1,
    min_extra_rows: int = 0,
    max_extra_rows: int | None = None,
    base_accuracy: float = 1.05,
    base_max_iters: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Compute a rectangular maximal-volume submatrix of a tall matrix.

    This function first applies :func:`maxvol` to find a square maximal-volume
    submatrix. It then greedily adds additional rows to obtain a rectangular
    submatrix of (approximately) maximal volume, subject to the constraints
    on the number of additional rows.

    Parameters
    ----------
    matrix :
        Tall input matrix of shape ``(n, r)`` with ``n > r``.
    accuracy :
        Accuracy parameter :math:`e > 0` controlling the stopping criterion
        for the greedy extension. The algorithm stops if the candidate
        contribution falls below ``accuracy**2`` (once the minimum size is
        reached).
    min_extra_rows :
        Minimum number of **additional** rows to be added on top of the
        initial square maxvol submatrix (must satisfy
        ``0 <= min_extra_rows <= n - r``).
    max_extra_rows :
        Maximum number of additional rows to be added. If ``None``, rows are
        added until the precision criterion governed by ``accuracy`` is
        satisfied, potentially selecting all rows (i.e. the full matrix).
        If ``r + max_extra_rows > n``, the effective maximum is clipped so
        that ``r + max_extra_rows == n``.
    base_accuracy :
        Accuracy parameter for the initial square maxvol call
        (must be >= 1). Passed as ``accuracy`` to :func:`maxvol`.
    base_max_iters :
        Maximum number of iterations for the initial square maxvol call
        (must be >= 1). Passed as ``max_iters`` to :func:`maxvol`.

    Returns
    -------
    row_indices : ndarray
        1D integer array of length ``r + dr`` with the selected row indices,
        where ``dr`` is the number of additional rows added. It satisfies
        ``min_extra_rows <= dr <= max_extra_rows`` (after clipping), or
        ``dr >= min_extra_rows`` if ``max_extra_rows`` is ``None``.
    coeff_matrix : ndarray
        2D array of shape ``(n, r + dr)`` such that approximately

        .. math::

            A \approx B \, A[\mathrm{row\_indices}, :],

        where ``B`` is ``coeff_matrix``.

    Raises
    ------
    ValueError
        If the minimum/maximum number of additional rows is inconsistent.

    Notes
    -----
    Rectangular maxvol reference:

    * A. Mikhalev, I. V. Oseledets,
      "Rectangular maximum-volume submatrices and their applications",
      *Linear Algebra and its Applications* 538 (2018), 187–211.
    """
    num_rows, rank = matrix.shape

    min_rank = rank + min_extra_rows
    max_rank = rank + max_extra_rows if max_extra_rows is not None else num_rows
    max_rank = min(max_rank, num_rows)

    if min_rank < rank or min_rank > max_rank or max_rank > num_rows:
        raise ValueError("Invalid minimum/maximum number of added rows.")

    # Initial square maxvol selection
    row_indices_initial, coeff_matrix = maxvol(
        matrix,
        accuracy=base_accuracy,
        max_iters=base_max_iters,
    )

    # Allocate space for up to max_rank rows
    row_indices = np.hstack(
        [
            row_indices_initial,
            np.zeros(max_rank - rank, dtype=row_indices_initial.dtype),
        ]
    )

    # Mask of rows that can still be selected
    selectable_mask = np.ones(num_rows, dtype=int)
    selectable_mask[row_indices_initial] = 0

    # Selection score: norm^2 of rows of coeff_matrix, zeroed for already selected rows
    score = selectable_mask * np.linalg.norm(coeff_matrix, axis=1) ** 2

    for col_idx in range(rank, max_rank):
        row_idx = np.argmax(score)

        if col_idx >= min_rank and score[row_idx] <= accuracy * accuracy:
            break

        row_indices[col_idx] = row_idx
        selectable_mask[row_idx] = 0

        v_vec = coeff_matrix.dot(coeff_matrix[row_idx])
        lambda_factor = 1.0 / (1.0 + v_vec[row_idx])

        coeff_matrix = np.hstack(
            [
                coeff_matrix - lambda_factor * np.outer(v_vec, coeff_matrix[row_idx]),
                (lambda_factor * v_vec).reshape(-1, 1),
            ]
        )

        score = selectable_mask * (score - lambda_factor * v_vec * v_vec)

    # Trim unused columns
    row_indices = row_indices[: coeff_matrix.shape[1]]

    # Normalize: selected rows form identity
    coeff_matrix[row_indices] = np.eye(coeff_matrix.shape[1], dtype=coeff_matrix.dtype)

    return row_indices, coeff_matrix
