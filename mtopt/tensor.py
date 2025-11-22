r"""
Tensor utilities for tensor rank cross and matrix train optimizers.

This module defines :class:`mtopt.tensor.Tensor`, a thin wrapper around
:class:`numpy.ndarray` that carries additional metadata describing a tensor
network structure, and :func:`mtopt.tensor.tensordot`, a convenience
function for contracting tensors along a named edge.

The main idea is to keep a lightweight but explicit mapping between tensor
axes and edges in a tensor network graph. Each axis of a
:class:`~mtopt.tensor.Tensor` is annotated by an ``edge`` label. These
labels are used to:

* keep track of how tensors are connected in a tensor network, and
* drive contractions via :func:`tensordot` by specifying the edge label
  instead of raw axis indices.

The numerical core relies entirely on :mod:`numpy`, so all usual ndarray
operations (slicing, broadcasting, etc.) remain available. Only a small
subset of operations is edge-aware:

* :class:`Tensor`
    Subclass of :class:`numpy.ndarray` with ``edges`` metadata and helpers
    like :meth:`Tensor.transpose` and :meth:`Tensor.flatten` that keep
    metadata in sync with axis permutations and reshaping.
* :func:`tensordot`
    Wrapper around :func:`numpy.tensordot` that contracts two
    :class:`Tensor` objects along a shared edge label and propagates
    the resulting edge metadata.

Example
-------
Construct a tensor with edge labels and contract it along a shared edge::

    >>> import numpy as np
    >>> from mtopt.tensor import Tensor, tensordot
    >>> A = Tensor(np.ones((2, 3)), edges=["i", "k"])
    >>> B = Tensor(np.ones((3, 4)), edges=["k", "j"])
    >>> C = tensordot(A, B, edge="k")
    >>> C.shape
    (2, 4)
    >>> C.edges
    [('i',), ('j',)]

The resulting tensor ``C`` behaves like a regular :class:`numpy.ndarray`
but retains edge information for further tensor-network manipulations.

This code has been taken and adapted from the PyQuTree package authored by Roman Ellerbrock.
"""


from __future__ import annotations

from collections.abc import Hashable
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike

from mtopt.network import back_permutation


# An edge is stored as a tuple, its contents are intentionally generic.
Edge = Tuple[Hashable, ...]


__all__ = ["Tensor", "tensordot"]


class Tensor(np.ndarray):
    r"""
    N-dimensional tensor with edge metadata.

    This class decorates a :class:`numpy.ndarray` with additional
    metadata describing a tensor network structure.

    Parameters
    ----------
    array : ArrayLike
        Array-like object used to construct the underlying
        :class:`numpy.ndarray`.
    edges : Edge
        Sequence of edge labels, one per axis of ``array``.
        Each element is converted to a canonical tuple via
        sorting, i.e. ``tuple(sorted(edge))``. This allows
        edges that are themselves iterables (e.g. pairs of
        node indices or tuples of edges for flattened axes).
    flattened_to : Optional[Edge]
        Optional edge label indicating which logical edge
        this tensor was flattened to. This metadata is
        not used for numerical operations but can be
        useful for bookkeeping in higher-level code.
    expanded_shape : Optional[Sequence[int]]
        Optional shape representing the "unflattened"
        or original shape before flattening and/or
        permutations. If ``None``, the current
        ``array.shape`` is used.

    Attributes
    ----------
    edges : List[Edge]
        Edge label for each axis of the tensor.
    flattened_to : Optional[Edge]
        Logical edge the tensor is currently flattened to,
        if applicable.
    expanded_shape : Tuple[int, ...]
        Shape of the tensor before flattening/permutation
        (for bookkeeping purposes).

    Notes
    -----
    The edge labels are kept separate from the actual
    axis order. Operations that reorder axes (such as
    :meth:`transpose`) update ``edges`` accordingly.
    """

    # Ensure NumPy prefers Tensor operations when mixing with ndarray.
    __array_priority__ = 100.0

    def __new__(
        cls,
        array: ArrayLike,
        edges: Sequence[Any],
        flattened_to: Optional[Any] = None,
        expanded_shape: Optional[Sequence[int]] = None,
    ) -> "Tensor":
        arr = np.asarray(array)
        if arr.ndim != len(edges):
            raise ValueError(
                f"Number of edges ({len(edges)}) does not match array ndim "
                f"({arr.ndim})."
            )

        obj = arr.view(cls)
        obj.edges: List[Edge] = [cls._normalize_edge(e) for e in edges]
        obj.flattened_to: Optional[Edge] = (
            cls._normalize_edge(flattened_to) if flattened_to is not None else None
        )
        obj.expanded_shape: Tuple[int, ...] = (
            tuple(expanded_shape) if expanded_shape is not None else tuple(arr.shape)
        )
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        """Propagate metadata when NumPy creates a new view."""
        if obj is None:
            return

        self.edges = getattr(obj, "edges", None) # pylint: disable=W0201
        self.flattened_to = getattr(obj, "flattened_to", None) # pylint: disable=W0201
        self.expanded_shape = getattr( # pylint: disable=W0201
            obj,
            "expanded_shape",
            getattr(obj, "shape", None),
        )

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_edge(edge: Any) -> Edge:
        """
        Canonicalize an edge descriptor.

        Parameters
        ----------
        edge :
            Edge descriptor. If it is iterable, its elements
            are sorted and stored as a tuple. Otherwise, it is
            wrapped into a 1-tuple.

        Returns
        -------
        Edge
            Canonical edge representation.
        """
        if edge is None:
            raise ValueError("Edge descriptor must not be None.")

        # Try to treat edge as an iterable of sortable elements.
        try:
            return tuple(sorted(edge))  # type: ignore[arg-type]
        except TypeError:
            # ``edge`` is not an iterable of sortable elements;
            # treat it as an atomic label.
            return (edge,)

    # ------------------------------------------------------------------
    # Tensor-specific operations
    # ------------------------------------------------------------------
    def flatten(self, edge: Any) -> "Tensor":
        r"""
        Flatten all axes except one into a single leading axis.

        The method reorders the tensor so that the axis corresponding
        to ``edge`` is last, then reshapes the tensor into a 2D array
        of shape ``(prod(other_axes), dim(edge))``.

        Parameters
        ----------
        edge :
            Edge label identifying the axis that is **not**
            flattened. The label is canonicalized using
            :meth:`_normalize_edge`.

        Returns
        -------
        Tensor
            A 2D :class:`Tensor` of shape ``(N, M)``, where ``M`` is the
            dimension of the axis corresponding to ``edge`` and ``N`` is
            the product of the remaining dimensions.

        Notes
        -----
        The new tensor will have two edge entries:

        * the first entry corresponds to all flattened axes grouped
          together (it may itself be a tuple-of-edges),
        * the second entry is the preserved ``edge``.

        This mirrors the original implementation where the grouped
        edges are represented as a single (nested) edge label.
        """
        e = self._normalize_edge(edge)

        # Compute permutation that moves the selected edge to the last axis.
        perm = back_permutation(self.edges, e)
        if len(perm) != self.ndim:
            raise ValueError(
                f"back_permutation returned invalid permutation of length "
                f"{len(perm)} for ndim={self.ndim}."
            )

        # Reorder tensor and edges.
        transposed: Tensor = self.transpose(perm)  # type: ignore[assignment]
        shape_perm = [transposed.shape[i] for i in range(transposed.ndim)]
        edges_perm = [transposed.edges[i] for i in range(transposed.ndim)]

        # Group all but the last edge into one "flattened" edge entry.
        grouped_edges = edges_perm[:-1]
        new_edges: List[Any] = [grouped_edges, e]

        new_shape = (int(np.prod(shape_perm[:-1], dtype=int)), shape_perm[-1])

        return Tensor(
            transposed.reshape(new_shape),
            new_edges,
            flattened_to=e,
            expanded_shape=self.expanded_shape,
        )

    def transpose(
        self,
        axes: Optional[Sequence[int]] = None,
    ) -> "Tensor":
        r"""
        Return a view of the tensor with permuted axes.

        Parameters
        ----------
        axes :
            Permutation of ``range(self.ndim)``. If ``None``, the axes
            are reversed (as in :meth:`numpy.ndarray.transpose`).

        Returns
        -------
        Tensor
            Tensor with permuted axes and correspondingly permuted
            edge labels.
        """
        if axes is None:
            axes = tuple(reversed(range(self.ndim)))
        else:
            axes = tuple(axes)

        if len(axes) != self.ndim:
            raise ValueError(
                f"axes must be a permutation of range({self.ndim}), "
                f"got length {len(axes)}."
            )

        permuted_arr = super().transpose(axes)
        permuted_edges = [self.edges[i] for i in axes]

        return Tensor(
            permuted_arr,
            permuted_edges,
            flattened_to=self.flattened_to,
            expanded_shape=self.expanded_shape,
        )


def tensordot(tensor_1: Tensor, tensor_2: Tensor, edge: Any) -> Tensor:
    r"""
    Contract two tensors along a shared edge.

    This is a thin wrapper around :func:`numpy.tensordot` that
    additionally propagates and updates edge metadata.

    Parameters
    ----------
    tensor_1, tensor_2 :
        Input tensors to be contracted. Both must be instances of
        :class:`Tensor`, and each must contain ``edge`` in their
        respective ``edges`` attribute.
    edge :
        Edge label along which to contract. It is canonicalized
        using :meth:`Tensor._normalize_edge`.

    Returns
    -------
    Tensor
        The contracted tensor with combined edge metadata.

    Raises
    ------
    ValueError
        If ``edge`` is not present in ``A.edges`` or ``B.edges``,
        or if the corresponding dimensions do not match.
    """
    e = Tensor._normalize_edge(edge) # pylint: disable=W0212

    try:
        itensor_1 = tensor_1.edges.index(e)
    except ValueError as exc:
        raise ValueError(f"Edge {e!r} not found in tensor_1.edges={tensor_1.edges!r}.") from exc

    try:
        itensor_2 = tensor_2.edges.index(e)
    except ValueError as exc:
        raise ValueError(f"Edge {e!r} not found in tensor_2.edges={tensor_2.edges!r}.") from exc

    if tensor_1.shape[itensor_1] != tensor_2.shape[itensor_2]:
        raise ValueError(
            "Cannot contract tensors: dimension mismatch along edge "
            f"{e!r} ({tensor_1.shape[itensor_1]} != {tensor_2.shape[itensor_2]})."
        )

    # Build edge list for the result.
    edges_1 = list(tensor_1.edges)
    edges_1.remove(e)
    edges_2 = list(tensor_2.edges)
    edges_2.remove(e)

    edges_c: List[Edge] = edges_1 + edges_2

    # Propagate the "large" endpoint of the contracted edge to other edges
    # that referenced the "small" endpoint.
    if len(e) >= 2:
        small, large = e[0], e[1]
        updated_edges_c: List[Any] = []
        for ex in edges_c:
            # ex can itself be a tuple-of-labels; map labels elementwise.
            try:
                ex_iter: Iterable[Any] = ex  # type: ignore[assignment]
            except TypeError:
                updated_edges_c.append(ex)
                continue

            mapped = tuple(large if lbl == small else lbl for lbl in ex_iter)
            updated_edges_c.append(mapped)
        edges_c = [Tensor._normalize_edge(ed) for ed in updated_edges_c] # pylint: disable=W0212
    else:
        # Fallback: just canonicalize.
        edges_c = [Tensor._normalize_edge(ed) for ed in edges_c] # pylint: disable=W0212

    result_array = np.tensordot(tensor_1, tensor_2, axes=(itensor_1, itensor_2))
    return Tensor(result_array, edges_c)
