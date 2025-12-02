r"""
Graph utilities for tensor-network structures.

This module provides helper routines for working with tensor-network graphs
represented as :class:`networkx.DiGraph` objects. It is adapted from the
PyQuTree package by Roman Ellerbrock.

The conventions used here are:

* The graph is a directed tree (or tree-like) structure.
* Physical degrees of freedom are represented by *leaf nodes* with
  negative indices, connected bidirectionally to their parent *core* node.
* Edges may carry attributes such as

  - ``"coordinate"`` - index of the physical coordinate,
  - ``"r"`` - rank associated with the edge in tensor-train/tree formats.

The module includes:

* Generic navigation helpers (:func:`sweep`, :func:`up_leaves`,
  :func:`pre_edges`, :func:`children`, etc.).
* Utilities to remove edges and add layer indices.
* Constructors for common tensor-network architectures:

  - :func:`tensor_train_graph`
  - :func:`tensor_train_operator_graph`
  - :func:`balanced_tree`
"""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np


__all__ = [
    "pre_edges",
    "is_leaf",
    "is_leaf_node",
    "up_edge",
    "flip",
    "back_permutation",
    "flatten_back",
    "collect",
    "up_edges_by_distance_to_root",
    "sweep",
    "up_sweep",
    "reverse_sweep",
    "add_leaves",
    "root",
    "up_leaves",
    "children",
    "star_sweep",
    "remove_edge",
    "add_layer_index",
    "tensor_train_graph",
    "tensor_train_operator_graph",
    "build_tree",
    "balanced_tree",
]


# ---------------------------------------------------------------------
# Local edge / node utilities
# ---------------------------------------------------------------------


def pre_edges(
    graph: nx.DiGraph,
    edge: Tuple[int, int],
    remove_flipped: bool = False,
) -> List[Tuple[int, int]]:
    r"""
    Return incoming edges into the source node of a given edge.

    Parameters
    ----------
    graph :
        Directed tensor-network graph.
    edge :
        Edge given as a pair ``(u, v)``.
    remove_flipped :
        If ``True``, the reversed edge ``(v, u)`` is removed from the list
        of predecessors if present.

    Returns
    -------
    list of tuple
        List of incoming edges ``(w, u)`` with ``u = edge[0]``.
    """
    predecessors = list(graph.in_edges(edge[0]))
    if remove_flipped:
        flipped = flip(edge)
        predecessors = [e for e in predecessors if e != flipped]
    return predecessors


def is_leaf(edge: Tuple[int, int], graph: nx.DiGraph) -> bool:
    r"""
    Check whether an edge is a *leaf edge*.

    An edge is considered a leaf edge if at least one of its endpoints
    has in-degree less than or equal to 1. In the standard constructions
    used here, this corresponds to edges attached to physical leaf nodes.

    Parameters
    ----------
    edge :
        Edge ``(u, v)`` to test.
    graph :
        Directed tensor-network graph.

    Returns
    -------
    bool
        ``True`` if the edge is a leaf edge, ``False`` otherwise.
    """
    return graph.in_degree(edge[0]) <= 1 or graph.in_degree(edge[1]) <= 1


def is_leaf_node(node: int, graph: nx.DiGraph) -> bool:
    r"""
    Check whether a node is a *leaf node*.

    A leaf node is defined here as a node with in-degree less than or equal
    to 1. For the standard tensor-network constructions in this module,
    this picks out the physical nodes with negative indices.

    Parameters
    ----------
    node :
        Node index.
    graph :
        Directed tensor-network graph.

    Returns
    -------
    bool
        ``True`` if the node is a leaf node, ``False`` otherwise.
    """
    return graph.in_degree(node) <= 1


def up_edge(edge: Tuple[int, int], graph: nx.DiGraph) -> bool:
    r"""
    Determine if an edge is oriented *upward* towards the root.

    We measure the shortest-path distance from each endpoint to the root
    node and declare the edge to be "upward" if the source node is farther
    away from the root than the target node.

    Parameters
    ----------
    edge :
        Edge ``(u, v)`` to test.
    graph :
        Directed tensor-network graph.

    Returns
    -------
    bool
        ``True`` if ``edge`` points from a deeper node to a shallower node
        (towards the root), ``False`` otherwise.
    """
    root_node = root(graph)
    distance_source = nx.shortest_path_length(graph, source=edge[0], target=root_node)
    distance_target = nx.shortest_path_length(graph, source=edge[1], target=root_node)
    return distance_source > distance_target


def flip(edge: Tuple[int, int]) -> Tuple[int, int]:
    r"""
    Return the reversed (flipped) version of an edge.

    Parameters
    ----------
    edge :
        Edge ``(u, v)``.

    Returns
    -------
    tuple
        Flipped edge ``(v, u)``.
    """
    return edge[1], edge[0]


def back_permutation(edges: Sequence[Any], edge: Any) -> List[int]:
    r"""
    Compute a permutation that moves a given edge label to the last position.

    This is used, for example, when flattening a tensor along a specific
    edge: the corresponding axis is moved to the back.

    Parameters
    ----------
    edges :
        Sequence of edge labels.
    edge :
        Edge label to be moved to the last position. Equality is tested
        via ``==`` and the first occurrence is used.

    Returns
    -------
    list of int
        Permutation of ``range(len(edges))`` that moves ``edge`` to the end.

    Raises
    ------
    ValueError
        If ``edge`` is not found in ``edges``.
    """
    index = edges.index(edge)
    perm = list(range(len(edges)))
    perm.remove(index)
    perm.append(index)
    return perm


def flatten_back(array: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    r"""
    Reshape a flattened array back to a tensor-like 2D view.

    This helper is currently a thin wrapper around :meth:`numpy.ndarray.reshape`,
    reshaping the input to ``(-1, shape[-1])``. It is intended to mirror
    the layout produced by certain flattening operations.

    Parameters
    ----------
    array :
        Input array.
    shape :
        Original shape; only the last dimension is used here.

    Returns
    -------
    ndarray
        Reshaped array of shape ``(-1, shape[-1])``.
    """
    return array.reshape((-1, shape[-1]))


def collect(
    graph: nx.DiGraph,
    edges: Iterable[Tuple[int, int]],
    key: str,
) -> List[Any]:
    r"""
    Collect an edge attribute from a list of edges.

    Parameters
    ----------
    graph :
        Directed tensor-network graph.
    edges :
        Iterable of edges ``(u, v)``.
    key :
        Edge attribute key to collect.

    Returns
    -------
    list
        List of attribute values ``graph.edges[e][key]`` for each edge ``e``.
    """
    items: List[Any] = []
    for e in edges:
        items.append(graph.edges[e][key])
    return items


# ---------------------------------------------------------------------
# Sweeps and root-related utilities
# ---------------------------------------------------------------------


def up_edges_by_distance_to_root(
    graph: nx.DiGraph,
    root_node: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    r"""
    Split edges into upward and downward sets by distance to root.

    For each edge ``(u, v)``, we compare the shortest-path distances
    ``dist[u]`` and ``dist[v]`` to the root. If ``dist[u] > dist[v]``,
    the edge is considered *upward* (pointing towards the root). If
    ``dist[u] < dist[v]``, it is considered *downward*.

    Parameters
    ----------
    graph :
        Directed tensor-network graph.
    root_node :
        Node index to be used as the root for distance computations.

    Returns
    -------
    up_edges : list of tuple
        Upward edges sorted by decreasing distance of their target node.
    down_edges : list of tuple
        Downward edges sorted by increasing distance of their target node.
    """
    distance = dict(nx.shortest_path_length(graph, source=root_node))

    up_edges = [edge for edge in graph.edges() if distance[edge[0]] > distance[edge[1]]]
    down_edges = [
        edge for edge in graph.edges() if distance[edge[0]] < distance[edge[1]]
    ]

    sorted_up_edges = sorted(up_edges, key=lambda e: -distance[e[1]])
    sorted_down_edges = sorted(down_edges, key=lambda e: distance[e[1]])
    return sorted_up_edges, sorted_down_edges


def sweep(
    graph: nx.DiGraph,
    include_leaves: bool = True,
) -> List[Tuple[int, int]]:
    r"""
    Construct a full up-and-down edge sweep of the network.

    The sweep consists of:

    1. All *upward* edges (towards the root), sorted by decreasing distance
       to the root.
    2. All *downward* edges (away from the root), sorted by increasing
       distance to the root.

    Parameters
    ----------
    graph :
        Directed tensor-network graph.
    include_leaves :
        If ``False``, leaf edges (see :func:`is_leaf`) are excluded
        from the returned sweep.

    Returns
    -------
    list of tuple
        Ordered list of edges for a full sweep.
    """
    up_edges, down_edges = up_edges_by_distance_to_root(graph, root(graph))
    sweep_edges = up_edges + down_edges
    if not include_leaves:
        sweep_edges = [edge for edge in sweep_edges if not is_leaf(edge, graph)]
    return sweep_edges


def up_sweep(
    graph: nx.DiGraph,
    include_leaves: bool = True,
) -> List[Tuple[int, int]]:
    r"""
    Construct an upward-only edge sweep.

    This returns only the set of edges that point towards the root, ordered
    by decreasing distance to the root (deepest edges first).

    Parameters
    ----------
    graph :
        Directed tensor-network graph.
    include_leaves :
        If ``False``, leaf edges (see :func:`is_leaf`) are excluded.

    Returns
    -------
    list of tuple
        Ordered list of upward edges.
    """
    up_edges, _ = up_edges_by_distance_to_root(graph, root(graph))
    sweep_edges = up_edges
    if not include_leaves:
        sweep_edges = [edge for edge in sweep_edges if not is_leaf(edge, graph)]
    return sweep_edges


def reverse_sweep(graph: nx.DiGraph) -> List[Tuple[int, int]]:
    r"""
    Reverse sweep: reverse of :func:`sweep`.

    Parameters
    ----------
    graph :
        Directed tensor-network graph.

    Returns
    -------
    list of tuple
        Reversed full sweep.
    """
    return list(reversed(sweep(graph)))


def add_leaves(graph: nx.DiGraph, num_leaves: int) -> None:
    r"""
    Add bidirectional leaf edges to the graph.

    For each ``i`` in ``0, ..., num_leaves - 1``, this creates a pair of
    nodes ``i`` (core) and ``-i - 1`` (leaf) and connects them via edges
    ``(i, -i - 1)`` and ``(-i - 1, i)``. The edges receive a ``"coordinate"``
    attribute equal to ``i``.

    Parameters
    ----------
    graph :
        Directed tensor-network graph to modify in place.
    num_leaves :
        Number of leaf (physical) sites to add.
    """
    for i in range(num_leaves):
        edge = (i, -i - 1)
        graph.add_edge(i, -i - 1)
        graph.add_edge(-i - 1, i)
        graph.edges[edge]["coordinate"] = i
        graph.edges[flip(edge)]["coordinate"] = i


def root(graph: nx.DiGraph) -> int:
    r"""
    Return the root node of the tree.

    By convention, we use the node with the maximum index as the root.

    Parameters
    ----------
    graph :
        Directed tensor-network graph.

    Returns
    -------
    int
        Root node index.
    """
    # NOTE: This relies on numeric node labels; can be changed later if needed.
    return max(graph.nodes)


def up_leaves(graph: nx.DiGraph) -> List[Tuple[int, int]]:
    r"""
    Return all leaf edges that point upward (towards the root).

    Parameters
    ----------
    graph :
        Directed tensor-network graph.

    Returns
    -------
    list of tuple
        Leaf edges for which :func:`up_edge` returns ``True``.
    """
    return [
        edge for edge in graph.edges if is_leaf(edge, graph) and up_edge(edge, graph)
    ]


def children(graph: nx.DiGraph, node: int) -> List[int]:
    r"""
    Return the *children* of a node in the directed tree.

    Here we define children as nodes that have an incoming edge into
    ``node`` and have a smaller index than ``node``. This matches the
    indexing conventions in the tensor-network constructors.

    Parameters
    ----------
    graph :
        Directed tensor-network graph.
    node :
        Node index whose children are requested.

    Returns
    -------
    list of int
        List of child node indices.
    """
    incoming_edges = graph.in_edges(node)
    return [edge[0] for edge in incoming_edges if edge[0] < node]


def _star_sweep(
    graph: nx.DiGraph,
    node: int,
    parent: int | None,
    sweep_edges: List[Tuple[int, int]],
) -> None:
    r"""
    Recursive helper for :func:`star_sweep`.

    Parameters
    ----------
    graph :
        Directed tensor-network graph.
    node :
        Current node in the traversal.
    parent :
        Parent node in the traversal, or ``None`` for the root.
    sweep_edges :
        List of edges (modified in place) that accumulates the sweep.
    """
    child_nodes = children(graph, node)
    for child in child_nodes:
        if child < 0:
            # Skip physical leaf nodes
            continue
        sweep_edges.append((node, child))
        _star_sweep(graph, child, node, sweep_edges)
    if parent is not None:
        sweep_edges.append((node, parent))


def star_sweep(
    graph: nx.DiGraph,
    exclude_leaves: bool = False,
) -> List[Tuple[int, int]]:
    r"""
    Construct a star-like sweep starting from the root.

    The sweep traverses edges from the root down to children recursively
    and then back up to the parent, collecting edges in a DFS-like order.

    Parameters
    ----------
    graph :
        Directed tensor-network graph.
    exclude_leaves :
        If ``True``, leaf edges are excluded from the final sweep.

    Returns
    -------
    list of tuple
        Ordered list of edges.
    """
    root_node = root(graph)
    sweep_edges: List[Tuple[int, int]] = []
    _star_sweep(graph, root_node, None, sweep_edges)
    if exclude_leaves:
        sweep_edges = [edge for edge in sweep_edges if not is_leaf(edge, graph)]
    return sweep_edges


def remove_edge(graph: nx.DiGraph, edge: Tuple[int, int]) -> nx.DiGraph:
    r"""
    Remove an edge and connect its predecessors directly to its target.

    Given an edge ``(v, w)``, this function:

    1. Collects all incoming edges ``(u, v)`` into ``v`` (excluding the
       flipped edge ``(w, v)``).
    2. Re-routes them to ``(u, w)``, copying over their edge attributes.
    3. Removes both ``(v, w)`` and its flipped counterpart ``(w, v)``.
    4. Removes the node ``v`` from the graph.

    Parameters
    ----------
    graph :
        Directed tensor-network graph to modify.
    edge :
        Edge ``(v, w)`` to remove.

    Returns
    -------
    nx.DiGraph
        The modified graph (same object as input).
    """
    predecessors = pre_edges(graph, edge, remove_flipped=True)
    source_node = edge[0]
    target_node = edge[1]

    # Redirect predecessors to the target node
    for pred_edge in predecessors:
        attr = graph.get_edge_data(*pred_edge)
        graph.add_edge(pred_edge[0], target_node, **attr)
        graph.remove_edge(*pred_edge)

    graph.remove_edge(*edge)
    graph.remove_edge(*flip(edge))
    graph.remove_node(source_node)
    return graph


# ---------------------------------------------------------------------
# Tensor-network structural helpers
# ---------------------------------------------------------------------


def add_layer_index(graph: nx.DiGraph, root_node: int | None = None) -> nx.DiGraph:
    r"""
    Add a ``"layer"`` attribute to each node: distance from the root.

    The layer index is defined as the shortest-path distance from the
    given root node (or from :func:`root(graph)` if ``root_node`` is
    not provided).

    Parameters
    ----------
    graph :
        Directed tensor-network graph.
    root_node :
        Root node for layer computation. If ``None``, ``root(graph)``
        is used.

    Returns
    -------
    nx.DiGraph
        The same graph with node attribute ``"layer"`` set.
    """
    if root_node is None:
        root_node = root(graph)
    for node in graph.nodes:
        layer = nx.shortest_path_length(graph, node, root_node)
        graph.nodes[node]["layer"] = layer
    return graph


def tensor_train_graph(
    num_cores: int,
    rank: int = 2,
    primitive_grid: int | List[Sequence[Any]] = 8,
) -> nx.DiGraph:
    r"""
    Generate a tensor-train (TT) network graph.

    The construction is:

    * Core nodes: ``0, 1, ..., num_cores - 1``.
    * For each core ``i`` a leaf node ``-i - 1`` is added and connected
      bidirectionally.
    * Core nodes are connected in a chain with forward edges
      ``(i, i + 1)`` and backward edges ``(i, i - 1)``.
    * Each edge gets a rank attribute ``"r"``:

      - Leaf edges: ``r = N_i``, where ``N_i`` is the physical dimension
        for coordinate ``i``.
      - Internal edges: ``r = min(rank, ∏ r_pre)`` where ``r_pre`` are
        the ranks of predecessor edges.

    Parameters
    ----------
    num_cores :
        Number of TT cores (physical sites).
    rank :
        Target internal TT rank.
    primitive_grid :
        Either an integer ``N`` specifying the same physical dimension
        for all sites, or a list/sequence of grids for each coordinate,
        where the length of each grid determines the physical dimension.

    Returns
    -------
    nx.DiGraph
        Directed tensor-train graph with edge attributes ``"r"`` and
        ``"coordinate"`` set.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_cores))

    # Leaf edges (physical indices)
    add_leaves(graph, num_cores)

    # Core chain forward edges
    for i in range(num_cores - 1):
        graph.add_edge(i, i + 1)

    # Core chain backward edges
    for i in range(num_cores - 1, 0, -1):
        graph.add_edge(i, i - 1)

    # Physical dimensions per coordinate
    if isinstance(primitive_grid, int):
        phys_dims = [primitive_grid] * num_cores
    else:
        phys_dims = [len(grid) for grid in primitive_grid]

    # First pass: assign ranks based on predecessors
    for edge in sweep(graph):
        if not is_leaf(edge, graph):
            predecessors = pre_edges(graph, edge, remove_flipped=True)
            r_max = (
                int(np.prod(collect(graph, predecessors, "r")))
                if predecessors
                else rank
            )
            graph.edges[edge]["r"] = min(rank, r_max)
        else:
            coord = graph.edges[edge]["coordinate"]
            graph.edges[edge]["r"] = phys_dims[coord]

    # Second pass: ensure consistency between opposite directions
    for edge in sweep(graph):
        if not is_leaf(edge, graph):
            r_val = graph.edges[edge]["r"]
            other = graph.edges[flip(edge)]["r"]
            graph.edges[edge]["r"] = min(r_val, other)

    return graph


def tensor_train_operator_graph(
    num_cores: int,
    rank: int = 2,
    phys_dim: int = 8,
) -> nx.DiGraph:
    r"""
    Generate a tensor-train operator graph.

    This is similar to :func:`tensor_train_graph`, but each core node
    connects to **two** leaf nodes representing bra/ket (or input/output)
    indices, suitable for tensor-train operator representations.

    Parameters
    ----------
    num_cores :
        Number of TT cores.
    rank :
        Internal TT rank used on non-leaf edges.
    phys_dim :
        Physical dimension used on leaf edges.

    Returns
    -------
    nx.DiGraph
        Directed tensor-train operator graph with edge attributes
        ``"r"`` and ``"coordinate"`` set on leaf edges.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_cores))

    # Leaf edges: two leaves (x, y) per core
    for i in range(num_cores):
        node = i
        x = -2 * i - 1
        y = -2 * i - 2
        graph.add_edge(node, x)
        graph.add_edge(x, node)
        graph.add_edge(node, y)
        graph.add_edge(y, node)

    # Core chain forward edges
    for i in range(num_cores - 1):
        graph.add_edge(i, i + 1)

    # Core chain backward edges
    for i in range(num_cores - 1, 0, -1):
        graph.add_edge(i, i - 1)

    # Rank assignment
    for edge in graph.edges():
        if not is_leaf(edge, graph):
            graph.edges[edge]["r"] = rank
        else:
            graph.edges[edge]["r"] = phys_dim

    # Coordinate indices for leaf edges (only upward ones)
    coord = 0
    for edge in graph.edges():
        if is_leaf(edge, graph) and up_edge(edge, graph):
            graph.edges[edge]["coordinate"] = coord
            coord += 1

    return graph


# ---------------------------------------------------------------------
# Balanced tree construction
# ---------------------------------------------------------------------


def _combine_nodes(
    nodes: List[int],
    next_node_id: int,
    edges: List[Tuple[int, int]],
) -> Tuple[List[int], int, List[Tuple[int, int]]]:
    r"""
    Helper for :func:`build_tree`: combine nodes into parents.

    Takes a list of current nodes and repeatedly combines them in pairs
    into a new parent node with ID starting from ``next_node_id``.
    For each pair ``(l, r)``, new parent ``p`` is added and edges
    ``(r, p)`` and ``(l, p)`` are appended to ``edges``.

    Parameters
    ----------
    nodes :
        List of current leaf/internal node indices.
    next_node_id :
        Next available node index for newly created parents.
    edges :
        List of edges (modified in place) to which new edges are appended.

    Returns
    -------
    nodes :
        Updated list of nodes where each processed pair is replaced by
        the new parent.
    next_node_id :
        Updated next node index.
    edges :
        Updated edge list with newly added parent edges.
    """
    for i in range(len(nodes) - 2, -1, -2):
        left = nodes[i]
        right = nodes[i + 1]
        # Remove the pair
        nodes.pop(i)
        nodes.pop(i)
        # Add new parent node
        nodes.append(next_node_id)
        edges.append((right, next_node_id))
        edges.append((left, next_node_id))
        next_node_id += 1
    return nodes, next_node_id, edges


def build_tree(num_leaves: int) -> List[Tuple[int, int]]:
    r"""
    Create a near-balanced tree structure over ``num_leaves`` leaves.

    The function returns a set of edges that connect the leaves into
    a close-to balanced tree by repeatedly combining pairs of nodes.

    Parameters
    ----------
    num_leaves :
        Number of leaf nodes.

    Returns
    -------
    list of tuple
        Edges connecting leaves and internal nodes.
    """
    nodes = list(range(num_leaves))

    # Special case for odd number of leaves: mark first as pseudo-node.
    if num_leaves % 2 == 1:
        nodes[0] = -1

    next_node_id = num_leaves
    edges: List[Tuple[int, int]] = []
    while len(nodes) > 1:
        nodes, next_node_id, edges = _combine_nodes(nodes, next_node_id, edges)
    return edges


def balanced_tree(
    num_leaves: int,
    rank: int = 2,
    phys_dim: int = 8,
) -> nx.DiGraph:
    r"""
    Construct a close-to balanced tree tensor network.

    This closely follows the original PyQuTree implementation:

    * Leaf nodes are negative integers ``-i-1`` connected to internal
      nodes ``i`` (with a special case for odd ``num_leaves`` to avoid
      an unnecessary extra internal node).
    * Internal nodes are non-negative integers, and the internal tree is
      built via :func:`build_tree`.
    * Each directed edge carries a rank ``"r"``:
      - leaf edges get physical dimension ``phys_dim``,
      - internal edges get a bond dimension up to ``rank``, capped by
        the product of incoming ranks.
    * Each *upward* leaf edge (as determined by :func:`up_edge`) is
      assigned a unique integer ``"coordinate"`` index, which is also
      copied to its reverse edge. This is what :func:`tensor_network_grid`
      / :func:`tn_grid` rely on to attach primitive grids.

    Parameters
    ----------
    num_leaves :
        Number of physical leaves / dimensions.
    rank :
        Target internal bond dimension.
    phys_dim :
        Physical dimension (e.g. primitive grid size) on each leaf edge.

    Returns
    -------
    DiGraph
        Directed graph representing the balanced tree tensor network.
    """
    graph = nx.DiGraph()

    # ------------------------------------------------------------------
    # 1. Leaf edges: (-i-1, i) and their reverse.
    #    For odd num_leaves, we start from 1 (PyQuTree "odd-leaf" trick).
    # ------------------------------------------------------------------
    start = num_leaves % 2  # special case for odd number of leaves
    for i in range(start, num_leaves):
        edge = (-i - 1, i)
        graph.add_edge(edge[0], edge[1])
        graph.add_edge(edge[1], edge[0])

    # ------------------------------------------------------------------
    # 2. Internal tree structure using build_tree(num_leaves)
    # ------------------------------------------------------------------
    tree_edges = build_tree(num_leaves)
    for edge in tree_edges:
        graph.add_edge(edge[0], edge[1])

    for edge in reversed(tree_edges):
        graph.add_edge(edge[1], edge[0])

    # ------------------------------------------------------------------
    # 3. Assign ranks "r" (same logic as original code)
    # ------------------------------------------------------------------
    for edge in sweep(graph):
        if not is_leaf(edge, graph):
            pre = pre_edges(graph, edge, remove_flipped=True)
            if pre:
                r_max = int(np.prod(collect(graph, pre, "r")))
            else:
                # Root or degenerate case: just use rank
                r_max = rank
            graph.edges[edge]["r"] = min(rank, r_max)
        else:
            # Leaf edges get physical dimension
            graph.edges[edge]["r"] = phys_dim

    # Ensure symmetry: r(edge) = r(flip(edge)) for internal edges
    for edge in sweep(graph):
        if not is_leaf(edge, graph):
            r_edge = graph.edges[edge]["r"]
            r_other = graph.edges[flip(edge)]["r"]
            graph.edges[edge]["r"] = min(r_edge, r_other)

    # ------------------------------------------------------------------
    # 4. Assign "coordinate" to each upward leaf edge and its reverse.
    # ------------------------------------------------------------------
    leaf_edges_up = [
        edge for edge in graph.edges if is_leaf(edge, graph) and up_edge(edge, graph)
    ]

    # Sort for deterministic coordinate ordering
    leaf_edges_up = sorted(leaf_edges_up, key=lambda e: (e[1], e[0]))

    if len(leaf_edges_up) != num_leaves:
        raise ValueError(
            f"balanced_tree internal check failed: expected {num_leaves} "
            f"upward leaf edges, found {len(leaf_edges_up)}."
        )

    for coord, edge in enumerate(leaf_edges_up):
        graph.edges[edge]["coordinate"] = coord
        graph.edges[flip(edge)]["coordinate"] = coord

    return graph
