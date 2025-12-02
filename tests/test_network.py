import numpy as np
import networkx as nx

from mtopt.network import (
    pre_edges,
    is_leaf,
    is_leaf_node,
    up_edge,
    flip,
    back_permutation,
    flatten_back,
    collect,
    up_edges_by_distance_to_root,
    sweep,
    up_sweep,
    reverse_sweep,
    add_leaves,
    root,
    up_leaves,
    children,
    star_sweep,
    remove_edge,
    add_layer_index,
    tensor_train_graph,
    tensor_train_operator_graph,
    build_tree,
    balanced_tree,
)


# ----------------------------------------------------------------------
# Basic local utilities
# ----------------------------------------------------------------------


def test_flip_and_pre_edges_remove_flipped():
    graph = nx.DiGraph()
    # 0 -> 1, 1 -> 0, 2 -> 1
    graph.add_edge(0, 1)
    graph.add_edge(1, 0)
    graph.add_edge(2, 1)

    edge = (1, 0)

    pre_all = pre_edges(graph, edge, remove_flipped=False)
    pre_filtered = pre_edges(graph, edge, remove_flipped=True)

    # Incoming edges into 1 are (0, 1) and (2, 1)
    assert set(pre_all) == {(0, 1), (2, 1)}
    # After removing flip(edge) = (0, 1), only (2, 1) remains
    assert pre_filtered == [(2, 1)]


def test_back_permutation_moves_edge_to_last():
    edges = ["a", "b", "c", "d"]
    perm = back_permutation(edges, "c")
    # "c" was at index 2 → should now be at last position
    assert perm[-1] == 2
    # Perm is a permutation of {0,1,2,3}
    assert sorted(perm) == [0, 1, 2, 3]


def test_flatten_back_shape():
    array = np.arange(12).reshape(3, 4)
    shape = (3, 4)
    flat = flatten_back(array, shape)
    # Should be (-1, shape[-1]) = (3, 4)
    assert flat.shape == (3, 4)


def test_collect_edge_attributes():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, r=2)
    graph.add_edge(1, 2, r=3)
    graph.add_edge(2, 3, r=4)

    edges = [(0, 1), (2, 3)]
    r_vals = collect(graph, edges, "r")

    assert r_vals == [2, 4]


# ----------------------------------------------------------------------
# Sweeps, root, and leaf utilities
# ----------------------------------------------------------------------


def test_add_leaves_and_up_leaves_and_is_leaf():
    # First, test add_leaves and is_leaf on a minimal graph
    graph = nx.DiGraph()
    num_cores = 3
    graph.add_nodes_from(range(num_cores))

    add_leaves(graph, num_cores)

    # There should be 2 * num_cores leaf edges (bidirectional)
    leaf_edges = [edge for edge in graph.edges() if edge[0] < 0 or edge[1] < 0]
    assert len(leaf_edges) == 2 * num_cores

    # All edges touching a negative node should be leaf edges
    for edge in leaf_edges:
        assert is_leaf(edge, graph)

    # Now test up_leaves / up_edge on a *connected* tensor train graph
    tt_graph = tensor_train_graph(num_cores=num_cores, rank=2, primitive_grid=4)

    upward_leaf_edges = up_leaves(tt_graph)
    # One upward leaf edge per core
    assert len(upward_leaf_edges) == num_cores

    for u, v in upward_leaf_edges:
        # Orientation: from leaf (negative) to core (non-negative)
        assert u < 0
        assert v >= 0
        assert is_leaf((u, v), tt_graph)
        assert up_edge((u, v), tt_graph)


def test_up_edges_and_sweep_order_basic_tt():
    num_cores = 3
    graph = tensor_train_graph(num_cores=num_cores, rank=2, primitive_grid=4)
    root_node = root(graph)

    up_edges, down_edges = up_edges_by_distance_to_root(graph, root_node)

    # Every edge is either up or down (for this TT construction)
    assert set(up_edges + down_edges) == set(graph.edges())

    # Check that up_sweep only contains upward edges
    up_only = up_sweep(graph, include_leaves=True)
    assert set(up_only) == set(up_edges)

    # sweep includes both, in that order
    full = sweep(graph, include_leaves=True)
    assert full[: len(up_edges)] == up_edges
    assert full[len(up_edges) :] == down_edges

    # reverse_sweep is just reversed sweep
    assert list(reversed(full)) == reverse_sweep(graph)


def test_children_and_star_sweep_on_small_tree():
    # Small balanced tree
    graph = balanced_tree(num_leaves=4, rank=2, phys_dim=3)
    root_node = root(graph)

    # There should be at least one internal node with children
    some_internal = None
    for node in graph.nodes():
        if node >= 0 and children(graph, node):
            some_internal = node
            break
    assert some_internal is not None

    child_nodes = children(graph, some_internal)
    assert all(c < some_internal for c in child_nodes)

    sweep_edges = star_sweep(graph, exclude_leaves=False)
    # At least some edges in the tree must appear
    assert len(sweep_edges) > 0
    # The sweep must start at the root or from edges adjacent to it
    assert any(root_node in edge for edge in sweep_edges)


# ----------------------------------------------------------------------
# Tensor train graph structure
# ----------------------------------------------------------------------


def test_tensor_train_graph_basic_structure_and_ranks():
    num_cores = 4
    rank = 3
    phys_dim = 5

    graph = tensor_train_graph(num_cores=num_cores, rank=rank, primitive_grid=phys_dim)

    # Core nodes are 0..num_cores-1
    core_nodes = list(range(num_cores))
    assert all(node in graph.nodes for node in core_nodes)

    # Leaves are negative indices; there should be num_cores of them
    leaf_nodes = {node for node in graph.nodes if node < 0}
    assert len(leaf_nodes) == num_cores

    # Leaf edges should have coordinate and r == phys_dim
    upward_leaf_edges = up_leaves(graph)
    coords = set()
    for edge in upward_leaf_edges:
        attrs = graph.edges[edge]
        assert "coordinate" in attrs
        assert "r" in attrs
        assert attrs["r"] == phys_dim
        coords.add(attrs["coordinate"])

    # Coordinates should be 0..num_cores-1
    assert coords == set(range(num_cores))

    # Internal edges should have r <= rank and be consistent between directions
    for u, v in graph.edges():
        if is_leaf((u, v), graph):
            continue
        r_uv = graph.edges[(u, v)]["r"]
        r_vu = graph.edges[(v, u)]["r"]
        assert r_uv <= rank
        assert r_vu <= rank
        assert r_uv == r_vu


def test_tensor_train_operator_graph_structure():
    num_cores = 2
    rank = 4
    phys_dim = 6

    graph = tensor_train_operator_graph(
        num_cores=num_cores, rank=rank, phys_dim=phys_dim
    )

    # Each core node has two leaf nodes (bra/ket)
    leaf_nodes = [n for n in graph.nodes if n < 0]
    assert len(leaf_nodes) == 2 * num_cores

    # Leaf edges (upward) should have coordinates assigned
    upward_leaf_edges = [
        edge for edge in graph.edges() if is_leaf(edge, graph) and up_edge(edge, graph)
    ]
    assert len(upward_leaf_edges) == 2 * num_cores

    coords = [graph.edges[edge]["coordinate"] for edge in upward_leaf_edges]
    assert sorted(coords) == list(range(2 * num_cores))

    # Rank assignments
    for edge in graph.edges():
        r_val = graph.edges[edge]["r"]
        if is_leaf(edge, graph):
            assert r_val == phys_dim
        else:
            assert r_val == rank


# ----------------------------------------------------------------------
# Balanced tree properties
# ----------------------------------------------------------------------


def test_build_tree_returns_edges_and_balanced_tree_ranks():
    num_leaves = 5
    edges = build_tree(num_leaves)
    assert len(edges) > 0

    rank = 3
    phys_dim = 7
    graph = balanced_tree(num_leaves=num_leaves, rank=rank, phys_dim=phys_dim)

    # Leaf edges have r == phys_dim
    for edge in graph.edges():
        r_val = graph.edges[edge]["r"]
        if is_leaf(edge, graph):
            assert r_val == phys_dim
        else:
            assert r_val <= rank

    # Internal edges should have consistent r in both directions
    for u, v in graph.edges():
        if is_leaf((u, v), graph):
            continue
        r_uv = graph.edges[(u, v)]["r"]
        r_vu = graph.edges[(v, u)]["r"]
        assert r_uv == r_vu


# ----------------------------------------------------------------------
# Graph mutation helpers
# ----------------------------------------------------------------------


def test_remove_edge_rewires_graph_and_removes_node():
    # Start from a small TT graph
    graph = tensor_train_graph(num_cores=3, rank=2, primitive_grid=4)

    # Pick a non-leaf edge, say (1, 2) if it exists
    candidate_edge = None
    for edge in graph.edges():
        if not is_leaf(edge, graph) and edge[0] == 1 and edge[1] == 2:
            candidate_edge = edge
            break

    # Fallback: just pick any non-leaf edge
    if candidate_edge is None:
        for edge in graph.edges():
            if not is_leaf(edge, graph):
                candidate_edge = edge
                break

    assert candidate_edge is not None
    source_node, target_node = candidate_edge

    # Predecessors into source_node excluding flipped edge
    predecessors_before = pre_edges(graph, candidate_edge, remove_flipped=True)
    predecessor_sources = {u for (u, _) in predecessors_before}

    graph = remove_edge(graph, candidate_edge)

    # Source node must be removed
    assert source_node not in graph.nodes

    # Predecessor edges should now point to target_node
    for u in predecessor_sources:
        assert (u, target_node) in graph.edges


def test_add_layer_index_sets_correct_shortest_path_distance():
    graph = tensor_train_graph(num_cores=3, rank=2, primitive_grid=4)
    root_node = root(graph)

    graph = add_layer_index(graph, root_node=root_node)

    for node in graph.nodes:
        layer = graph.nodes[node]["layer"]
        # layer should equal shortest path length from node to root_node
        dist = nx.shortest_path_length(graph, source=node, target=root_node)
        assert layer == dist


def test_is_leaf_node_for_tt_graph():
    graph = tensor_train_graph(num_cores=3, rank=2, primitive_grid=4)

    for node in graph.nodes:
        if node < 0:
            # negative indices are physical leaf nodes
            assert is_leaf_node(node, graph)
        else:
            # core nodes typically have degree > 1
            # (but we only assert that some core nodes are not leaf nodes)
            pass

    assert any(not is_leaf_node(node, graph) for node in graph.nodes if node >= 0)
