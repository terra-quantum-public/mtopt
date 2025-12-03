import numpy as np
import pandas as pd
import networkx as nx

from mtopt.optimization import (
    tree_tensor_network_optimize,
    numpy_array_to_tuple,
    tensor_network_cur,
    OptimizationLogger,
    Objective,
)
from mtopt.network import (
    tensor_train_graph,
    is_leaf,
)
from mtopt.grid import tensor_network_grid


def test_optimization_logger_records_and_best_row():
    """OptimizationLogger should accumulate rows and report the best 'f'."""
    logger = OptimizationLogger()

    logger({"x1": 0.0, "f": 1.0})
    logger({"x1": 1.0, "f": 0.5})

    # Two records stored
    assert isinstance(logger.dataframe, pd.DataFrame)
    assert len(logger.dataframe) == 2

    # Best objective is 0.5
    assert logger.dataframe["f"].min() == 0.5

    # __str__ should mention "Optimal value"
    s = str(logger)
    assert "Optimal value" in s
    assert "f" in s


def test_numpy_array_to_tuple_rounds_and_flattens():
    """numpy_array_to_tuple should flatten and round to the requested precision."""
    arr = np.array([[0.123456789, 0.987654321]])
    tpl = numpy_array_to_tuple(arr, precision=4)

    assert isinstance(tpl, tuple)
    # 2 entries because array has 2 elements
    assert len(tpl) == 2
    assert tpl[0] == round(0.123456789, 4)
    assert tpl[1] == round(0.987654321, 4)


def test_objective_caching_and_transformer_and_logging():
    """Objective should cache, apply transformer, and optionally log metadata."""
    call_counter = {"count": 0}

    def error_fn(x: np.ndarray) -> float:
        call_counter["count"] += 1
        return float(np.sum(x**2))

    # Use sqrt as a simple nontrivial transformer
    objective = Objective(error_fn, transformer=np.sqrt)

    x1 = np.array([0.1, 0.2])
    x2 = np.array([0.3, 0.4])

    # First call at x1: hits underlying function
    v1 = objective(x1)
    assert call_counter["count"] == 1
    assert objective.function_calls == 1
    assert objective.cache_hits == 0
    assert len(objective.cache) == 1

    # Second call at same point: should use cache, no extra function call
    v2 = objective(x1.copy())
    assert np.isclose(v1, v2)
    assert call_counter["count"] == 1
    assert objective.function_calls == 1
    assert objective.cache_hits == 1
    assert len(objective.cache) == 1

    # New point x2 with logging metadata
    v3 = objective(x2, sweep=0, node=1)
    assert call_counter["count"] == 2
    assert objective.function_calls == 2
    assert len(objective.cache) == 2

    # Logger should have exactly one record (for x2)
    df = objective.logger.dataframe
    assert len(df) == 1
    assert set(["x1", "x2", "f", "sweep", "node"]).issubset(df.columns)
    assert df["sweep"].iloc[0] == 0
    assert df["node"].iloc[0] == 1

    # Transformer should be applied (sqrt of sum of squares)
    expected_raw = np.sum(x2**2)
    assert np.isclose(v3, np.sqrt(expected_raw))


def _build_small_tt_network(num_cores: int = 3, rank: int = 2) -> nx.DiGraph:
    """Helper: small tensor train network with default physical dimension."""
    # Positional arguments only to avoid depending on keyword names
    graph = tensor_train_graph(num_cores, rank)
    return graph


def test_ttn_opt_runs_and_populates_tensors():
    """ttn_opt should run without errors and populate some node/edge tensors."""
    np.random.seed(123)

    num_cores = 3
    rank = 2
    num_points_per_dim = 6

    graph = _build_small_tt_network(num_cores=num_cores, rank=rank)

    primitive_grids = [
        np.linspace(0.0, 1.0, num_points_per_dim) for _ in range(num_cores)
    ]

    def error_fn(x: np.ndarray) -> float:
        # Simple smooth objective: sum of squares
        return float(np.sum(x**2))

    objective = Objective(error_fn)

    # Run a couple of sweeps
    graph_out = tree_tensor_network_optimize(
        graph,
        objective,
        num_sweeps=2,
        primitive_grid=primitive_grids,
    )

    # We must at least have evaluated the function a few times
    assert objective.function_calls > 0
    assert len(objective.cache) > 0

    # There should be at least one node with an attached tensor "A" and a "grid"
    node_has_tensors = any(
        ("A" in data and "grid" in data) for _, data in graph_out.nodes(data=True)
    )
    assert node_has_tensors

    # Internal edges (non-leaf) should have grids; some should have tensors "A"
    internal_edges = [e for e in graph_out.edges if not is_leaf(e, graph_out)]
    assert len(internal_edges) > 0

    some_edge_has_grid = any("grid" in graph_out.edges[e] for e in internal_edges)
    assert some_edge_has_grid

    # After at least one sweep, some internal edge should have an "A" tensor
    some_edge_has_tensor = any("A" in graph_out.edges[e] for e in internal_edges)
    assert some_edge_has_tensor


def test_tn_cur_builds_cur_like_tensors():
    """tn_cur should attach CUR-like tensors on nodes and edges using existing grids."""
    np.random.seed(456)

    num_cores = 3
    rank = 2
    num_points_per_dim = 5

    # Build TT network and attach grids, but do not run optimization yet
    graph = _build_small_tt_network(num_cores=num_cores, rank=rank)

    primitive_grids = [
        np.linspace(0.0, 1.0, num_points_per_dim) for _ in range(num_cores)
    ]
    graph = tensor_network_grid(graph, primitive_grids)

    def error_fn(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    objective = Objective(error_fn)

    graph_cur = tensor_network_cur(graph, objective)

    # There should be at least one (physical) node with an "A" tensor
    physical_nodes = [n for n in graph_cur.nodes if n >= 0]
    assert len(physical_nodes) > 0

    node_has_A = any("A" in graph_cur.nodes[n] for n in physical_nodes)
    assert node_has_A

    # And at least one internal edge with an "A" tensor
    internal_edges = [e for e in graph_cur.edges if not is_leaf(e, graph_cur)]
    assert len(internal_edges) > 0

    edge_has_A = any("A" in graph_cur.edges[e] for e in internal_edges)
    assert edge_has_A

    # CUR construction should have triggered multiple objective evaluations
    assert objective.function_calls > 0
    assert len(objective.cache) > 0
