"""
This module provides a small collection of convenience routines for
visualizing:

* 3D point clouds on grids (with Plotly),
* tensor-train diagrams and tree tensor networks (with Matplotlib),
* function values over grids as :class:`pandas.DataFrame` objects, and
* animated 3D scatter plots (Plotly + imageio).

Most functions are thin wrappers around :mod:`matplotlib`, :mod:`plotly`,
:mod:`pandas`, :mod:`imageio`, and :mod:`networkx`. These external
libraries are required for the corresponding plotting utilities and are
imported when this module is imported.

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

from typing import Any, List, Sequence
import shutil
import os

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import networkx as nx
import pandas as pd
import numpy as np

from mtopt.network import add_layer_index, up_leaves, children
from mtopt.grid import Grid, direct_sum


__all__ = [
    "plot_xyz",
    "plot_tensor_train_diagram",
    "plot_tensor_network_xyz",
    "tensor_network_to_dataframe",
    "plot_tree",
    "tensor_network_grid_to_dataframe",
    "concat_pandas",
    "grid_animation",
    "grid_animation_to_gif",
]


# ---------------------------------------------------------------------------
# Basic 3D scatter plotting
# ---------------------------------------------------------------------------


def plot_xyz(
    grid: Grid | np.ndarray,
    values: np.ndarray,
    ranges: Sequence[Sequence[float]] | None = None,
):
    r"""
    Create a 3D scatter plot of function values over a 3D grid.

    This helper uses :mod:`plotly.graph_objects` to render a 3D scatter
    plot of points with coordinates taken from ``grid`` and colors given
    by ``values``.

    Parameters
    ----------
    grid :
        Either a :class:`~mtopt.grid.Grid` instance with exactly three
        coordinates, or an array-like object of shape ``(n_points, 3)``
        containing the :math:`x, y, z` coordinates of the points.
    values :
        Array of shape ``(n_points,)`` giving the function values to
        visualize. These are used as the color dimension.
    ranges :
        Optional explicit axis ranges as a sequence of three
        ``(min, max)`` pairs, one for each of the :math:`x, y, z` axes.
        If ``None``, the axes are auto-ranged by Plotly.

    Returns
    -------
    plotly.graph_objects.Figure
        The constructed Plotly figure.

    Notes
    -----
    This function requires :mod:`plotly` to be installed. If it is not
    available, an :class:`ImportError` is raised.
    """

    if isinstance(grid, Grid):
        points = grid.grid
    else:
        points = np.asarray(grid)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            "plot_xyz expects a 3D grid with shape (n_points, 3); "
            f"got shape {points.shape}."
        )

    values = np.asarray(values)
    if values.shape[0] != points.shape[0]:
        raise ValueError(
            "Length of `values` must match number of grid points: "
            f"{values.shape[0]} vs {points.shape[0]}."
        )

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(
                    size=5,
                    color=values,
                    colorscale="Viridis",
                    colorbar=dict(title="Function value"),
                ),
            )
        ]
    )

    if ranges is None:
        scene_kwargs = dict(
            xaxis=dict(title="x", autorange=True),
            yaxis=dict(title="y", autorange=True),
            zaxis=dict(title="z", autorange=True),
            aspectmode="cube",
        )
    else:
        if len(ranges) != 3:
            raise ValueError("`ranges` must be a sequence of three (min, max) pairs.")
        scene_kwargs = dict(
            xaxis=dict(title="x", range=ranges[0], autorange=False),
            yaxis=dict(title="y", range=ranges[1], autorange=False),
            zaxis=dict(title="z", range=ranges[2], autorange=False),
            aspectmode="cube",
        )

    fig.update_layout(scene=scene_kwargs, margin=dict(l=0, r=0, b=0, t=0))
    return fig


# ---------------------------------------------------------------------------
# Tensor-train / TN diagrams
# ---------------------------------------------------------------------------


def plot_tensor_train_diagram(
    graph: nx.DiGraph,
    draw_ranks: bool = True,
):
    r"""
    Plot a simple tensor-train (TT) diagram using :mod:`matplotlib`.

    The TT is assumed to be represented as a directed graph where
    physical (leaf) nodes have negative indices and core nodes have
    non-negative indices, as constructed by
    :func:`mtopt.network.tensor_train_graph`.

    Parameters
    ----------
    graph :
        Directed graph representing the tensor train.
    draw_ranks :
        If ``True``, edge ranks stored under the ``"r"`` attribute are
        drawn as edge labels.

    Returns
    -------
    matplotlib.axes.Axes
        The Matplotlib axes on which the diagram was drawn.

    Notes
    -----
    This function requires :mod:`matplotlib` to be installed. If it is
    not available, an :class:`ImportError` is raised.
    """

    num_leaves = len(up_leaves(graph))

    # Leaf nodes are indexed -1, -2, ..., -num_leaves
    leaf_positions = [np.array([-(-i - 1), 0.1]) for i in range(-num_leaves, 0)]
    core_positions = [np.array([-i, 0.0]) for i in range(num_leaves)]
    positions_array = leaf_positions + core_positions

    pos = {i: positions_array[i] for i in range(-num_leaves, num_leaves)}

    ax = plt.gca()
    ax.set_aspect(15)  # stretch horizontally

    nx.draw(
        graph,
        pos=pos,
        with_labels=False,
        node_size=250,
        font_size=8,
        ax=ax,
    )

    if draw_ranks:
        ranks = nx.get_edge_attributes(graph, "r")
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=ranks, font_size=14, ax=ax)

    return ax


def plot_tensor_network_xyz(
    graph: nx.Graph,
    func,
    q_to_x=None,
    ranges: Sequence[Sequence[float]] | None = None,
):
    r"""
    Visualize function values on all node grids of a tensor network.

    This helper:

    1. Collects all node grids stored under the ``"grid"`` attribute,
       and forms their direct sum via :func:`mtopt.grid.direct_sum`.
    2. Optionally applies a coordinate transformation ``q_to_x`` to
       each grid point.
    3. Evaluates ``func`` on all resulting points.
    4. Calls :func:`plot_xyz` to render a 3D scatter plot.

    Parameters
    ----------
    graph :
        Tensor-network-like graph with node attribute ``"grid"`` set to
        :class:`~mtopt.grid.Grid` instances.
    func :
        Callable with signature ``func(point) -> float``, where
        ``point`` is a 1D array-like representing the coordinates of a
        grid point.
    q_to_x :
        Optional callable mapping coordinates from an internal space
        (e.g. :math:`q`) to a physical space (e.g. :math:`x`). It is
        applied via :meth:`mtopt.grid.Grid.transform` before evaluating
        ``func``.
    ranges :
        Optional axis ranges forwarded to :func:`plot_xyz`.

    Returns
    -------
    plotly.graph_objects.Figure
        The resulting Plotly figure.
    """
    node_grids_dict = nx.get_node_attributes(graph, "grid")
    node_grids: List[Grid] = [g for g in node_grids_dict.values() if g is not None]

    if not node_grids:
        raise ValueError("No node grids found on graph (missing 'grid' attributes).")

    grid = direct_sum(node_grids)

    if q_to_x is not None:
        grid = grid.transform(q_to_x)

    values = grid.evaluate(func)
    fig = plot_xyz(grid, values, ranges=ranges)
    return fig


def tensor_network_to_dataframe(graph: nx.Graph, func=None):
    r"""
    Convert node grids of a tensor network into a :class:`pandas.DataFrame`.

    Each row of the resulting DataFrame corresponds to a single grid
    point from a node grid.

    Parameters
    ----------
    graph :
        Tensor-network-like graph with node attribute ``"grid"`` set to
        :class:`~mtopt.grid.Grid` instances.
    func :
        Optional callable with signature ``func(point) -> float`` used
        to compute a scalar value at each grid point. If provided, the
        resulting DataFrame contains an additional column ``"f"``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:

        * ``"xyz"`` - the raw point (1D array of coordinates),
        * ``"node"`` - the node identifier,
        * optionally ``"f"`` - the function value at that point.

    Notes
    -----
    This function requires :mod:`pandas` to be installed. If it is not
    available, an :class:`ImportError` is raised.
    """

    node_grids_dict = nx.get_node_attributes(graph, "grid")

    points: List[np.ndarray] = []
    node_ids: List[Any] = []
    values: List[float] = []

    for node_id, grid in node_grids_dict.items():
        if grid is None:
            continue
        for point in grid.grid:
            points.append(point)
            node_ids.append(node_id)
            if func is not None:
                values.append(func(point))

    data = {"xyz": points, "node": node_ids}
    if func is not None:
        data["f"] = values

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Tree layout for TNs
# ---------------------------------------------------------------------------


def plot_tree(
    graph: nx.DiGraph,
    draw_ranks: bool = True,
):
    r"""
    Plot a tree-like tensor network laid out by depth (layer index).

    The layout uses :func:`mtopt.network.add_layer_index` to assign a
    ``"layer"`` attribute to each node, then:

    * places leaf nodes (negative indices) evenly along the top,
    * positions internal nodes at the horizontal center of their children.

    Parameters
    ----------
    graph :
        Directed graph representing a tree-like tensor network.
    draw_ranks :
        If ``True``, edge ranks stored under the ``"r"`` attribute are
        drawn as edge labels.

    Returns
    -------
    matplotlib.axes.Axes
        The Matplotlib axes on which the tree was drawn.
    """

    graph = add_layer_index(graph)
    num_leaves = len(up_leaves(graph))
    leaf_grid = np.linspace(0.0, 1.0, num_leaves)

    pos: dict[Any, tuple[float, float]] = {
        node: (0.0, 0.0) for node in sorted(graph.nodes)
    }

    for node in sorted(graph.nodes):
        layer = graph.nodes[node]["layer"]
        y = -float(layer)

        if node < 0:
            # Leaf: position by its index
            leaf_id = -node - 1
            x = float(leaf_grid[leaf_id])
        else:
            # Internal node: center over its children
            child_nodes = children(graph, node)
            if child_nodes:
                child_positions = np.array(
                    [pos[c][0] for c in child_nodes], dtype=float
                )
                x = float(np.mean(child_positions))
            else:
                x = 0.0

        pos[node] = (x, y)

    fig, ax = plt.subplots()
    nx.draw(graph, pos=pos, with_labels=False, node_size=500, ax=ax)

    if draw_ranks:
        ranks = nx.get_edge_attributes(graph, "r")
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=ranks, font_size=18, ax=ax)

    return ax


# ---------------------------------------------------------------------------
# DataFrame helpers for grids
# ---------------------------------------------------------------------------


def tensor_network_grid_to_dataframe(graph: nx.Graph, objective) -> "pd.DataFrame":
    r"""
    Convert node grids and objective values to a :class:`pandas.DataFrame`.

    For each node with a ``"grid"`` attribute, this function collects
    all grid points, evaluates an objective on each point and returns a
    DataFrame with one row per grid point and separate columns for the
    coordinates.

    Parameters
    ----------
    graph :
        Tensor-network-like graph with node attribute ``"grid"`` set to
        :class:`~mtopt.grid.Grid` instances.
    objective :
        Either a callable with signature ``objective(point) -> float``
        or an object with a method ``Err(point) -> float`` (for
        compatibility with some legacy code).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:

        * ``"node"`` - node identifier,
        * ``"f"`` - objective value at the grid point,
        * ``"x1"``, ``"x2"``, ..., ``"xD"`` - coordinate components.

    Notes
    -----
    This function requires :mod:`pandas` to be installed. If it is not
    available, an :class:`ImportError` is raised.
    """

    node_attributes = nx.get_node_attributes(graph, "grid")
    # Extract raw NumPy arrays for each node that has a grid
    node_grids = {
        node: grid.grid for node, grid in node_attributes.items() if grid is not None
    }

    records: List[dict[str, Any]] = []

    # Determine how to call the objective:
    if callable(objective):
        eval_fn = objective
    elif hasattr(objective, "Err"):
        eval_fn = objective.Err
    else:
        raise TypeError(
            "objective must be callable or have a method 'Err(point)'. "
            f"Got object of type {type(objective)!r}."
        )

    for node_id, grid_array in node_grids.items():
        for point in grid_array:
            value = eval_fn(point)
            records.append({"node": node_id, "f": value, "grid": point})

    df = pd.DataFrame.from_records(records)

    if df.empty:
        return df

    # Expand 'grid' column into x1, x2, ..., xD
    num_coords = len(df.loc[0, "grid"])
    coord_cols = [f"x{i + 1}" for i in range(num_coords)]
    df[coord_cols] = df["grid"].apply(lambda p: pd.Series(p, index=coord_cols))

    df = df.drop(columns="grid")
    return df


def concat_pandas(dataframes: Sequence["pd.DataFrame"]):
    r"""
    Concatenate a sequence of DataFrames and add a ``"time"`` column.

    This is a small utility helpful for building animation frames. Each
    input DataFrame is assigned an integer ``time`` label corresponding
    to its position in the input sequence, then all are concatenated
    and a ``"size"`` column with value ``1`` is added.

    Parameters
    ----------
    dataframes :
        Sequence of :class:`pandas.DataFrame` instances to concatenate.

    Returns
    -------
    pandas.DataFrame
        Concatenated DataFrame with additional columns ``"time"`` and
        ``"size"``.

    Notes
    -----
    This function requires :mod:`pandas` to be installed. If it is not
    available, an :class:`ImportError` is raised.
    """

    frames: List[pd.DataFrame] = []
    for t, df in enumerate(dataframes):
        df_copy = df.copy()
        df_copy["time"] = t
        frames.append(df_copy)

    if not frames:
        return pd.DataFrame()

    concatenated = pd.concat(frames, ignore_index=True)
    concatenated["size"] = 1
    return concatenated


# ---------------------------------------------------------------------------
# Plotly-based animations
# ---------------------------------------------------------------------------


def grid_animation(
    dataframe,
    color: str = "f",
):
    r"""
    Create an interactive 3D scatter animation from a grid DataFrame.

    The input ``dataframe`` is expected to contain at least the columns:

    * ``"x1"``, ``"x2"``, ``"x3"`` - coordinates of each point,
    * ``"time"`` - frame index (used to animate over time),
    * ``"node"`` - node identifier (used as animation group),
    * ``"size"`` - marker size,
    * and a column specified by ``color`` (default ``"f"``) for coloring.

    This helper uses :mod:`plotly.express.scatter_3d` with an animation
    slider and adjusts the frame and transition durations for a smooth
    playback.

    Parameters
    ----------
    dataframe :
        :class:`pandas.DataFrame` containing the columns described above.
    color :
        Name of the column to use for coloring the points. Defaults to ``"f"``.

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure with frames suitable for interactive animation.

    Notes
    -----
    This function requires :mod:`pandas` and :mod:`plotly` to be installed.
    If they are not available, an :class:`ImportError` is raised.
    """

    fig = px.scatter_3d(
        dataframe,
        x="x1",
        y="x2",
        z="x3",
        animation_frame="time",
        animation_group="node",
        size="size",
        color=color,
        hover_name="time",
        size_max=15,
        width=1000,
        height=800,
    )

    # Tweak animation speed if updatemenus are present
    if fig.layout.updatemenus:
        button_args = fig.layout.updatemenus[0].buttons[0].args[1]
        button_args["frame"]["duration"] = 100
        button_args["transition"]["duration"] = 20

    fig.update_layout(
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False),
        )
    )

    camera = dict(
        eye=dict(x=1.5, y=-1.0, z=1.2),
        center=dict(x=0.0, y=0.0, z=0.0),
        up=dict(x=0.0, y=0.0, z=1.0),
    )
    fig.update_layout(scene_camera=camera)

    return fig


def grid_animation_to_gif(
    dataframe,
    color: str = "f",
    gif_filename: str = "animation.gif",
    frames_folder: str = ".frames",
    fps: int = 10,
    frame_indices: Sequence[int] | None = None,
) -> str:
    r"""
    Render a 3D grid animation to an animated GIF.

    This helper iterates over time steps found in ``dataframe["time"]``,
    generates a static 3D scatter plot for each time step, saves the frames
    as PNGs, and combines them into a GIF using :mod:`imageio`.

    Parameters
    ----------
    dataframe :
        :class:`pandas.DataFrame` with at least columns ``"x1"``,
        ``"x2"``, ``"x3"`` and ``"time"``. All rows with the same
        ``"time"`` value are plotted together in a single frame.
    color :
        Name of the column to use for coloring the points. Defaults to ``"f"``.
    gif_filename :
        Output filename for the resulting GIF (e.g. ``"animation.gif"``).
    frames_folder :
        Temporary folder used to store individual PNG frames before they
        are assembled into a GIF.
    fps :
        Frames per second of the resulting GIF. Defaults to ``10``.
    frame_indices :
        Optional sequence of specific time indices to render. If ``None``,
        all unique values of ``dataframe["time"]`` are used. If given, only
        those time values are rendered (and must be present in the DataFrame).

    Returns
    -------
    str
        Path to the generated GIF file (``gif_filename``).

    Notes
    -----
    This function requires :mod:`pandas`, :mod:`plotly`, and :mod:`imageio`
    to be installed. If they are not available, an :class:`ImportError`
    is raised.
    """

    os.makedirs(frames_folder, exist_ok=True)

    # Determine which time steps to render
    unique_times = np.sort(dataframe["time"].unique())
    if frame_indices is not None:
        frame_times = [t for t in unique_times if t in frame_indices]
    else:
        frame_times = unique_times

    frame_paths: list[str] = []

    for time_value in frame_times:
        sub_dataframe = dataframe[dataframe["time"] == time_value]

        fig = px.scatter_3d(
            sub_dataframe,
            x="x1",
            y="x2",
            z="x3",
            color=color,
            size_max=10,
            width=1000,
            height=800,
        )
        fig.update_layout(
            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="z",
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
                zaxis=dict(showticklabels=False),
            )
        )

        camera = dict(
            eye=dict(x=1.5, y=-1.0, z=1.2),
            center=dict(x=0.0, y=0.0, z=0.0),
            up=dict(x=0.0, y=0.0, z=1.0),
        )
        fig.update_traces(marker=dict(opacity=0.75, size=6))
        fig.update_layout(scene_camera=camera)

        frame_path = os.path.join(frames_folder, f"frame_{int(time_value):04d}.png")
        fig.write_image(frame_path, format="png")
        frame_paths.append(frame_path)

    # Read frames and write GIF
    images = [imageio.imread(path) for path in frame_paths]
    imageio.mimsave(gif_filename, images, fps=fps)

    # Clean up temporary frames
    shutil.rmtree(frames_folder, ignore_errors=True)

    return gif_filename


# --- Benchmarking convenience plots (used by examples/benchmarking/benchmark.py) ---


def make_plots(df_results, f_opt_map=None, out_dir: str = "plots") -> None:
    """Generate lightweight benchmark plots.

    Saves, for each function, a scatter plot of best_f vs objective calls,
    grouped by method.

    This is intentionally dependency-light and safe to call even if other
    plotting utilities in this module are unused.
    """
    import os

    os.makedirs(out_dir, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except Exception:
        # plotting is optional; do not fail benchmarks if matplotlib isn't available
        return

    # Defensive: only plot if expected columns exist
    required = {"Function", "Method", "Objective calls", "best_f"}
    if not required.issubset(set(df_results.columns)):
        return

    for fn, dff in df_results.groupby("Function"):
        plt.figure()
        for method, dfm in dff.groupby("Method"):
            plt.scatter(
                dfm["Objective calls"], dfm["best_f"], label=str(method), alpha=0.7
            )
        plt.xlabel("Objective calls")
        plt.ylabel("best_f")
        plt.title(str(fn))
        plt.legend()
        path = os.path.join(out_dir, f"{fn}_bestf_vs_calls.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
