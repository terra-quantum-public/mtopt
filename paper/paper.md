---
title: 'mtopt: tensor rank cross and matrix train cross representations for sample‑efficient black‑box optimization'
tags:
  - python
  - optimization
  - tensor networks
authors:
  - name: Aleksandr Berezutskii
    orcid: 0009-0002-8149-135X
    affiliation: 1
  - name: Roman Ellerbrock
    orcid: 0000-0003-3555-6263
    affiliation: 1
  - name: Alexander C. Paul
    orcid: 0000-0002-7547-445X
    affiliation: 1
affiliations:
 - name: Terra Quantum AG, Freddie-Mercury Str. 5, Munich, DE 80979
   index: 1
date: 01 February 2026
bibliography: paper.bib
---

# Summary

**mtopt** is a lightweight Python library that implements two optimizers for black-box optimization on discrete grids by extending matrix cross approximation to two distinct types of tensor networks. The first, **Tensor Rank Cross (TRC)** is a cross approximation for tensor rank decomposition (also called canonical diadic decomposition (Candecomp)). The second, **Matrix Train Cross (MTC)** approximation is a cross approximation that combines features of TRC and Tensor Train (also called Matrix Product State (MPS)) decomposition. Both methods build low‑rank tensor representations of a function from a small number of function evaluations, then extract candidate optima directly from the representations. The package is designed specifically for high-dimensional functions with multiple local minima, where each function evaluation is computationally expensive.


# Statement of need

Most real-world optimization problems are high-dimensional and nonconvex, with little room for extra evaluations. General‑purpose gradient‑free methods (for example, direct search or evolutionary strategies) are robust but often require many evaluations to locate good solutions. Tensor network decompositions offer an orthogonal strategy: they approximate the function on a discrete grid with a compact representation. In this structure, minima and maxima are obtained automatically as part of the compression[oseledets2010tt][sozykin2022ttopt][dolgov2025tensor]. Here, we present a lightweight toolchain that extends the idea of cross approximation to different tensor network architectures including tree tensor networks, tensor rank decompositions, and a combination of tensor trains and tensor rank decompositions (what we call matrix trains).

# Functionality

* **Tensor Rank Cross (TRC) optimizer.** Performs cross approximation for rank-(r) tensor rank decomposition on a user-supplied product grid. Each sweep alternates one-leg updates (cross-approximation on one dimension while holding others fixed) with selection via maximum-volume principle or linear-sum-assignment on the cross matrix. It returns the updated rank-(r) skeleton (pivot set) and the current candidate minima/maxima.

* **Matrix Train Cross (MTC) optimizer.** Optimizes a function using cross approximation in an (N)-site matrix-train representation (a hybrid between tensor train and tensor rank decomposition) directly on the grid. The core step partitions dimensions into two groups and recombines them to form an (r\times r) cross matrix. It then uses the Hungarian (linear-sum) assignment to pick the next rank-(r) slice, followed by one-leg updates described above.

* **Tree Tensor Network (TTN) optimizer.** Performs cross approximation on a user-defined Tree Tensor Network for a given function. The library lets you specify the tree structure manually, or it can construct one automatically as either a nearly balanced tree or a chain (the tensor-train case). Each pass forms local Cartesian subgrids at internal nodes, evaluates the objective, builds edge-wise cross matrices, selects pivots (maximum-volume or linear-sum assignment), and applies one-leg updates. It stops when pivots stabilize or the evaluation budget is reached.

* **Hyperparameters, budgets, and logging.** Topology, ranks, sweeps, 1D grid parameters (uniform or custom), and seeds are user-controlled. Function evaluations are cached and recorded, and the framework tracks objective-call counts. Computational and evaluation cost are bounded by the chosen rank, number of sweeps, and grid size. Deterministic seeding enables fair common-random-numbers comparisons across methods.

* **Benchmarks & baselines.** The repo includes a benchmarking suite with CSV outputs and plots, comparing TRC and MTC against TTOpt and SciPy baselines (Differential Evolution and Dual Annealing). The benchmarks can conveniently be extended to other optimizers.


# Minimal example

```python
import numpy as np
from mtopt.grid import Grid, tensor_network_grid, build_node_grid
from mtopt.network import balanced_tree, root
from mtopt.optimization import (
    TensorRankOptimization,
    MatrixTrainOptimization,
    Objective,
    random_grid_points,
    tree_tensor_network_optimize,
)

# Black-box objective (accept **kwargs to ignore optimizer metadata like `epoch`)
def sphere(x, **_):
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**2))

# 1) Primitive 1D grids
x0 = np.linspace(-2.0, 2.0, 51)
x1 = np.linspace(-2.0, 2.0, 51)
x2 = np.linspace(-2.0, 2.0, 51)
g0, g1, g2 = Grid(x0, coords=0), Grid(x1, coords=1), Grid(x2, coords=2)
primitives = [g0, g1, g2]

r = 6
epochs = 8

# --- TRC ---
trc = TensorRankOptimization(primitives, r=r)
skel_trc = random_grid_points(primitives, r=r, seed=42)
skel_trc = trc.optimize(skel_trc, function=sphere, num_epochs=epochs)
vals_trc = skel_trc.evaluate(sphere)
i_trc = int(np.argmin(vals_trc))
x_trc, f_trc = skel_trc.grid[i_trc], float(vals_trc[i_trc])
print("TRC  -> x* =", np.round(x_trc, 4), "f* =", f"{f_trc:.6f}")

# --- MTC ---
mtc = MatrixTrainOptimization(primitives, r=r)
skel_mtc = random_grid_points(primitives, r=r, seed=42)
skel_mtc = mtc.optimize(skel_mtc, function=sphere, num_epochs=epochs)
vals_mtc = skel_mtc.evaluate(sphere)
i_mtc = int(np.argmin(vals_mtc))
x_mtc, f_mtc = skel_mtc.grid[i_mtc], float(vals_mtc[i_mtc])
print("MTC  -> x* =", np.round(x_mtc, 4), "f* =", f"{f_mtc:.6f}")

# --- TTN ---
G = balanced_tree(num_leaves=3, rank=r, phys_dim=len(x0))
G = tensor_network_grid(G, primitive_grid=[x0, x1, x2])
obj = Objective(sphere)
G = tree_tensor_network_optimize(G, obj, num_sweeps=epochs)

# Extract best candidate from the root node’s grid
build_node_grid(G)
root_grid = G.nodes[root(G)]["grid"]
vals_ttn = root_grid.evaluate(sphere)
i_ttn = int(np.argmin(vals_ttn))
x_ttn, f_ttn = root_grid.grid[i_ttn], float(vals_ttn[i_ttn])
print("TTN  -> x* =", np.round(x_ttn, 4), "f* =", f"{f_ttn:.6f}")
```

Following the minimal example above, users can easily adapt the objective, discretization, rank schedule, and evaluation budget to their application. Additional usage examples as well as the detailed API documentation are provided in the
[mtopt Documentation](https://mtopt.readthedocs.io/en/latest/).

# References
