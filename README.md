# `tq-mtopt` — tensor rank cross and matrix train cross optimizers.

[![Lint](https://github.com/terra-quantum-public/mtopt/actions/workflows/lint.yml/badge.svg)](https://github.com/terra-quantum-public/mtopt/actions/workflows/lint.yml)
[![Tests](https://github.com/terra-quantum-public/mtopt/actions/workflows/tests.yml/badge.svg)](https://github.com/terra-quantum-public/mtopt/actions/workflows/tests.yml)
[![Dependency Review](https://github.com/terra-quantum-public/mtopt/actions/workflows/dependency-review.yml/badge.svg)](https://github.com/terra-quantum-public/mtopt/actions/workflows/dependency-review.yml)

##### `tq-mtopt` is a lightweight Python library that implements several optimizers on discrete grids by extending matrix cross approximation to several types of tensor networks. The first, **Tensor Rank Cross (TRC)** is a cross approximation for tensor rank decomposition (also called canonical diadic decomposition (Candecomp)). The second, **Matrix Train Cross (MTC)** approximation is a cross approximation that combines features of TRC and Tensor Train (also called Matrix Product State (MPS)) decomposition. Finally, the package also includes a general Tree Tensor Network (TTN) optimizer that can be applied to any user-defined tree structure. The methods build low‑rank tensor representations of a function from a small number of function evaluations, then extract candidate optima directly from the representations. The package is designed specifically for high-dimensional functions with multiple local minima, where each function evaluation is computationally expensive.

## Installation

To install the current release from the Terra Quantum package registry, use [pip](https://pip.pypa.io/en/stable/):

```bash
pip install tq-mtopt --extra-index-url https://europe-python.pkg.dev/bright-primacy-486519-b9/tq-public/simple
```

Or clone the repository and use [poetry](https://python-poetry.org/):

```bash
poetry install
```

### Optional dependencies

The core library has no plotting dependencies. Functions in `tq_mtopt.plot` require one or more of the following, depending on what you use:

| Extra | Required by |
|---|---|
| `plotly` | 3D point cloud and animated scatter plots |
| `matplotlib` | Tensor-train and tree tensor network diagrams |
| `imageio` | Exporting animations (GIF/video) |

Install only what you need:

```bash
pip install plotly matplotlib imageio
```

## Minimal example

```python
import numpy as np
from tq_mtopt.grid import Grid, tensor_network_grid, build_node_grid
from tq_mtopt.network import balanced_tree, root
from tq_mtopt.optimization import (
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

rank = 6
epochs = 8

# --- TRC ---
trc = TensorRankOptimization(primitives)
skel_trc = random_grid_points(primitives, n_samples=10, seed=42)
skel_trc = trc.optimize(skel_trc, function=sphere, num_epochs=epochs)
vals_trc = skel_trc.evaluate(sphere)
i_trc = int(np.argmin(vals_trc))
x_trc, f_trc = skel_trc.grid[i_trc], float(vals_trc[i_trc])
print("TRC  -> x* =", np.round(x_trc, 4), "f* =", f"{f_trc:.6f}")

# --- MTC ---
mtc = MatrixTrainOptimization(primitives)
skel_mtc = random_grid_points(primitives, n_samples=10, seed=42)
skel_mtc = mtc.optimize(skel_mtc, function=sphere, num_epochs=epochs)
vals_mtc = skel_mtc.evaluate(sphere)
i_mtc = int(np.argmin(vals_mtc))
x_mtc, f_mtc = skel_mtc.grid[i_mtc], float(vals_mtc[i_mtc])
print("MTC  -> x* =", np.round(x_mtc, 4), "f* =", f"{f_mtc:.6f}")

# --- TTN ---
G = balanced_tree(num_leaves=3, rank=rank, phys_dim=len(x0))
G = tensor_network_grid(G, primitive_grid=[x0, x1, x2])
obj = Objective(sphere)
G = tree_tensor_network_optimize(G, obj, num_sweeps=epochs)
df = obj.logger.dataframe
coord_cols = [c for c in df.columns if c.startswith("x")]
top_k = df.nsmallest(3, "f")[coord_cols + ["f"]].reset_index(drop=True)
top_k.index += 1
print("TTN top-3 optima:")
print(top_k.to_string())
```

## Examples

The examples module contains full workflows that demonstrate typical use cases. Each example is fully documented and serves as a starting point for building your own experiments. The package has been tested on macOS and Linux and does not currently support Windows.

## Cite
If you happen to find `tq-mtopt` useful in your work, please consider supporting development by citing it. (Here goes the BibTeX entry for our future JOSS paper as well for the algorithm paper.)
```
@article{x,
  title={x},
  author={x},
  journal={x},
  volume={x},
  number={x},
  pages={x},
  year={x}
}
```

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide — including how to fork the repository, set up an upstream remote, open a pull request, and our commit message conventions.

To report bugs or request features, open an [issue](https://github.com/terra-quantum-public/mtopt/issues).

## License

This project is licensed under the terms described in [LICENSE.md](LICENSE.md).
See [NOTICE.md](NOTICE.md) for additional attribution and third-party notices.

## Documentation

Full documentation is available at [tq-mtopt.readthedocs.io](https://tq-mtopt.readthedocs.io/en/latest/).
