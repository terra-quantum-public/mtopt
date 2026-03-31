# `mtopt` — tensor rank cross and matrix train cross optimizers.

[![lint](https://github.com/terra-quantum-io/mtopt/actions/workflows/lint.yml/badge.svg)](https://github.com/terra-quantum-io/mtopt/actions/workflows/lint.yml)
[![tests](https://github.com/terra-quantum-io/mtopt/actions/workflows/tests.yml/badge.svg)](https://github.com/terra-quantum-io/mtopt/actions/workflows/tests.yml)
[![dependency review](https://github.com/terra-quantum-io/mtopt/actions/workflows/dependency-review.yml/badge.svg)](https://github.com/terra-quantum-io/mtopt/actions/workflows/dependency-review.yml)
[![CI](https://github.com/terra-quantum-io/mtopt/actions/workflows/ci.yml/badge.svg)](https://github.com/terra-quantum-io/mtopt/actions/workflows/ci.yml)

##### `mtopt` is a lightweight Python library that implements several optimizers on discrete grids by extending matrix cross approximation to several types of tensor networks. The first, **Tensor Rank Cross (TRC)** is a cross approximation for tensor rank decomposition (also called canonical diadic decomposition (Candecomp)). The second, **Matrix Train Cross (MTC)** approximation is a cross approximation that combines features of TRC and Tensor Train (also called Matrix Product State (MPS)) decomposition. Finally, the package also includes a general Tree Tensor Network (TTN) optimizer that can be applied to any user-defined tree structure. The methods build low‑rank tensor representations of a function from a small number of function evaluations, then extract candidate optima directly from the representations. The package is designed specifically for high-dimensional functions with multiple local minima, where each function evaluation is computationally expensive.

## Installation

To install the current release, use the package manager [pip](https://pip.pypa.io/en/stable/).

```bash
pip install mtopt
```

Otherwise, you can clone the repository and use [poetry](https://python-poetry.org/).

```bash
poetry install
```

## Minimal example

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

# Extract best candidate from the root node’s grid
build_node_grid(G)
root_grid = G.nodes[root(G)]["grid"]
vals_ttn = root_grid.evaluate(sphere)
i_ttn = int(np.argmin(vals_ttn))
x_ttn, f_ttn = root_grid.grid[i_ttn], float(vals_ttn[i_ttn])
print("TTN  -> x* =", np.round(x_ttn, 4), "f* =", f"{f_ttn:.6f}")
```

## Examples

The examples module contains full workflows that demonstrate typical use cases. Each example is fully documented and serves as a starting point for building your own experiments. The package has been tested on macOS and Linux and does not currently support Windows.

## Cite
If you happen to find `mtopt` useful in your work, please consider supporting development by citing it. (Here goes the BibTeX entry for our future JOSS paper as well for the algorithm paper.)
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

## Contribution guidelines

If you want to contribute to `mtopt`, be sure to follow GitHub's contribution guidelines.
This project adheres to our [code of conduct](https://github.com/terra-quantum-io/mtopt/blob/main/CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

We use [GitHub issues](https://github.com/terra-quantum-io/mtopt/issues) for
tracking requests and bugs, please direct specific questions to the maintainers.

The `mtopt` project strives to abide by generally accepted best practices in
open-source software development, such as:

*   apply the desired changes and resolve any code
    conflicts,
*   run the tests and ensure they pass,
*   build the package from source.

Developers may find the following guidelines useful:

- **Running tests.**
  Tests are executed using [pytest](https://docs.pytest.org/):
  ```bash
  pytest tests
  ```

- **Building documentation.**
  Documentation is built with [Sphinx](https://www.sphinx-doc.org/).
  A convenience script is provided:

  ```bash
  ./generate_docs.sh
  ```

- **Coding style.**
  The project follows the [Black](https://black.readthedocs.io/en/stable/) code style.
  Please run Black before submitting a pull request:

  ```bash
  black .
  ```

## License

This project is licensed under the custom [Terra Quantum License](https://terraquantum.io/content/legal/eula-tq42-tqml/).

## Documentation

Full documentation is available at [mtopt.readthedocs.io](https://mtopt.readthedocs.io/en/latest/).
