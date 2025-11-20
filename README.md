# `mtopt` — Tensor rank cross and matrix train optimizers.


##### `mtopt` is a lightweight Python library that implements two optimizers for black-box optimization on discrete grids: Tensor Rank Cross (TRC) and Matrix Train (MT). Both methods build low‑rank tensor representations of the objective from a small number of function evaluations, then extract candidate optima directly from the representations. The package targets use‑cases where objective gradients are unavailable or unreliable, and where each function call is expensive (simulation, hardware loop, or batched evaluation).

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
# TBD
```

## Examples

The examples module contains full workflows that demonstrate typical use cases, such as TBD. Each example is fully documented and serves as a starting point for building your own experiments.
The package has been tested on macOS and Linux and does not currently support Windows.

## Cite
If you happen to find `mtopt` useful in your work, please consider supporting development by citing it. (Here goes the BibTeX entry for our future JOSS paper.)
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

If you want to contribute to `mdopt`, be sure to follow GitHub's contribution guidelines.
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

- **Pre-commit hooks.**
  [Pre-commit](https://pre-commit.com/) hooks are configured to enforce consistent style automatically.
  To enable them:

  ```bash
  pre-commit install
  ```

## License

This project is licensed under the custom [Terra Quantum License](https://terraquantum.io/content/legal/eula-tq42-tqml/).

## Documentation

Full documentation is available at [mtopt.readthedocs.io](https://mtopt.readthedocs.io/en/latest/).
