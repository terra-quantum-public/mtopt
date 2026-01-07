---
title: 'mtopt: tensor rank cross and matrix train representations for sample‑efficient black‑box optimization'
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
date: 01 November 2025
bibliography: paper.bib
---

# Summary

**mtopt** is a lightweight Python library that implements two optimizers for black-box optimization on discrete grids: **Tensor Rank Cross (TRC)** and **Matrix Train (MT)**. Both methods build low‑rank tensor representations of the objective from a small number of function evaluations, then extract candidate optima directly from the representations. The package targets use‑cases where objective gradients are unavailable or unreliable, and where each function call is expensive.

# Statement of need

Most real-world optimization tasks are high-dimensional and nonconvex, with little room for extra evaluations. General‑purpose gradient‑free methods (e.g., direct search or evolutionary strategies) are robust but often require many evaluations to locate good solutions. Low‑rank tensor decompositions offer an orthogonal strategy: approximate the discretized objective on a grid with a compact representation and use its structure to propose maximizers/minimizers efficiently. Despite promising results in recent literature [oseledets2010tt][sozykin2022ttopt][dolgov2025tensor], there remains a gap between theory and a practical, lightweight toolchain that exposes these ideas behind a consistent API and strict evaluation budgets.

# Functionality

* **Tensor Rank Cross (TRC) optimizer.** Performs rank-(r) cross updates on a user-supplied product grid. Each sweep alternates **one-leg mutations** (cross-sampling one primitive while holding others fixed) with **selection** via max-volume or assignment on the evaluated candidate matrix. It returns the updated rank-(r) skeleton (pivot set) and the current incumbent. Rank, number of sweeps, grid resolution, and seeds are user-controlled.

* **Matrix Train (MT) optimizer.** Optimizes an (N)-site matrix-train directly on the grid. The core step **recombines** left/right blocks to form an (r\times r) candidate matrix and uses the **Hungarian (linear-sum) assignment** to pick the next rank-(r) slice, interleaved with one-leg mutation updates. Grouped/segment assignments are supported via a simple grouped assignment routine (experimental).

* **Search spaces.** Any space that can be discretized into **per-dimension 1D grids**: continuous boxes (e.g., uniform linspaces), categorical/discrete sets, or mixed discrete–continuous spaces by mixing primitives. Custom per-dimension grids are accepted.

* **Budgets & reproducibility.** Work is bounded by rank, number of sweeps, and grid size; the framework tracks objective-call counts and supports **deterministic seeding** for fair comparisons (common-random-numbers across methods).

* **Benchmarks & baselines.** The repo includes a benchmarking suite with CSV outputs and plots, comparing TRC/MT against **TTOpt** and **SciPy** baselines (Differential Evolution and Dual Annealing).


# Minimal example

```python
import numpy as np
from mtopt.grid import Grid, cartesian_product

# Black-box objective
def sphere(x):
    x = np.asarray(x, dtype=float)
    return np.sum(x**2)

# 1) Primitive 1D grids (3 dimensions)
g0 = Grid(np.linspace(-2.0, 2.0, 51), coords=0)
g1 = Grid(np.linspace(-2.0, 2.0, 51), coords=1)
g2 = Grid(np.linspace(-2.0, 2.0, 51), coords=2)

# 2) 3D Cartesian product and random downsampling
grid3d = cartesian_product([g0, g1, g2]).random_subset(2000)  # may not include the origin

# 3) Evaluate objective and extract the best point from the sampled grid
values = grid3d.evaluate(sphere)
i_star = int(np.argmin(values))
x_star = grid3d.grid[i_star]        # point in R^3
f_star = float(values[i_star])

print(f"sampled-grid best value ≈ {f_star:.6f}")
print("x* =", np.round(x_star, 4))

# 4) Compare with the true minimum of the continuous problem
x_true = np.zeros(3)
f_true = 0.0
dist = np.linalg.norm(x_star - x_true)
gap = f_star - f_true

print(f"true minimum = {f_true:.6f} at x_true = {x_true}")
print(f"distance to true minimizer = {dist:.4e}")
print(f"difference from the minimum = {gap:.4e}")
```

Following the minimal example above, users can easily adapt the objective, discretization (grid density or node type), rank schedule, and evaluation budget to their application. Additional usage examples as well as the detailed API documentation are provided in the
[mtopt Documentation](https://mtopt.readthedocs.io/en/latest/).

# References
