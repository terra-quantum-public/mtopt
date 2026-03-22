"""Benchmarking functions and some corresponding helpers for optimization algorithms."""

from __future__ import annotations
from functools import lru_cache
from typing import Callable, Dict, List, Tuple, Optional
import math
import numpy as np


def ackley(x):
    """Global min 0 at x=0, domain [-32.768, 32.768]^num_variables. F1 from the TTOpt paper."""
    x = np.asarray(x)
    num_variables = x.size
    a, b, c = 20.0, 0.2, 2 * np.pi
    s1 = np.sum(x**2) / num_variables
    s2 = np.sum(np.cos(c * x)) / num_variables
    return -a * np.exp(-b * np.sqrt(s1)) - np.exp(s2) + a + math.e


def alpine1(x):
    """Global min 0 at x=0, domain [-10, 10]^num_variables. F2 from the TTOpt paper."""
    x = np.asarray(x)
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x))


def brown(x):
    """Global min 0 at x=0, domain [-1, 4]^num_variables. F3 from the TTOpt paper."""
    x = np.asarray(x)
    s = 0.0
    for i in range(x.size - 1):
        xi, xj = x[i], x[i + 1]
        s += (xi**2) ** (xj**2 + 1) + (xj**2) ** (xi**2 + 1)
    return s


def exp_neg_norm2(x):
    """Global min -1 at x=0, domain [-1, 1]^num_variables. F4 from the TTOpt paper."""
    x = np.asarray(x)
    return -np.exp(-np.sum(x**2))


def griewank(x):
    """Global min 0 at x=0, domain [-600, 600]^num_variables. F5 from the TTOpt paper."""
    x = np.asarray(x, dtype=float)
    i = np.arange(1, x.size + 1, dtype=float)
    return np.sum(x * x) / 4000.0 - np.prod(np.cos(x / np.sqrt(i))) + 1.0


def michalewicz(x, m: int = 10):
    """
    Global mins: ≈ -9.66015 for 10 dimensions, ≈ -4.687656 for 5 dimensions,
    domain [0, π]^num_variables. F6 from the TTOpt paper.
    """
    x = np.asarray(x, dtype=float)
    i = np.arange(1, x.size + 1, dtype=float)
    return -np.sum(np.sin(x) * (np.sin(i * x**2 / np.pi)) ** (2 * m))


def qing(x):
    """Global min 0 at x_i=sqrt(i), domain [0, 500]^num_variables. F7 from the TTOpt paper."""
    x = np.asarray(x, dtype=float)
    i = np.arange(1, x.size + 1, dtype=float)
    return np.sum((x**2 - i) ** 2)


def rastrigin(x):
    """Global min 0 at x=0, domain [-5.12, 5.12]^num_variables. F8 from the TTOpt paper."""
    x = np.asarray(x, dtype=float)
    return 10.0 * x.size + np.sum(x * x - 10.0 * np.cos(2 * np.pi * x))


def schaffer(x):
    """Global min 0 at x=0, domain [-100, 100]^num_variables. F9 from the TTOpt paper."""
    x = np.asarray(x, dtype=float)
    xi2 = x[:-1] ** 2 + x[1:] ** 2
    return np.sum(0.5 + (np.sin(np.sqrt(xi2)) ** 2 - 0.5) / (1.0 + 0.001 * xi2) ** 2)


def schwefel(x):
    """Global min 0 at x≈420.968746, domain [-500, 500]^num_variables. F10 from the TTOpt paper."""
    x = np.asarray(x, dtype=float)
    return 418.9829 * x.size - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def rosenbrock(x, a=1.0, b=100.0):
    """Global min 0 at x=(1,...,1), domain [0, 2]^num_variables."""
    x = np.asarray(x, dtype=float)
    return np.sum(b * (x[1:] - x[:-1] ** 2) ** 2 + (a - x[:-1]) ** 2)


@lru_cache(maxsize=None)
def _multiwell_params(
    D: int, seed: int
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Cached parameters for Multiwell:
      centers: (m, D)
      amplitudes: (m,)
      df: degrees of freedom (float)
      power: exponent (float)
      max_amp: maximum amplitude, i.e. max_k a_k (float)

    The global minimum of f(x) = min_k w_k(x) + max_amp is 0 by construction:
    w_k(x) = -a_k / (1 + ||x-c_k||^2/df)^power >= -a_k, so min_k w_k >= -max_amp,
    and equality holds at the center of the deepest well.
    """
    m = D
    df = float(D)
    power = float(D)

    rng = np.random.RandomState(int(seed))  # deterministic across runs
    centers = rng.uniform(-4.0, 4.0, size=(m, D)).astype(float)

    # Strictly increasing amplitudes -> unique deepest well at index m-1.
    amplitudes = (0.5 + 1.5 * (np.arange(1, m + 1, dtype=float) / m)).astype(float)

    max_amp = float(np.max(amplitudes))

    # Make cached arrays immutable to prevent accidental mutation of shared state.
    centers.setflags(write=False)
    amplitudes.setflags(write=False)
    return centers, amplitudes, df, power, max_amp


def multiwell(x, seed: int = 42) -> float:
    """
    Multiwell benchmark (Student-t wells), with **known global optimum value**.

    For each well k:
        w_k(x) = -a_k / (1 + ||x - c_k||^2 / df)^power,  with df=D and power=D.

    Since (1 + ...)^power >= 1, we have w_k(x) >= -a_k, hence
        min_k w_k(x) >= -max_k a_k.

    We return:
        f(x) = min_k w_k(x) + max_k a_k,

    so f(x) >= 0 for all x and f(c_{k*}) = 0 at the center of the deepest well.

    Domain: [-5, 5]^D recommended
    Global minimum (value): 0.0 (by construction)
    """
    x = np.asarray(x, dtype=float)
    D = int(x.size)

    centers, amplitudes, df, power, max_amp = _multiwell_params(D, int(seed))

    diff = centers - x[None, :]
    dist_sq = np.sum(diff * diff, axis=1)  # (m,)
    wells = -amplitudes / (1.0 + dist_sq / df) ** power  # (m,)
    return float(np.min(wells) + max_amp)


# Registry of functions and default bounds per dimension
FUNCTION_REGISTRY: Dict[str, Tuple[Callable, Tuple[float, float]]] = {
    "Ackley": (ackley, (-32.768, 32.768)),
    "Alpine1": (alpine1, (-10.0, 10.0)),
    "Brown": (brown, (-1.0, 4.0)),
    "Exponential": (exp_neg_norm2, (-1.0, 1.0)),
    "Griewank": (griewank, (-600.0, 600.0)),
    "Michalewicz": (michalewicz, (0.0, np.pi)),
    "Qing": (qing, (0.0, 500.0)),
    "Rastrigin": (rastrigin, (-5.12, 5.12)),
    "Schaffer": (schaffer, (-100.0, 100.0)),
    "Schwefel": (schwefel, (-500.0, 500.0)),
    "Rosenbrock": (rosenbrock, (0.0, 2.0)),
    "Multiwell": (multiwell, (-5.0, 5.0)),
}


# Known global minima for Michalewicz (m=10) at specific dimensions
_MICHALEWICZ_KNOWN = {
    5: -4.687656,  # ≈ value for num_dims=5
    10: -9.66015,  # ≈ value for num_dims=10
}


# Known global minima (value only). Unknown/depends on D -> None.
F_OPT: Dict[str, float | None] = {
    "Ackley": 0.0,
    "Alpine1": 0.0,
    "Brown": 0.0,
    "Exponential": -1.0,
    "Griewank": 0.0,
    "Michalewicz": None,  # num_dims-dependent (e.g., ≈-9.66015 for num_dims=10, ≈-4.687656 for 5)
    "Qing": 0.0,
    "Rastrigin": 0.0,
    "Schaffer": 0.0,
    "Schwefel": 0.0,
    "Rosenbrock": 0.0,
    "Multiwell": 0.0,  # by construction (see multiwell shift)
}


def resolve_f_opt_map(
    base: Dict[str, Optional[float]], num_dimensions: int
) -> Dict[str, Optional[float]]:
    """
    Return a copy of base F_OPT with Michalewicz filled for D in {5,10} (m=10),
    otherwise leave it as None.
    """
    out = dict(base)
    if "Michalewicz" in out:
        out["Michalewicz"] = _MICHALEWICZ_KNOWN.get(num_dimensions, None)
    return out


def get_tests(num_variables: int, names: List[str]):
    """Get list of benchmark functions with bounds for given number of variables."""
    tests = []
    for nm in names:
        if nm not in FUNCTION_REGISTRY:
            raise KeyError(f"Unknown function: {nm}")
        fn, (lo, hi) = FUNCTION_REGISTRY[nm]
        bounds = [(lo, hi)] * num_variables
        tests.append((nm, fn, bounds))
    return tests
