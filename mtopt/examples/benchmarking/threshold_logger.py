"""
Utilities for true calls-to-threshold logging.

We define "true calls-to-threshold" as the earliest scalar objective evaluation
index (1-based) at which the running best error

    eps = |f_best - f_opt|

first becomes <= a given threshold.

This module is optimizer-agnostic: you wrap the *raw* objective with
`RawEvalLogger`, pass it to whatever optimizer, and later compute the first-hit
times from the recorded raw evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Sequence
import numpy as np


@dataclass
class RawEvalLogger:
    """Wrap a callable f(x) and record raw objective values in evaluation order.

    Supports scalar and batched evaluations:
      - if f(x) returns a scalar -> one evaluation is recorded,
      - if f(x) returns an array-like -> all entries are recorded in row-major order.
    """

    f: Callable[[Any], Any]

    def __post_init__(self) -> None:
        self.values: list[float] = []

    def __call__(self, x: Any) -> Any:
        y = self.f(x)
        arr = np.asarray(y)

        if arr.ndim == 0:
            self.values.append(float(arr))
        else:
            self.values.extend([float(v) for v in arr.reshape(-1)])
        return y

    @property
    def calls(self) -> int:
        return len(self.values)


def first_hit_calls(
    values: Sequence[float],
    f_opt: float,
    thresholds: Iterable[float],
) -> Dict[float, Optional[int]]:
    """Compute first-hit evaluation indices for running-best error vs thresholds.

    Parameters
    ----------
    values:
        Raw objective values in evaluation order (length = #scalar evals).
    f_opt:
        Known global optimum value f*.
    thresholds:
        Thresholds tau for which to find the first eval index where
        |f_best - f*| <= tau, with f_best the running minimum.

    Returns
    -------
    dict:
        {tau: first_call_index_or_None}, where call indices are 1-based.
    """
    thr_list = [float(t) for t in thresholds]
    hits: Dict[float, Optional[int]] = {t: None for t in thr_list}

    if not np.isfinite(f_opt):
        return hits

    best = float("inf")
    remaining = set(thr_list)

    for i, v in enumerate(values):
        vv = float(v)
        if vv < best:
            best = vv
        err = abs(best - float(f_opt))

        done_now = []
        for t in remaining:
            if err <= t:
                hits[t] = i + 1  # 1-based
                done_now.append(t)
        for t in done_now:
            remaining.remove(t)

        if not remaining:
            break

    return hits
