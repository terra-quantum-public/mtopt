"""Runner to compare TRC/MTC/TTOpt/... optimizers on various benchmark functions."""

from __future__ import annotations
import re
from typing import Iterable, List, Tuple, Callable, Dict, Optional
from tqdm import tqdm
import numpy as np
import pandas as pd
from numpy.random import SeedSequence
from helpers import run_trc, run_trc_z, run_mtc, run_ttopt, run_de, run_da


# A dictionary mapping optimizers' names to their corresponding runner functions
METHODS = {
    "TRC": run_trc,
    "TRC-z": run_trc_z,
    "MTC": run_mtc,
    "TTOpt": run_ttopt,
    "DE": run_de,
    "DA": run_da,
}


def _fmt_sci(x: float, digits: int = 1) -> str:
    """Format a float in scientific notation like 4.7e-6 (no zero-padded exponent)."""
    if not np.isfinite(x):
        return "nan"
    if x == 0.0:
        return "0"
    s = f"{float(x):.{digits}e}"  # e.g. '4.7e-06'
    return re.sub(r"e([+-])0*(\d+)$", r"e\1\2", s)  # -> '4.7e-6'


def _format_best_errors_table_for_csv(
    table: pd.DataFrame, digits: int = 1
) -> pd.DataFrame:
    """Return a copy with numeric cells rendered in sci notation; 'None' left as-is."""
    out = table.copy()
    for col in out.columns:
        vals = pd.to_numeric(out[col], errors="coerce")
        mask = vals.notna() & np.isfinite(vals)
        out.loc[mask, col] = vals[mask].astype(float).map(lambda v: _fmt_sci(v, digits))
    return out


def save_best_errors_csv(
    df_results: pd.DataFrame,
    f_opt_map: Dict[str, Optional[float]],
    path: str = "best_errors.csv",
    sci_digits: int = 2,
) -> pd.DataFrame:
    """Build the best-errors table, write it as CSV with scientific-notation cells."""
    table = make_best_errors_table(df_results, f_opt_map)
    table_fmt = _format_best_errors_table_for_csv(table, digits=sci_digits)
    table_fmt.to_csv(path, index=True)
    return table


def make_best_errors_table(
    df_results: pd.DataFrame,
    f_opt_map: Dict[str, Optional[float]],
) -> pd.DataFrame:
    """
    Build (rows=Method, cols=Function) with best error across all ranks & experiments,
    computed on the ORIGINAL objective scale; unknown-min columns filled with 'None'.
    """
    df = df_results.copy()
    df["f_opt"] = df["Function"].map(f_opt_map)

    all_functions = sorted(df["Function"].unique())
    all_methods = sorted(df["Method"].unique())

    known = df["f_opt"].notna() & np.isfinite(df["f_opt"])
    dfk = df[known].copy()

    if dfk.empty:
        table = pd.DataFrame(index=all_methods)
    else:
        dfk["error"] = (dfk["best_f"] - dfk["f_opt"]).abs()
        best = dfk.groupby(["Method", "Function"], as_index=False)["error"].min()
        table = best.pivot(index="Method", columns="Function", values="error").reindex(
            index=all_methods
        )

    for fn in all_functions:
        if fn not in getattr(table, "columns", []):
            table[fn] = np.nan
    table = table.reindex(columns=all_functions).astype(object)

    unknown_funcs = [
        fn
        for fn in all_functions
        if not (
            fn in f_opt_map and pd.notna(f_opt_map[fn]) and np.isfinite(f_opt_map[fn])
        )
    ]
    for fn in unknown_funcs:
        table[fn] = "None"

    return table


def compare_all(
    num_dimensions: int,
    num_grid_points: int,
    num_experiments: int,
    ranks: Iterable[int],
    num_sweeps: int,
    seed: int | list[int],
    tests: List[Tuple[str, Callable, list]],
    methods: Iterable[str],
    *,
    f_opt_map: Dict[str, Optional[float]] | None = None,
    thresholds: tuple[float, ...] = (1e-2, 1e-3, 1e-4, 1e-5),
    grid_type: str = "plain",
    qtt_levels: int = 16,
    qtt_base: int = 2,
    qtt_z: int = 3,
) -> pd.DataFrame:
    """
    Run benchmarks across (ranks x experiments x test functions x optimizers).

    Common Random Numbers (CRN): for a fixed (base seed, exp_idx), the spawned
    seed is shared across all methods.
    """
    if grid_type not in {"plain", "qtt"}:
        raise ValueError("grid_type must be 'plain' or 'qtt'")

    rows = []
    ranks_list = list(ranks)
    methods_list = list(methods)

    base_seeds = seed if isinstance(seed, list) else [int(seed)]
    exp_global = 0

    for base_seed in base_seeds:
        ss = SeedSequence(int(base_seed))
        if num_experiments is None:
            raise ValueError("num_experiments must be provided")
        children = ss.spawn(int(num_experiments))
        spawned_seeds = [int(c.generate_state(1, dtype=np.uint32)[0]) for c in children]

        for rank in ranks_list:
            print(f"running rank = {rank} (base_seed={base_seed})")
            for exp_local, spawned_seed in tqdm(list(enumerate(spawned_seeds))):
                for name, f, bounds in tests:
                    if len(bounds) != num_dimensions:
                        raise ValueError(
                            f"Bounds length ({len(bounds)}) != num_dimensions ({num_dimensions}) for {name}"
                        )

                    f_opt = None
                    if f_opt_map is not None:
                        f_opt = f_opt_map.get(name, None)

                    for method in methods_list:
                        if method not in METHODS:
                            raise KeyError(f"Unknown method: {method}")

                        calls, f_min, hits, _ = METHODS[method](
                            f,
                            bounds,
                            num_grid_points,
                            rank,
                            num_sweeps,
                            seed=spawned_seed,
                            grid_type=grid_type,
                            qtt_levels=qtt_levels,
                            qtt_base=qtt_base,
                            qtt_z=qtt_z,
                            f_opt=f_opt,
                            thresholds=thresholds,
                        )

                        rows.append(
                            {
                                "Function": name,
                                "Method": method,
                                "Rank": rank,
                                "Experiment": exp_global,
                                "Seed": spawned_seed,
                                "BaseSeed": int(base_seed),
                                "Objective calls": calls,
                                "best_f": f_min,
                                **{
                                    f"calls_to_{float(t):.0e}": (
                                        hits.get(float(t))
                                        if hits.get(float(t)) is not None
                                        else np.nan
                                    )
                                    for t in thresholds
                                },
                            }
                        )
                exp_global += 1

    df = (
        pd.DataFrame(rows)
        .sort_values(["Function", "Method", "Rank", "Experiment", "Seed"])
        .reset_index(drop=True)
    )
    return df
