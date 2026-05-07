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
