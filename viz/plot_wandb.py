"""
W&B Plotting Utility.

Fetch runs from Weights & Biases by config filters, group across seeds,
and plot mean +/- 95% confidence interval for any logged metric.

Usage as a library::

    from viz.plot_wandb import plot_grouped_metric

    plot_grouped_metric(
        project="history_dagger",
        y_key="eval/return",
        x_key="eval/dagger_step",
        groups=[
            {
                "label": "Standard DAgger",
                "filters": {"exp_name": "history_dagger", "env_name": "darkroom-easy-small"},
            },
            {
                "label": "Disagree (0.1, mask)",
                "filters": {
                    "disagreement_threshold": 0.1,
                    "label_strategy": "mask",
                    "env_name": "darkroom-easy-small",
                },
            },
        ],
        title="Eval Returns",
        xlabel="DAgger Step",
        ylabel="Return",
        save_path="plots/eval_returns.pdf",
    )

See ``__main__`` block for a full CLI example.
"""

import argparse
import math
import os
from typing import Callable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb


def fetch_runs(
    project: str,
    filters: dict,
    entity: Optional[str] = None,
) -> list:
    """Query W&B for runs whose config matches all key/value pairs in *filters*.

    Returns a list of ``wandb.apis.public.Run`` objects.
    """
    api = wandb.Api()
    mongo_filters = {f"config.{k}": v for k, v in filters.items()}
    path = f"{entity}/{project}" if entity else project
    return list(api.runs(path, filters=mongo_filters))


def extract_metric(
    run,
    y_key: str,
    x_key: str,
) -> tuple:
    """Pull ``(x, y)`` pairs from a run's full history.

    Uses ``scan_history`` to iterate over every logged row (no sampling),
    keeping only rows where both *x_key* and *y_key* are present and finite.

    Returns ``(x_vals, y_vals)`` as sorted 1-D numpy arrays.
    """
    xs, ys = [], []
    for row in run.scan_history(keys=[x_key, y_key]):
        x_val = row.get(x_key)
        y_val = row.get(y_key)
        if x_val is None or y_val is None:
            continue
        try:
            xf, yf = float(x_val), float(y_val)
        except (TypeError, ValueError):
            continue
        if math.isfinite(xf) and math.isfinite(yf):
            xs.append(xf)
            ys.append(yf)

    x_vals = np.asarray(xs)
    y_vals = np.asarray(ys)
    order = np.argsort(x_vals)
    return x_vals[order], y_vals[order]


def _aggregate_across_seeds(
    all_x: list,
    all_y: list,
) -> tuple:
    """Compute mean and 95 % CI across seed runs.

    If x-values are identical across runs, aggregation is exact.
    Otherwise values are linearly interpolated onto the union of x-grids.

    Returns ``(x, mean, ci)`` arrays of shape ``(num_x_points,)``.
    """
    if len(all_x) == 0:
        return np.array([]), np.array([]), np.array([])

    if len(all_x) == 1:
        return all_x[0], all_y[0], np.zeros_like(all_y[0])

    x_ref = all_x[0]
    all_same = all(
        len(x) == len(x_ref) and np.allclose(x, x_ref) for x in all_x
    )

    if all_same:
        x = x_ref
        ys = np.stack(all_y, axis=0)
    else:
        x = np.sort(np.unique(np.concatenate(all_x)))
        ys = np.stack(
            [np.interp(x, xr, yr) for xr, yr in zip(all_x, all_y)],
            axis=0,
        )

    n = ys.shape[0]
    mean = ys.mean(axis=0)
    std = ys.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
    ci = 1.96 * std / np.sqrt(n)
    return x, mean, ci


def plot_grouped_metric(
    groups: list,
    project: str,
    y_key: str,
    x_key: str,
    entity: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    colors: Optional[list] = None,
    figsize: tuple = (8, 5),
    integer_x: bool = False,
    show_legend: bool = True,
) -> plt.Figure:
    """Fetch W&B runs per group, aggregate across seeds, plot mean +/- 95 % CI.

    Args:
        groups: Each element is a dict with ``"label"`` (legend text) and
            ``"filters"`` (dict of W&B config key -> value).
        project: W&B project name.
        y_key: Metric column for the y-axis.
        x_key: Metric column for the x-axis.
        entity: W&B entity (team / username). ``None`` uses default.
        title: Plot title.
        xlabel / ylabel: Axis labels (default to the metric key names).
        save_path: Save the figure to this path if given.
        ax: Plot onto an existing Axes; if ``None`` a new figure is created.
        colors: One colour per group (cycles through default palette otherwise).
        figsize: Figure size when creating a new figure.
        integer_x: Force integer x-axis ticks.
        show_legend: If ``True`` (default), draw a legend on this axes.

    Returns:
        The matplotlib ``Figure``.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, group in enumerate(groups):
        label = group["label"]
        filters = group["filters"]
        if "color" in group:
            color = group["color"]
        elif colors and i < len(colors):
            color = colors[i]
        else:
            color = default_colors[i % len(default_colors)]

        runs = fetch_runs(project, filters, entity=entity)
        if not runs:
            print(f"  WARNING: no runs for '{label}' with filters {filters}")
            continue

        print(f"  Group '{label}': {len(runs)} run(s)")

        all_x, all_y = [], []
        for run in runs:
            try:
                x_vals, y_vals = extract_metric(run, y_key, x_key)
                if len(x_vals) > 0:
                    all_x.append(x_vals)
                    all_y.append(y_vals)
            except Exception as e:
                print(f"    Skipping run {run.name}: {e}")

        if not all_x:
            print(f"    No valid data for '{label}'")
            continue

        x, mean, ci = _aggregate_across_seeds(all_x, all_y)
        ax.plot(x, mean, label=label, color=color)
        ax.fill_between(x, mean - ci, mean + ci, alpha=0.2, color=color)

    ax.set_xlabel(xlabel or x_key)
    ax.set_ylabel(ylabel or y_key)
    if title:
        ax.set_title(title)
    if integer_x:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    if show_legend:
        ax.legend()
    ax.grid(True, alpha=0.3)

    if own_fig:
        fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved to {save_path}")

    return fig


def plot_grouped_metric_grid(
    env_names: list,
    groups_fn: Callable[[str], list],
    project: str,
    y_key: str,
    x_key: str,
    entity: Optional[str] = None,
    suptitle: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    save_path: Optional[str] = None,
    colors: Optional[list] = None,
    figsize_per_panel: tuple = (8, 4),
    ncols: int = 1,
    integer_x: bool = False,
    legend_ncol: Optional[int] = None,
) -> plt.Figure:
    """One subplot per environment, same group structure in each panel.

    Args:
        env_names: List of environment names (one subplot each).
        groups_fn: ``groups_fn(env_name)`` returns a list of group dicts
            (each with ``"label"`` and ``"filters"``).
        project / entity / y_key / x_key: Forwarded to
            :func:`plot_grouped_metric`.
        suptitle: Figure super-title.
        xlabel / ylabel: Shared axis labels.
        save_path: Save the full grid figure.
        colors: Shared colour list across panels.
        figsize_per_panel: ``(width, height)`` per subplot.
        ncols: Max columns in the grid (default 1 = stacked rows).
        integer_x: Force integer x-axis ticks.
        legend_ncol: Number of columns in the shared legend. Auto-chosen
            if ``None``.

    Returns:
        The matplotlib ``Figure``.
    """
    n = len(env_names)
    nrows = max(1, (n + ncols - 1) // ncols)
    fig_w = figsize_per_panel[0] * min(n, ncols)
    fig_h = figsize_per_panel[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    for idx, env_name in enumerate(env_names):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        groups = groups_fn(env_name)

        print(f"\n--- {env_name} ---")
        plot_grouped_metric(
            groups=groups,
            project=project,
            y_key=y_key,
            x_key=x_key,
            entity=entity,
            title=env_name,
            xlabel=xlabel,
            ylabel=ylabel,
            ax=ax,
            colors=colors,
            integer_x=integer_x,
            show_legend=False,
        )

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        ncol = legend_ncol or min(len(handles), 4)
        fig.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=ncol,
            fontsize="small",
            frameon=True,
        )

    fig.tight_layout(rect=[0, 0.08, 1, 0.96])

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved grid to {save_path}")

    return fig


def plot_strategy_env_grid(
    env_names: list,
    strategies: list,
    groups_fn: Callable[[str, str], list],
    project: str,
    y_key: str,
    x_key: str,
    entity: Optional[str] = None,
    suptitle: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize_per_panel: tuple = (6, 4),
    integer_x: bool = False,
    legend_ncol: Optional[int] = None,
    col_titles: Optional[list] = None,
) -> plt.Figure:
    """Env-rows x strategy-columns grid with shared y-axis per row.

    Args:
        env_names: One row per environment.
        strategies: One column per strategy.
        groups_fn: ``groups_fn(env_name, strategy)`` returns a list of
            group dicts for that cell.
        project / entity / y_key / x_key: Forwarded to
            :func:`plot_grouped_metric`.
        suptitle: Figure super-title.
        xlabel / ylabel: Shared axis labels.
        save_path: Save the figure.
        figsize_per_panel: ``(width, height)`` per subplot.
        integer_x: Force integer x-axis ticks.
        legend_ncol: Columns in the shared legend.
        col_titles: Column header labels (defaults to strategy names).

    Returns:
        The matplotlib ``Figure``.
    """
    nrows = len(env_names)
    ncols = len(strategies)
    fig_w = figsize_per_panel[0] * ncols
    fig_h = figsize_per_panel[1] * nrows
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_w, fig_h), squeeze=False,
        sharey="row",
    )

    titles = col_titles or strategies

    for r, env_name in enumerate(env_names):
        for c, strategy in enumerate(strategies):
            ax = axes[r][c]
            groups = groups_fn(env_name, strategy)

            show_ylabel = c == 0
            cell_ylabel = ylabel if show_ylabel else None
            cell_title = f"{env_name} — {titles[c]}" if r == 0 else env_name if c == 0 else None

            if r == 0:
                ax.set_title(titles[c], fontsize=11)

            print(f"\n--- {env_name} / {strategy} ---")
            plot_grouped_metric(
                groups=groups,
                project=project,
                y_key=y_key,
                x_key=x_key,
                entity=entity,
                xlabel=xlabel if r == nrows - 1 else None,
                ylabel=cell_ylabel,
                ax=ax,
                integer_x=integer_x,
                show_legend=False,
            )

            if c == 0:
                ax.annotate(
                    env_name, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 8, 0),
                    xycoords="axes fraction", textcoords="offset points",
                    ha="right", va="center", fontsize=11, fontweight="bold",
                    rotation=90,
                )

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)

    all_handles, all_labels = [], []
    seen = set()
    for c in range(ncols):
        for h, l in zip(*axes[0][c].get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                all_handles.append(h)
                all_labels.append(l)
    if all_handles:
        ncol_leg = legend_ncol or min(len(all_handles), 4)
        fig.legend(
            all_handles, all_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=ncol_leg,
            fontsize="small",
            frameon=True,
        )

    fig.tight_layout(rect=[0.04, 0.08, 1, 0.96])

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved grid to {save_path}")

    return fig


# ------------------------------------------------------------------
# CLI example
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot W&B metrics grouped across seeds",
    )
    parser.add_argument("--project", type=str, default="history_dagger")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument(
        "--envs",
        nargs="+",
        default=["darkroom-easy-small", "junction-3", "navigation-episodic"],
    )
    parser.add_argument("--save_dir", type=str, default="plots")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
    )
    parser.add_argument(
        "--label_strategies",
        nargs="+",
        default=["mask", "blend"],
    )
    cli_args = parser.parse_args()
    os.makedirs(cli_args.save_dir, exist_ok=True)

    sorted_thresholds = sorted(cli_args.thresholds)
    n_thresh = len(sorted_thresholds)

    strategy_cmaps = {"mask": "Reds", "blend": "Blues"}

    def _strategy_colors(strategy: str) -> list:
        """Sequential light-to-dark shades, one per sorted threshold."""
        cmap = plt.get_cmap(strategy_cmaps.get(strategy, "viridis"))
        lo, hi = 0.25, 0.85
        return [cmap(lo + (hi - lo) * i / max(1, n_thresh - 1))
                for i in range(n_thresh)]

    def make_groups(env_name: str, strategy: str) -> list:
        """Standard DAgger baseline + threshold curves for one strategy."""
        baseline = {
            "label": "Standard DAgger",
            "color": "gray",
            "filters": {
                "exp_name": "history_dagger",
                "env_name": env_name,
            },
        }
        palette = _strategy_colors(strategy)
        groups = [baseline]
        for j, thresh in enumerate(sorted_thresholds):
            # Early "mask" runs were logged before the label_strategy arg
            # existed, so the config value is null in W&B.
            filter_strategy = None if strategy == "mask" else strategy
            groups.append({
                "label": f"{strategy} thresh={thresh}",
                "color": palette[j],
                "filters": {
                    "exp_name": "history_dagger_disagreement",
                    "env_name": env_name,
                    "disagreement_threshold": thresh,
                    "label_strategy": filter_strategy,
                },
            })
        return groups

    # --- eval/return vs eval/dagger_step ---
    print("=" * 60)
    print("Plotting: eval/return vs eval/dagger_step")
    print("=" * 60)

    plot_strategy_env_grid(
        env_names=cli_args.envs,
        strategies=cli_args.label_strategies,
        groups_fn=make_groups,
        project=cli_args.project,
        y_key="eval/return",
        x_key="eval/dagger_step",
        entity=cli_args.entity,
        suptitle="Eval Returns vs DAgger Step",
        xlabel="DAgger Step",
        ylabel="Eval Return",
        save_path=os.path.join(cli_args.save_dir, "eval_returns_vs_dagger_step.pdf"),
        integer_x=True,
        col_titles=[s.capitalize() for s in cli_args.label_strategies],
    )

    # --- dataset/effective_size vs dagger_step ---
    # print("\n" + "=" * 60)
    # print("Plotting: dataset/effective_size vs dagger_step")
    # print("=" * 60)

    # plot_strategy_env_grid(
    #     env_names=cli_args.envs,
    #     strategies=cli_args.label_strategies,
    #     groups_fn=make_groups,
    #     project=cli_args.project,
    #     y_key="dataset/effective_size",
    #     x_key="dagger_step",
    #     entity=cli_args.entity,
    #     suptitle="Expert Query Rate vs DAgger Step",
    #     xlabel="DAgger Step",
    #     ylabel="Effective Dataset Size (%)",
    #     save_path=os.path.join(
    #         cli_args.save_dir, "effective_size_vs_dagger_step.pdf"
    #     ),
    #     integer_x=True,
    #     col_titles=[s.capitalize() for s in cli_args.label_strategies],
    # )

    plt.close("all")
    print(f"\nAll plots saved to {cli_args.save_dir}/")
