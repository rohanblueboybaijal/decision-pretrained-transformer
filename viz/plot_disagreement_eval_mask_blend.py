"""
Plot eval return and expert query rate vs DAgger step from Weights & Biases.

For one environment, draws **mask | blend** columns; each column has **three**
rows: eval return (top), expert query rate over the **full replay buffer** (middle),
and **expert-labeled share for the current DAgger iteration only** (bottom;
``dataset/expert_query_rate_this_step``). Coloured curves are one per disagreement
threshold. Standard history DAgger (``train_history_dagger.py``, ``exp_name``
default ``history_dagger``) is drawn in **black** on the first two rows when those
runs exist (eval return and ``dataset/effective_size`` on the buffer query row).

Expects runs logged by ``train_history_dagger_disagreement.py`` with metrics
``eval/return``, ``eval/return_std``, ``eval/dagger_step``, per-step
``dataset/expert_query_rate``, and ``dataset/expert_query_rate_this_step`` (with
``dagger_step``). Baseline runs use ``dataset/effective_size`` vs ``dagger_step``
(typically 100%) on the buffer row only.

Usage (from repo root)::

    python -m viz.plot_disagreement_eval_mask_blend \\
        --env_name darkroom-easy-small \\
        --project history_dagger_v1 \\
        --save_path plots/disagreement_eval_mask_blend.pdf

Requires ``wandb`` logged in (``wandb login``).
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from viz.plot_wandb import fetch_runs


def extract_scalar_vs_dagger_step(
    run,
    y_key: str,
    x_key: str = "dagger_step",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aligned (x, y) where *x_key* is ``dagger_step`` and *y_key* is any metric logged
    on that same row (e.g. ``dataset/expert_query_rate``, ``dataset/effective_size``).

    Duplicate *x* values: last occurrence wins. No stderr column — use
    ``_aggregate_runs_mean_std`` across seeds for a band.
    """
    rows: list[tuple[float, float]] = []
    for row in run.scan_history(keys=[x_key, y_key]):
        xv, yv = row.get(x_key), row.get(y_key)
        if xv is None or yv is None:
            continue
        try:
            xf, yf = float(xv), float(yv)
        except (TypeError, ValueError):
            continue
        if math.isfinite(xf) and math.isfinite(yf):
            rows.append((xf, yf))

    if not rows:
        return np.array([]), np.array([])

    by_x: dict[float, float] = {}
    for xf, yf in rows:
        by_x[xf] = yf
    xs = sorted(by_x.keys())
    ys = np.array([by_x[x] for x in xs], dtype=np.float64)
    return np.array(xs, dtype=np.float64), ys


def _aggregate_runs_mean_std(
    runs: list,
    y_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and sample std of *y_key* vs ``dagger_step`` across runs (same x-grid logic as eval)."""
    curves = []
    for run in runs:
        x, y = extract_scalar_vs_dagger_step(run, y_key)
        if len(x) > 0:
            curves.append((x, y))
    if not curves:
        return np.array([]), np.array([]), np.array([])
    if len(curves) == 1:
        y = curves[0][1]
        return curves[0][0], y, np.zeros_like(y)

    all_same = all(
        len(c[0]) == len(curves[0][0]) and np.allclose(c[0], curves[0][0])
        for c in curves
    )
    if all_same:
        x = curves[0][0]
        ys = np.stack([c[1] for c in curves], axis=0)
        return x, ys.mean(axis=0), ys.std(axis=0, ddof=0)

    x_union = np.sort(np.unique(np.concatenate([c[0] for c in curves])))
    ys_i = []
    for x, y in curves:
        ys_i.append(np.interp(x_union, x, y))
    stack = np.stack(ys_i, axis=0)
    return x_union, stack.mean(axis=0), stack.std(axis=0, ddof=0)


def extract_eval_return_vs_dagger(
    run,
    x_key: str = "eval/dagger_step",
    y_key: str = "eval/return",
    err_key: str = "eval/return_std",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aligned (x, y, stderr) from a run's history.

    Rows must contain all three keys; invalid or missing values are skipped.
    Sorted by x. If duplicate x appear, the last occurrence wins (then sorted).
    """
    rows: list[tuple[float, float, float]] = []
    for row in run.scan_history(keys=[x_key, y_key, err_key]):
        xv, yv, ev = row.get(x_key), row.get(y_key), row.get(err_key)
        if xv is None or yv is None or ev is None:
            continue
        try:
            xf, yf, ef = float(xv), float(yv), float(ev)
        except (TypeError, ValueError):
            continue
        if math.isfinite(xf) and math.isfinite(yf) and math.isfinite(ef):
            rows.append((xf, yf, ef))

    if not rows:
        return np.array([]), np.array([]), np.array([])

    # Deduplicate x: keep last (most recent log for that dagger step)
    by_x: dict[float, tuple[float, float]] = {}
    for xf, yf, ef in rows:
        by_x[xf] = (yf, ef)
    xs = sorted(by_x.keys())
    ys = np.array([by_x[x][0] for x in xs], dtype=np.float64)
    es = np.array([by_x[x][1] for x in xs], dtype=np.float64)
    return np.array(xs, dtype=np.float64), ys, es


def _aggregate_runs_same_x(runs: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average y and stderr across runs when x-grids match; else interpolate."""
    curves = []
    for run in runs:
        x, y, e = extract_eval_return_vs_dagger(run)
        if len(x) > 0:
            curves.append((x, y, e))
    if not curves:
        return np.array([]), np.array([]), np.array([])
    if len(curves) == 1:
        return curves[0]

    all_same = all(
        len(c[0]) == len(curves[0][0]) and np.allclose(c[0], curves[0][0])
        for c in curves
    )
    if all_same:
        x = curves[0][0]
        ys = np.stack([c[1] for c in curves], axis=0)
        es = np.stack([c[2] for c in curves], axis=0)
        return x, ys.mean(axis=0), es.mean(axis=0)

    x_union = np.sort(np.unique(np.concatenate([c[0] for c in curves])))
    ys_i = []
    es_i = []
    for x, y, e in curves:
        ys_i.append(np.interp(x_union, x, y))
        es_i.append(np.interp(x_union, x, e))
    return (
        x_union,
        np.mean(np.stack(ys_i, axis=0), axis=0),
        np.mean(np.stack(es_i, axis=0), axis=0),
    )


def _fetch_baseline_runs(
    env_name: str,
    project: str,
    entity: Optional[str],
    seed: Optional[int],
    baseline_exp_name: str,
) -> list:
    filters: dict = {"exp_name": baseline_exp_name, "env_name": env_name}
    if seed is not None:
        filters["seed"] = seed
    return fetch_runs(project, filters, entity=entity)


def _plot_baseline_eval_from_runs(
    ax: plt.Axes,
    runs: list,
    *,
    legend_label: str,
    show_in_legend: bool,
) -> bool:
    """Standard history DAgger eval return (black)."""
    x, y, err = _aggregate_runs_same_x(runs)
    if len(x) == 0:
        return False
    lab = legend_label if show_in_legend else "_nolegend_"
    ax.plot(x, y, color="black", label=lab, linewidth=2, zorder=10)
    ax.fill_between(x, y - err, y + err, color="black", alpha=0.18, zorder=9)
    return True


def _plot_baseline_query_from_runs(
    ax: plt.Axes,
    runs: list,
    *,
    legend_label: str,
    show_in_legend: bool,
) -> bool:
    """Standard DAgger ``dataset/effective_size`` vs step (typically 100%)."""
    x, y, std = _aggregate_runs_mean_std(runs, "dataset/effective_size")
    if len(x) == 0:
        return False
    lab = legend_label if show_in_legend else "_nolegend_"
    ax.plot(x, y, color="black", label=lab, linewidth=2, zorder=10)
    lo = np.maximum(y - std, 0.0)
    hi = np.minimum(y + std, 100.0)
    ax.fill_between(x, lo, hi, color="black", alpha=0.18, zorder=9)
    return True


def plot_mask_blend_columns(
    env_name: str,
    project: str,
    thresholds: list[float],
    exp_name: str = "history_dagger_disagreement",
    entity: Optional[str] = None,
    seed: Optional[int] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple[float, float] = (12, 4.5),
    baseline_exp_name: str = "history_dagger",
    show_baseline: bool = True,
) -> plt.Figure:
    """
    Two columns (mask | blend) and three rows: eval return; expert query rate on the
    full buffer (%); expert-labeled fraction for the data collected in that DAgger
    iteration only (%). Each colored line = one threshold; band = mean ± stderr
    (eval) or ± std across seeds (query metrics).

    When *show_baseline* is True, overlays standard ``train_history_dagger.py`` runs
    (``exp_name=baseline_exp_name``) as black lines on the first two rows only.

    If *seed* is set, filters ``config.seed`` to that value.
    """
    strategies = ["mask", "blend"]
    n_t = len(thresholds)
    # Light blue → dark blue (match viz/plot_disagreement_timestep_grid.py)
    cmap = plt.get_cmap("Blues")
    colors = [cmap(0.28 + 0.67 * i / max(1, n_t - 1)) for i in range(n_t)]

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(figsize[0], max(figsize[1] * 2.55, 10.0)),
        squeeze=False,
        sharex="col",
        sharey="row",
    )

    fig.suptitle(
        title or (
            f"{env_name}: eval return, buffer query rate, and per-iteration "
            "expert share vs DAgger step"
        ),
        fontsize=13,
    )

    baseline_runs: Optional[list] = None
    if show_baseline:
        baseline_runs = _fetch_baseline_runs(
            env_name, project, entity, seed, baseline_exp_name
        )
        if not baseline_runs:
            print(
                f"  WARNING: no baseline runs for exp_name={baseline_exp_name!r} "
                f"env={env_name!r} (history DAgger reference lines skipped)",
            )

    for col, strategy in enumerate(strategies):
        ax_ret = axes[0, col]
        ax_q = axes[1, col]
        ax_qstep = axes[2, col]
        for i, thresh in enumerate(thresholds):
            filters: dict = {
                "exp_name": exp_name,
                "env_name": env_name,
                "disagreement_threshold": thresh,
                "label_strategy": strategy,
            }
            if seed is not None:
                filters["seed"] = seed

            runs = fetch_runs(project, filters, entity=entity)
            if not runs:
                print(f"  WARNING: no runs for {env_name} strategy={strategy} thresh={thresh}")
                continue

            label = f"thresh={thresh:g}"
            x, y, err = _aggregate_runs_same_x(runs)
            if len(x) == 0:
                print(f"  WARNING: no eval history for {strategy} thresh={thresh}")
            else:
                ax_ret.plot(x, y, color=colors[i], label=label, linewidth=2)
                ax_ret.fill_between(x, y - err, y + err, color=colors[i], alpha=0.22)

            xq, yq, stdq = _aggregate_runs_mean_std(runs, "dataset/expert_query_rate")
            if len(xq) == 0:
                print(f"  WARNING: no dataset/expert_query_rate for {strategy} thresh={thresh}")
            else:
                ax_q.plot(xq, yq, color=colors[i], linewidth=2)
                ax_q.fill_between(xq, yq - stdq, yq + stdq, color=colors[i], alpha=0.22)

            xqs, yqs, stdqs = _aggregate_runs_mean_std(
                runs, "dataset/expert_query_rate_this_step"
            )
            if len(xqs) == 0:
                print(
                    f"  WARNING: no dataset/expert_query_rate_this_step for "
                    f"{strategy} thresh={thresh}",
                )
            else:
                ax_qstep.plot(xqs, yqs, color=colors[i], linewidth=2)
                ax_qstep.fill_between(
                    xqs, yqs - stdqs, yqs + stdqs, color=colors[i], alpha=0.22
                )

        if show_baseline and baseline_runs:
            _plot_baseline_eval_from_runs(
                ax_ret,
                baseline_runs,
                legend_label=baseline_exp_name,
                show_in_legend=(col == 0),
            )
            _plot_baseline_query_from_runs(
                ax_q,
                baseline_runs,
                legend_label=baseline_exp_name,
                show_in_legend=False,
            )

        ax_qstep.set_xlabel("DAgger step")
        if col == 0:
            ax_ret.set_ylabel("Eval return")
            ax_q.set_ylabel("Expert query rate\n(full buffer, %)")
            ax_qstep.set_ylabel("Expert-labeled\n(this iteration, %)")
        ax_ret.set_title(strategy.capitalize())
        ax_ret.grid(True, alpha=0.3)
        ax_q.grid(True, alpha=0.3)
        ax_qstep.grid(True, alpha=0.3)
        ax_ret.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax_q.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax_qstep.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax_ret.tick_params(labelbottom=False)
        ax_q.tick_params(labelbottom=False)

    handles, labels = [], []
    for ax in axes[0]:
        h, lab = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, lab
            break
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(6, len(handles)),
            fontsize="small",
            frameon=True,
        )

    fig.tight_layout(rect=[0, 0.06, 1, 0.93])

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description=(
            "W&B: eval return, buffer query rate, and per-iteration expert share "
            "vs DAgger step (mask | blend, three rows)"
        ),
    )
    parser.add_argument(
        "--env_name",
        type=str,
        required=True,
        help="Single environment name (e.g. darkroom-easy-small)",
    )
    parser.add_argument("--project", type=str, default="history_dagger_v1")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument(
        "--exp_name",
        type=str,
        default="history_dagger_disagreement",
        help="Must match train_history_dagger_disagreement --exp_name",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="If set, only runs with this config.seed",
    )
    parser.add_argument(
        "--baseline_exp_name",
        type=str,
        default="history_dagger",
        help="W&B config.exp_name for vanilla history DAgger (black reference line)",
    )
    parser.add_argument(
        "--no_baseline",
        action="store_true",
        help="Do not overlay the standard history DAgger curve",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Output path (.pdf or .png). Default: plots/disagreement_eval_<env>.pdf",
    )
    args = parser.parse_args()

    save_path = args.save_path or os.path.join(
        "plots",
        f"disagreement_eval_{args.env_name.replace('-', '_')}_mask_blend.pdf",
    )

    plot_mask_blend_columns(
        env_name=args.env_name,
        project=args.project,
        thresholds=sorted(args.thresholds),
        exp_name=args.exp_name,
        entity=args.entity,
        seed=args.seed,
        save_path=save_path,
        baseline_exp_name=args.baseline_exp_name,
        show_baseline=not args.no_baseline,
    )
    plt.close("all")


if __name__ == "__main__":
    main()
