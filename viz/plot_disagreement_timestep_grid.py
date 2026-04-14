"""
Mean ensemble disagreement vs env timestep, by DAgger iteration — grid plot.

**Data source: local ``metrics.json``** (written by ``train_history_dagger_disagreement.py``).
Each run’s file lists ``dagger_steps[*].disagreement_per_timestep`` with
``mean_per_timestep`` and ``stderr_per_timestep``. W&B stores images/tables per step;
reconstructing this grid from the API is fragile, so this script scans your results
directory and matches runs by config.

Layout: **2 columns** (mask | blend), **one row per threshold**. In each cell, one
curve per DAgger iteration (step 0 = BC placeholder unless you hide it).

Usage (from repo root)::

    python -m viz.plot_disagreement_timestep_grid \\
        --results_dir ./history_dagger_disagreement_results \\
        --env_name darkroom-easy-small \\
        --seed 1 \\
        --save_path plots/disagreement_timestep_grid.pdf
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _thresh_close(a: float, b: float, rtol: float = 1e-9, atol: float = 1e-12) -> bool:
    af, bf = float(a), float(b)
    return abs(af - bf) <= atol + rtol * max(abs(af), abs(bf), 1e-15)


def _metrics_for_cell(
    grouped: dict[tuple[float, str], dict[str, Any]],
    thresh: float,
    strat: str,
) -> Optional[dict[str, Any]]:
    for (t, s), data in grouped.items():
        if s != strat:
            continue
        if _thresh_close(t, thresh):
            return data
    return None


def load_metrics_paths(
    results_dir: str,
    env_name: str,
    seed: int,
    exp_name: str = "history_dagger_disagreement",
) -> list[tuple[str, dict[str, Any]]]:
    """Return ``(path, metrics_json)`` for runs whose config matches filters."""
    out: list[tuple[str, dict[str, Any]]] = []
    results_dir = os.path.abspath(results_dir)
    if not os.path.isdir(results_dir):
        return out

    for name in os.listdir(results_dir):
        run_dir = os.path.join(results_dir, name)
        if not os.path.isdir(run_dir):
            continue
        path = os.path.join(run_dir, "metrics.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        cfg = data.get("config") or {}
        if cfg.get("env_name") != env_name:
            continue
        if int(cfg.get("seed", -1)) != int(seed):
            continue
        if cfg.get("exp_name", exp_name) != exp_name:
            continue
        out.append((path, data))

    return out


def group_by_thresh_strategy(
    loaded: list[tuple[str, dict[str, Any]]],
) -> dict[tuple[float, str], dict[str, Any]]:
    """Key: (disagreement_threshold, label_strategy) -> metrics dict (first match wins)."""
    grouped: dict[tuple[float, str], dict[str, Any]] = {}
    for _path, data in loaded:
        cfg = data.get("config") or {}
        thresh = float(cfg["disagreement_threshold"])
        strat = str(cfg.get("label_strategy", "mask"))
        key = (thresh, strat)
        if key not in grouped:
            grouped[key] = data
    return grouped


def plot_disagreement_timestep_grid(
    results_dir: str,
    env_name: str,
    seed: int = 1,
    exp_name: str = "history_dagger_disagreement",
    thresholds: Optional[list[float]] = None,
    strategies: tuple[str, str] = ("mask", "blend"),
    show_stderr: bool = True,
    hide_bc_step: bool = False,
    save_path: Optional[str] = None,
    figsize_per_cell: tuple[float, float] = (5.0, 3.2),
) -> plt.Figure:
    """
    For each threshold (row) and strategy (column), plot mean disagreement vs timestep;
    one line per DAgger iteration index present in ``dagger_steps``.
    """
    loaded = load_metrics_paths(results_dir, env_name, seed, exp_name=exp_name)
    grouped = group_by_thresh_strategy(loaded)

    if not grouped:
        raise FileNotFoundError(
            f"No metrics.json under {results_dir} matching "
            f"env_name={env_name!r}, seed={seed}, exp_name={exp_name!r}",
        )

    if thresholds is None:
        thresh_set = {t for (t, _s) in grouped.keys()}
        thresholds = sorted(thresh_set)

    nrows = len(thresholds)
    ncols = len(strategies)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows),
        squeeze=False,
        sharex="col",
    )

    # Light blue → dark blue (skip near-white tail of "Blues")
    cmap = plt.get_cmap("Blues")

    for r, thresh in enumerate(thresholds):
        for c, strat in enumerate(strategies):
            ax = axes[r][c]
            data = _metrics_for_cell(grouped, thresh, strat)

            if data is None:
                ax.text(
                    0.5,
                    0.5,
                    "no run",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=11,
                    color="gray",
                )
                ax.set_title(f"thresh={thresh:g} — {strat} (missing)")
                ax.grid(True, alpha=0.2)
                continue

            steps_payload = data.get("dagger_steps") or []
            n_curves = len(steps_payload)
            if n_curves == 0:
                ax.text(0.5, 0.5, "no dagger_steps", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"thresh={thresh:g} — {strat}")
                continue

            for i, snap in enumerate(steps_payload):
                ds = int(snap.get("dagger_step", i))
                if hide_bc_step and ds == 0:
                    continue
                disc = snap.get("disagreement_per_timestep")
                if not disc or "mean_per_timestep" not in disc:
                    continue
                mean = np.asarray(disc["mean_per_timestep"], dtype=np.float64)
                stderr = None
                if show_stderr and "stderr_per_timestep" in disc:
                    stderr = np.asarray(disc["stderr_per_timestep"], dtype=np.float64)
                t_axis = np.arange(len(mean))
                color = cmap(0.28 + 0.67 * (i / max(1, n_curves - 1)))
                label = f"iter {ds}"
                ax.plot(t_axis, mean, color=color, label=label, linewidth=1.6)
                if stderr is not None and len(stderr) == len(mean):
                    ax.fill_between(
                        t_axis,
                        mean - stderr,
                        mean + stderr,
                        color=color,
                        alpha=0.18,
                    )

            # Expert-query threshold for this row (disagreement > thresh → query)
            ax.axhline(
                thresh,
                color="0.35",
                linestyle=":",
                linewidth=0.9,
                alpha=0.85,
                zorder=1,
            )

            ax.set_ylabel("mean disagreement")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"thresh={thresh:g} — {strat}")
            if r == nrows - 1:
                ax.set_xlabel("env timestep")

    # Legend: use last axis that has lines (np.flatiter is not reversible)
    for ax in np.ravel(axes)[::-1]:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=min(12, len(handles)),
                fontsize="x-small",
                frameon=True,
            )
            break

    fig.suptitle(
        f"{env_name} (seed={seed}): mean disagreement vs timestep by DAgger iter",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")

    return fig


def main():
    p = argparse.ArgumentParser(
        description="Grid: mean disagreement vs timestep (mask | blend × thresholds), from metrics.json",
    )
    p.add_argument(
        "--results_dir",
        type=str,
        default="./history_dagger_disagreement_results",
        help="Directory containing per-run subfolders with metrics.json",
    )
    p.add_argument("--env_name", type=str, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--exp_name", type=str, default="history_dagger_disagreement")
    p.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=None,
        help="Row order; default: all thresholds found for this env/seed",
    )
    p.add_argument(
        "--no_stderr",
        action="store_true",
        help="Do not shade mean ± stderr",
    )
    p.add_argument(
        "--hide_bc_step",
        action="store_true",
        help="Omit DAgger iteration 0 (BC placeholder curve)",
    )
    p.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Output .pdf / .png (default: plots/disagreement_timestep_<env>.pdf)",
    )
    args = p.parse_args()

    save_path = args.save_path or os.path.join(
        "plots",
        f"disagreement_timestep_{args.env_name.replace('-', '_')}.pdf",
    )

    plot_disagreement_timestep_grid(
        results_dir=args.results_dir,
        env_name=args.env_name,
        seed=args.seed,
        exp_name=args.exp_name,
        thresholds=args.thresholds,
        show_stderr=not args.no_stderr,
        hide_bc_step=args.hide_bc_step,
        save_path=save_path,
    )
    plt.close("all")


if __name__ == "__main__":
    main()
