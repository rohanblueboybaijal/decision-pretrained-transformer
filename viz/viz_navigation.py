"""
Visualize expert and learner trajectories on the 2D Navigation arena.

Loads SequenceDataset pickles (which contain goal info) and plots agent
paths in the continuous [-radius, radius]^2 space, comparing expert vs
learner behavior.  Goals lie on a semi-circle.

Usage:
    # Compare expert (step 0) vs learner (step 1) for a specific run
    python -m viz.viz_navigation \
        --expert_path history_dagger_results/history_dagger-navigation-episodic-seed42/dagger_step_0/train_dataset.pkl \
        --learner_path history_dagger_results/history_dagger-navigation-episodic-seed42/dagger_step_1/train_dataset.pkl \
        --num_trajs 6 --save_path navigation_vis.png

    # Visualize only expert trajectories
    python -m viz.viz_navigation \
        --expert_path history_dagger_results/history_dagger-navigation-episodic-seed42/dagger_step_0/train_dataset.pkl \
        --num_trajs 6
"""

import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import numpy as np

from .viz_common import (
    load_sequence_dataset,
    sample_diverse_trajectories,
    match_by_goal,
)


_COLOR_TO_CMAP = {
    "royalblue": "Blues",
    "blue": "Blues",
    "green": "Greens",
    "red": "Reds",
    "orange": "Oranges",
    "purple": "Purples",
}


def _get_cmap_for_color(color):
    """Return a sequential colormap matching the given base color."""
    if isinstance(color, str) and color in _COLOR_TO_CMAP:
        return plt.get_cmap(_COLOR_TO_CMAP[color])
    base_rgb = mcolors.to_rgb(color)
    colors = [(1.0, 1.0, 1.0), base_rgb, tuple(c * 0.4 for c in base_rgb)]
    return mcolors.LinearSegmentedColormap.from_list("custom", colors, N=256)


def get_radius(env_name):
    """Extract radius from the env name.  Currently always 1.0."""
    return 1.0


GOAL_TOLERANCE = 0.2


def draw_arena(ax, radius):
    """
    Draw the continuous 2D navigation arena.

    Renders a dark background with concentric distance rings,
    crosshairs at the origin, and a dashed semi-circle showing
    the goal region.
    """
    margin = 0.15
    lim = radius + margin
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_facecolor("black")

    for r in np.linspace(0.25, radius, int(radius / 0.25)):
        circle = plt.Circle((0, 0), r, fill=False, color="#333333",
                             linewidth=0.5, linestyle="-", zorder=1)
        ax.add_patch(circle)

    ax.axhline(0, color="#2a2a2a", linewidth=0.5, zorder=1)
    ax.axvline(0, color="#2a2a2a", linewidth=0.5, zorder=1)

    boundary = plt.Rectangle((-radius, -radius), 2 * radius, 2 * radius,
                              fill=False, edgecolor="#555555", linewidth=1.0,
                              linestyle="--", zorder=2)
    ax.add_patch(boundary)

    theta = np.linspace(0, np.pi, 200)
    arc_x = radius * np.cos(theta)
    arc_y = radius * np.sin(theta)
    ax.plot(arc_x, arc_y, color="#776622", linewidth=1.2, linestyle="--",
            alpha=0.6, zorder=2)

    ax.tick_params(labelsize=7, colors="white")
    for spine in ax.spines.values():
        spine.set_color("gray")


def plot_trajectory_on_arena(ax, states, actions, goal, radius, color, label,
                             alpha=0.8, linewidth=1.8, use_gradient=False,
                             show_step_numbers=False):
    """
    Plot a single trajectory on the navigation arena.

    Args:
        ax: matplotlib Axes (arena should already be drawn)
        states: (T, 2) array of (x, y) positions
        actions: (T, action_dim) one-hot actions
        goal: (2,) goal position
        radius: arena radius
        color: line color
        label: legend label
        use_gradient: if True, draw a light-to-dark gradient over time
        show_step_numbers: if True, draw numbered markers at regular intervals
    """
    xs, ys = states[:, 0], states[:, 1]
    T = len(xs)

    if use_gradient and T > 1:
        cmap = _get_cmap_for_color(color)
        norm = plt.Normalize(vmin=0, vmax=T - 1)

        points = np.column_stack([xs, ys]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm,
                            linewidths=linewidth, alpha=alpha, zorder=5,
                            label=label)
        lc.set_array(np.arange(T - 1, dtype=float))
        ax.add_collection(lc)
    else:
        cmap = None
        norm = None
        ax.plot(xs, ys, color=color, alpha=alpha, linewidth=linewidth,
                label=label, zorder=5)

        step = max(1, T // 12)
        for i in range(0, T - 1, step):
            dx = xs[i + 1] - xs[i]
            dy = ys[i + 1] - ys[i]
            if abs(dx) > 0.001 or abs(dy) > 0.001:
                ax.annotate(
                    "", xy=(xs[i + 1], ys[i + 1]), xytext=(xs[i], ys[i]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2,
                                    alpha=alpha * 0.9),
                    zorder=6,
                )

    if show_step_numbers and T > 1:
        if cmap is None:
            cmap = _get_cmap_for_color(color)
            norm = plt.Normalize(vmin=0, vmax=T - 1)
        marker_step = max(1, T // 8)
        for i in range(0, T, marker_step):
            mc = cmap(norm(i))
            ax.plot(xs[i], ys[i], "o", color=mc, markersize=10, zorder=9,
                    markeredgecolor="white", markeredgewidth=0.6)
            ax.text(xs[i], ys[i], str(i), fontsize=5, ha="center",
                    va="center", color="white", fontweight="bold", zorder=10)

    ax.plot(xs[0], ys[0], "s", color=color, markersize=8, zorder=11,
            markeredgecolor="white", markeredgewidth=0.8)

    tol_circle = plt.Circle(goal, GOAL_TOLERANCE, fill=False, color="gold",
                            linewidth=0.7, linestyle=":", alpha=0.4, zorder=11)
    ax.add_patch(tol_circle)

    ax.plot(goal[0], goal[1], "*", color="gold", markersize=14, zorder=12,
            markeredgecolor="white", markeredgewidth=0.8)


def plot_comparison(expert_trajs, learner_trajs, radius, num_trajs,
                    save_path=None, title_suffix="", show_step_numbers=True):
    """
    Plot expert vs learner trajectories side-by-side for shared goals.

    Falls back to learner-only mode when goals don't overlap.
    """
    expert_sel, learner_sel, matched = match_by_goal(
        expert_trajs, learner_trajs, num_trajs)

    if not matched:
        print("WARNING: No shared goals between expert and learner data "
              "(train/eval goal sets are disjoint). "
              "Falling back to learner-only mode.")
        plot_single_set(
            learner_trajs, radius, num_trajs, color="dodgerblue",
            set_label="Learner", save_path=save_path,
            show_step_numbers=show_step_numbers,
        )
        return

    n = len(expert_sel)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5.5 * rows),
                             squeeze=False)
    fig.patch.set_facecolor("#1a1a1a")

    for idx in range(n):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        et = expert_sel[idx]
        lt = learner_sel[idx]
        goal = lt["goal"]

        draw_arena(ax, radius)

        plot_trajectory_on_arena(
            ax, et["states"], et["expert_actions"], goal, radius,
            alpha=0.3, linewidth=5, color="limegreen", label="Expert",
        )
        plot_trajectory_on_arena(
            ax, lt["states"], lt["actions"], goal, radius,
            color="dodgerblue", label="Learner", alpha=0.7, linewidth=1.5,
            use_gradient=True, show_step_numbers=show_step_numbers,
        )

        total_expert_reward = float(np.sum(et["rewards"]))
        total_learner_reward = float(np.sum(lt["rewards"]))
        ax.set_title(
            f"Goal ({goal[0]:.2f}, {goal[1]:.2f})\n"
            f"Expert R={total_expert_reward:.0f}  "
            f"Learner R={total_learner_reward:.0f}",
            fontsize=9, color="white",
        )

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    legend_handles = [
        mpatches.Patch(color="limegreen", label="Expert"),
        mpatches.Patch(color="dodgerblue", label="Learner"),
        plt.Line2D([0], [0], marker="s", color="gray", markersize=8,
                   linestyle="None", label="Start (0, 0)"),
        plt.Line2D([0], [0], marker="*", color="gold", markersize=12,
                   markeredgecolor="white", linestyle="None", label="Goal"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=9, frameon=True, facecolor="#1a1a1a",
               edgecolor="gray", labelcolor="white")

    fig.suptitle(f"Expert vs Learner — Navigation{title_suffix}",
                 fontsize=13, y=1.01, color="white")
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved comparison plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_single_set(trajs, radius, num_trajs, color, set_label,
                    save_path=None, show_step_numbers=True):
    """Plot trajectories from a single dataset (expert or learner)."""
    trajs = sample_diverse_trajectories(trajs, num_trajs)
    n = len(trajs)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    is_learner = set_label.lower() == "learner"

    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5.5 * rows),
                             squeeze=False)
    fig.patch.set_facecolor("#1a1a1a")

    for idx, traj in enumerate(trajs):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        goal = traj["goal"]

        draw_arena(ax, radius)

        plot_trajectory_on_arena(
            ax, traj["states"], traj["actions"], goal, radius,
            color=color, label=set_label,
            use_gradient=is_learner,
            show_step_numbers=is_learner and show_step_numbers,
        )

        total_reward = float(np.sum(traj["rewards"]))
        ax.set_title(
            f"Goal ({goal[0]:.2f}, {goal[1]:.2f})  R={total_reward:.0f}",
            fontsize=9, color="white",
        )

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"{set_label} Trajectories — Navigation", fontsize=13,
                 color="white")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved {set_label.lower()} plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def render_trajectory_video(expert_trajs_list, learner_trajs_list,
                            radius, save_path, fps=4):
    """
    Render a step-by-step video with a subplot grid of trajectory pairs.

    All panels animate in sync: each frame advances every learner
    trajectory by one timestep, with the full expert path shown as a
    faded background.

    Args:
        expert_trajs_list: list of expert traj dicts, or None to skip
            the expert overlay.
        learner_trajs_list: list of learner traj dicts
        radius: arena radius
        save_path: output .mp4 path
        fps: frames per second
    """
    from matplotlib.animation import FuncAnimation

    n = len(learner_trajs_list)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    T = max(len(lt["states"]) for lt in learner_trajs_list)

    cmap = _get_cmap_for_color("royalblue")
    norm = plt.Normalize(vmin=0, vmax=max(T - 2, 1))

    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5.5 * rows),
                             squeeze=False)
    fig.patch.set_facecolor("#1a1a1a")

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    def update(t):
        for idx in range(n):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            ax.cla()

            lt = learner_trajs_list[idx]
            goal = lt["goal"]
            lr_states = lt["states"]
            lr_rewards = lt["rewards"]
            T_i = len(lr_states)
            t_i = min(t, T_i - 1)

            draw_arena(ax, radius)

            if expert_trajs_list is not None:
                ex_states = expert_trajs_list[idx]["states"]
                ax.plot(ex_states[:, 0], ex_states[:, 1], color="limegreen",
                        alpha=0.25, linewidth=4, zorder=3)

            tol_circle = plt.Circle(goal, GOAL_TOLERANCE, fill=False,
                                    color="gold", linewidth=0.7,
                                    linestyle=":", alpha=0.4, zorder=11)
            ax.add_patch(tol_circle)

            ax.plot(goal[0], goal[1], "*", color="gold", markersize=14,
                    zorder=12, markeredgecolor="white", markeredgewidth=0.8)

            if t_i > 0:
                pts = np.column_stack([lr_states[:t_i + 1, 0],
                                       lr_states[:t_i + 1, 1]]).reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                lc = LineCollection(segs, cmap=cmap, norm=norm,
                                    linewidths=2.0, alpha=0.85, zorder=5)
                lc.set_array(np.arange(t_i, dtype=float))
                ax.add_collection(lc)

            ax.plot(lr_states[0, 0], lr_states[0, 1], "s", color="dodgerblue",
                    markersize=8, zorder=11, markeredgecolor="white",
                    markeredgewidth=0.8)

            mc = cmap(norm(min(t_i, T_i - 2)))
            ax.plot(lr_states[t_i, 0], lr_states[t_i, 1], "o", color="white",
                    markersize=18, zorder=13)
            ax.plot(lr_states[t_i, 0], lr_states[t_i, 1], "o", color=mc,
                    markersize=14, zorder=14, markeredgecolor="white",
                    markeredgewidth=2.0)

            cum_reward = float(np.sum(lr_rewards[:t_i]))
            ax.set_title(
                f"Goal ({goal[0]:.2f}, {goal[1]:.2f})\n"
                f"Step {t_i}/{T_i - 1}   R={cum_reward:.0f}",
                fontsize=9, color="white",
            )
        return []

    anim = FuncAnimation(fig, update, frames=T, blit=False, repeat=False)
    fig.tight_layout()
    anim.save(save_path, writer="ffmpeg", fps=fps, dpi=120,
              savefig_kwargs={"facecolor": fig.get_facecolor()})
    plt.close(fig)
    print(f"Saved video to {save_path}  ({n} panels, {T} frames, {fps} fps)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize Navigation expert/learner trajectories"
    )
    parser.add_argument("--expert_path", type=str, default=None,
                        help="Path to expert SequenceDataset pickle (step 0)")
    parser.add_argument("--learner_path", type=str, default=None,
                        help="Path to learner SequenceDataset pickle (step 1+)")
    parser.add_argument("--env_name", type=str, default="navigation-episodic",
                        help="Environment name (determines radius)")
    parser.add_argument("--num_trajs", type=int, default=6)
    parser.add_argument("--positive_only", action="store_true",
                        help="Only show trajectories with positive return")
    parser.add_argument("--no_step_numbers", action="store_true",
                        help="Disable timestep number markers on learner trajectories")
    parser.add_argument("--video", action="store_true",
                        help="Render a step-by-step MP4 video instead of a static plot")
    parser.add_argument("--fps", type=int, default=4,
                        help="Frames per second for video mode (default 4)")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save figure/video (shows interactively if omitted)")

    args = parser.parse_args()
    radius = get_radius(args.env_name)
    show_step_numbers = not args.no_step_numbers

    if args.video:
        save = args.save_path or "navigation_trajectory.mp4"

        if args.expert_path and args.learner_path:
            expert_trajs = load_sequence_dataset(args.expert_path)
            learner_trajs = load_sequence_dataset(args.learner_path,
                                                  positive_only=args.positive_only)

            e_sel, l_sel, matched = match_by_goal(
                expert_trajs, learner_trajs, args.num_trajs)

            if not matched:
                print("WARNING: No shared goals between expert and learner data "
                      "(train/eval goal sets are disjoint). "
                      "Rendering learner-only video.")
                l_sel = sample_diverse_trajectories(learner_trajs, args.num_trajs)
                e_sel = None

            print(f"Animating {len(l_sel)} trajectory panels ...")
            render_trajectory_video(e_sel, l_sel, radius, save, fps=args.fps)

        elif args.learner_path:
            learner_trajs = load_sequence_dataset(args.learner_path,
                                                  positive_only=args.positive_only)
            l_sel = sample_diverse_trajectories(learner_trajs, args.num_trajs)
            print(f"Rendering learner-only video with {len(l_sel)} panels ...")
            render_trajectory_video(None, l_sel, radius, save, fps=args.fps)

        else:
            parser.error("--video requires at least --learner_path")

    elif args.expert_path and args.learner_path:
        print(f"Loading expert data from {args.expert_path}")
        expert_trajs = load_sequence_dataset(args.expert_path)
        print(f"  {len(expert_trajs)} trajectories")

        print(f"Loading learner data from {args.learner_path}")
        learner_trajs = load_sequence_dataset(args.learner_path,
                                              positive_only=args.positive_only)
        print(f"  {len(learner_trajs)} trajectories")

        plot_comparison(
            expert_trajs, learner_trajs, radius, args.num_trajs,
            save_path=args.save_path,
            show_step_numbers=show_step_numbers,
        )

    elif args.expert_path:
        print(f"Loading data from {args.expert_path}")
        trajs = load_sequence_dataset(args.expert_path)
        print(f"  {len(trajs)} trajectories")
        plot_single_set(
            trajs, radius, args.num_trajs, color="limegreen",
            set_label="Expert", save_path=args.save_path,
            show_step_numbers=show_step_numbers,
        )

    elif args.learner_path:
        print(f"Loading data from {args.learner_path}")
        trajs = load_sequence_dataset(args.learner_path,
                                      positive_only=args.positive_only)
        print(f"  {len(trajs)} trajectories")
        plot_single_set(
            trajs, radius, args.num_trajs, color="dodgerblue",
            set_label="Learner", save_path=args.save_path,
            show_step_numbers=show_step_numbers,
        )

    else:
        parser.error(
            "Provide --expert_path and/or --learner_path"
        )
