"""
Visualize expert and learner trajectories on the Junction (plus-shaped) maze.

Loads SequenceDataset pickles (which contain goal info) and plots agent
paths on the plus-shaped corridor, comparing expert vs learner behavior.

Usage:
    # Compare expert (step 0) vs learner (step 1) for a specific run
    python viz/viz_junction.py \
        --expert_path history_dagger_results/history_dagger-junction-3-seed42/dagger_step_0/train_dataset.pkl \
        --learner_path history_dagger_results/history_dagger-junction-3-seed42/dagger_step_1/train_dataset.pkl \
        --num_trajs 6 --save_path junction_vis.png

    # Visualize only expert trajectories
    python viz/viz_junction.py \
        --expert_path history_dagger_results/history_dagger-junction-3-seed42/dagger_step_0/train_dataset.pkl \
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

GOAL_LABELS = {
    "left": "Left",
    "right": "Right",
    "up": "Up",
}


def get_corridor_length(env_name):
    """Extract corridor length from env name like 'junction-3'."""
    parts = env_name.split("-")
    return int(parts[1]) if len(parts) > 1 else 3


def build_valid_cells(corridor_length):
    """Build the set of valid (x, y) cells for the plus-shaped maze."""
    L = corridor_length
    cells = set()
    for y in range(L + 1):
        cells.add((L, y))
    for x in range(2 * L + 1):
        cells.add((x, L))
    for y in range(L + 1, 2 * L + 1):
        cells.add((L, y))
    return cells


def goal_direction(goal, corridor_length):
    """Return 'left', 'right', or 'up' for a goal position."""
    L = corridor_length
    gx, gy = int(round(goal[0])), int(round(goal[1]))
    if gx == 0 and gy == L:
        return "left"
    elif gx == 2 * L and gy == L:
        return "right"
    elif gx == L and gy == 2 * L:
        return "up"
    return "unknown"


def draw_maze(ax, corridor_length, cell_size=1.0):
    """
    Draw the plus-shaped maze structure as filled cells with walls.

    The corridor is rendered as light-colored rectangles with a border
    to distinguish walkable cells from the void.
    """
    L = corridor_length
    valid = build_valid_cells(L)
    grid_max = 2 * L

    ax.set_xlim(-0.6, grid_max + 0.6)
    ax.set_ylim(-0.6, grid_max + 0.6)
    ax.set_aspect("equal")

    corridor_color = "#2a2a2a"
    wall_color = "#555555"

    for (cx, cy) in valid:
        rect = plt.Rectangle(
            (cx - 0.5, cy - 0.5), cell_size, cell_size,
            facecolor=corridor_color, edgecolor=wall_color,
            linewidth=0.6, zorder=1,
        )
        ax.add_patch(rect)

    ax.set_facecolor("black")
    ax.set_xticks(range(grid_max + 1))
    ax.set_yticks(range(grid_max + 1))
    ax.tick_params(labelsize=6, colors="white")
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_color("gray")


def draw_landmarks(ax, corridor_length):
    """Draw start, junction, and goal endpoint markers."""
    L = corridor_length
    start = (L, 0)
    junction = (L, L)
    goals = {
        "left": (0, L),
        "right": (2 * L, L),
        "up": (L, 2 * L),
    }

    ax.plot(*junction, "D", color="#666666", markersize=7, zorder=2,
            markeredgecolor="white", markeredgewidth=0.6, alpha=0.5)

    for label, (gx, gy) in goals.items():
        ax.plot(gx, gy, "o", color="#444444", markersize=9, zorder=2,
                markeredgecolor="#888888", markeredgewidth=0.8)
        ax.annotate(label, (gx, gy), fontsize=5, ha="center", va="bottom",
                    xytext=(0, 6), textcoords="offset points", color="#aaaaaa")


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


def plot_trajectory_on_maze(ax, states, actions, goal, corridor_length,
                            color, label, alpha=0.8, linewidth=1.8,
                            use_gradient=False, show_step_numbers=False):
    """
    Plot a single trajectory on the junction maze.

    Args:
        ax: matplotlib Axes (maze should already be drawn)
        states: (T, 2) array of (x, y) positions
        actions: (T, action_dim) one-hot actions
        goal: (2,) goal position
        corridor_length: corridor length L
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
            if abs(dx) > 0.01 or abs(dy) > 0.01:
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
            ax.text(xs[i], ys[i], str(i), fontsize=5, ha="center", va="center",
                    color="white", fontweight="bold", zorder=10)

    ax.plot(xs[0], ys[0], "s", color=color, markersize=8, zorder=11,
            markeredgecolor="white", markeredgewidth=0.8)

    ax.plot(goal[0], goal[1], "*", color="gold", markersize=14, zorder=12,
            markeredgecolor="white", markeredgewidth=0.8)


def plot_comparison(expert_trajs, learner_trajs, corridor_length, num_trajs,
                    save_path=None, title_suffix="",
                    show_step_numbers=True):
    """
    Plot expert vs learner trajectories side-by-side for shared goals.

    If goals don't overlap (e.g. train vs eval split), warns and falls
    back to learner-only mode using the learner's own goals.
    """
    expert_sel, learner_sel, matched = match_by_goal(
        expert_trajs, learner_trajs, num_trajs)

    if not matched:
        print("WARNING: No shared goals between expert and learner data "
              "(train/eval goal sets are disjoint). "
              "Falling back to learner-only mode.")
        plot_single_set(
            learner_trajs, corridor_length, num_trajs, color="dodgerblue",
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

        draw_maze(ax, corridor_length)
        draw_landmarks(ax, corridor_length)

        plot_trajectory_on_maze(
            ax, et["states"], et["expert_actions"], goal, corridor_length,
            alpha=0.3, linewidth=5, color="limegreen", label="Expert",
        )
        plot_trajectory_on_maze(
            ax, lt["states"], lt["actions"], goal, corridor_length,
            color="dodgerblue", label="Learner", alpha=0.7, linewidth=1.5,
            use_gradient=True, show_step_numbers=show_step_numbers,
        )

        direction = goal_direction(goal, corridor_length)
        total_expert_reward = float(np.sum(et["rewards"]))
        total_learner_reward = float(np.sum(lt["rewards"]))
        ax.set_title(
            f"Goal: {direction} ({int(goal[0])}, {int(goal[1])})\n"
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
                   linestyle="None", label="Start"),
        plt.Line2D([0], [0], marker="*", color="gold", markersize=12,
                   markeredgecolor="white", linestyle="None", label="Goal"),
        plt.Line2D([0], [0], marker="D", color="#666666", markersize=6,
                   markeredgecolor="white", linestyle="None", label="Junction"),
    ]
    leg = fig.legend(handles=legend_handles, loc="lower center", ncol=5,
                     fontsize=9, frameon=True, facecolor="#1a1a1a",
                     edgecolor="gray", labelcolor="white")

    fig.suptitle(f"Expert vs Learner — Junction Maze{title_suffix}",
                 fontsize=13, y=1.01, color="white")
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved comparison plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_single_set(trajs, corridor_length, num_trajs, color, set_label,
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

        draw_maze(ax, corridor_length)
        draw_landmarks(ax, corridor_length)

        plot_trajectory_on_maze(
            ax, traj["states"], traj["actions"], goal, corridor_length,
            color=color, label=set_label,
            use_gradient=is_learner,
            show_step_numbers=is_learner and show_step_numbers,
        )

        direction = goal_direction(goal, corridor_length)
        total_reward = float(np.sum(traj["rewards"]))
        ax.set_title(
            f"Goal: {direction} ({int(goal[0])}, {int(goal[1])})  "
            f"R={total_reward:.0f}",
            fontsize=9, color="white",
        )

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"{set_label} Trajectories — Junction Maze", fontsize=13,
                 color="white")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved {set_label.lower()} plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_all_goals_overlay(trajs, corridor_length, max_per_goal=5,
                           save_path=None):
    """
    Single-panel plot with all trajectories colored by goal direction.

    Useful for getting an overview of how the agent distributes across
    the three branches of the junction.
    """
    goal_colors = {"left": "#e63946", "right": "#457b9d", "up": "#2a9d8f"}

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#1a1a1a")
    draw_maze(ax, corridor_length)
    draw_landmarks(ax, corridor_length)

    counts = {"left": 0, "right": 0, "up": 0}
    for traj in trajs:
        direction = goal_direction(traj["goal"], corridor_length)
        if direction == "unknown" or counts.get(direction, 0) >= max_per_goal:
            continue
        counts[direction] += 1
        color = goal_colors.get(direction, "gray")
        xs, ys = traj["states"][:, 0], traj["states"][:, 1]
        ax.plot(xs, ys, color=color, alpha=0.5, linewidth=1.2, zorder=4)
        ax.plot(xs[0], ys[0], "s", color=color, markersize=5, zorder=5,
                markeredgecolor="white", markeredgewidth=0.5)

    legend_handles = [
        mpatches.Patch(color=c, label=f"Goal: {d}")
        for d, c in goal_colors.items()
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper right",
              facecolor="#1a1a1a", edgecolor="gray", labelcolor="white")
    ax.set_title(f"Trajectory Overview — Junction (L={corridor_length})",
                 fontsize=12, color="white")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved overlay plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def render_trajectory_video(expert_trajs_list, learner_trajs_list,
                            corridor_length, save_path, fps=4):
    """
    Render a step-by-step video with a subplot grid of trajectory pairs.

    All panels animate in sync: each frame advances every learner
    trajectory by one timestep, with the full expert path shown as a
    faded background.

    Args:
        expert_trajs_list: list of expert traj dicts, or None to skip
            the expert overlay.
        learner_trajs_list: list of learner traj dicts
        corridor_length: corridor length L
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
            direction = goal_direction(goal, corridor_length)
            lr_states = lt["states"]
            lr_rewards = lt["rewards"]
            T_i = len(lr_states)
            t_i = min(t, T_i - 1)

            draw_maze(ax, corridor_length)
            draw_landmarks(ax, corridor_length)

            if expert_trajs_list is not None:
                ex_states = expert_trajs_list[idx]["states"]
                ax.plot(ex_states[:, 0], ex_states[:, 1], color="limegreen",
                        alpha=0.25, linewidth=4, zorder=3)

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

            cum_reward = float(np.sum(lr_rewards[:t_i + 1]))
            ax.set_title(
                f"Goal: {direction} ({int(goal[0])}, {int(goal[1])})\n"
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
        description="Visualize Junction maze expert/learner trajectories"
    )
    parser.add_argument("--expert_path", type=str, default=None,
                        help="Path to expert SequenceDataset pickle (step 0)")
    parser.add_argument("--learner_path", type=str, default=None,
                        help="Path to learner SequenceDataset pickle (step 1+)")
    parser.add_argument("--env_name", type=str, default="junction-3",
                        help="Environment name (determines corridor length)")
    parser.add_argument("--num_trajs", type=int, default=6)
    parser.add_argument("--positive_only", action="store_true",
                        help="Only show trajectories with positive return")
    parser.add_argument("--overlay", action="store_true",
                        help="Plot all trajectories overlaid on one maze")
    parser.add_argument("--max_per_goal", type=int, default=5,
                        help="Max trajectories per goal direction in overlay mode")
    parser.add_argument("--no_step_numbers", action="store_true",
                        help="Disable timestep number markers on learner trajectories")
    parser.add_argument("--video", action="store_true",
                        help="Render a step-by-step MP4 video instead of a static plot")
    parser.add_argument("--fps", type=int, default=4,
                        help="Frames per second for video mode (default 4)")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save figure/video (shows interactively if omitted)")

    args = parser.parse_args()
    corridor_length = get_corridor_length(args.env_name)
    show_step_numbers = not args.no_step_numbers

    if args.video:
        save = args.save_path or "junction_trajectory.mp4"

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
            render_trajectory_video(e_sel, l_sel, corridor_length,
                                    save, fps=args.fps)

        elif args.learner_path:
            learner_trajs = load_sequence_dataset(args.learner_path,
                                                  positive_only=args.positive_only)
            l_sel = sample_diverse_trajectories(learner_trajs, args.num_trajs)
            print(f"Rendering learner-only video with {len(l_sel)} panels ...")
            render_trajectory_video(None, l_sel, corridor_length,
                                    save, fps=args.fps)

        else:
            parser.error("--video requires at least --learner_path")

    elif args.overlay:
        path = args.expert_path or args.learner_path
        if not path:
            parser.error("Provide --expert_path or --learner_path for overlay mode")
        print(f"Loading data from {path}")
        trajs = load_sequence_dataset(path, positive_only=args.positive_only)
        print(f"  {len(trajs)} trajectories")
        plot_all_goals_overlay(trajs, corridor_length,
                               max_per_goal=args.max_per_goal,
                               save_path=args.save_path)

    elif args.expert_path and args.learner_path:
        print(f"Loading expert data from {args.expert_path}")
        expert_trajs = load_sequence_dataset(args.expert_path)
        print(f"  {len(expert_trajs)} trajectories")

        print(f"Loading learner data from {args.learner_path}")
        learner_trajs = load_sequence_dataset(args.learner_path,
                                              positive_only=args.positive_only)
        print(f"  {len(learner_trajs)} trajectories")

        plot_comparison(
            expert_trajs, learner_trajs, corridor_length, args.num_trajs,
            save_path=args.save_path,
            show_step_numbers=show_step_numbers,
        )

    elif args.expert_path:
        print(f"Loading data from {args.expert_path}")
        trajs = load_sequence_dataset(args.expert_path)
        print(f"  {len(trajs)} trajectories")
        plot_single_set(
            trajs, corridor_length, args.num_trajs, color="limegreen",
            set_label="Expert", save_path=args.save_path,
            show_step_numbers=show_step_numbers,
        )

    elif args.learner_path:
        print(f"Loading data from {args.learner_path}")
        trajs = load_sequence_dataset(args.learner_path,
                                      positive_only=args.positive_only)
        print(f"  {len(trajs)} trajectories")
        plot_single_set(
            trajs, corridor_length, args.num_trajs, color="dodgerblue",
            set_label="Learner", save_path=args.save_path,
            show_step_numbers=show_step_numbers,
        )

    else:
        parser.error(
            "Provide --expert_path and/or --learner_path"
        )
