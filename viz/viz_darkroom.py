"""
Visualize expert and learner trajectories on the Darkroom grid.

Loads SequenceDataset pickles (which contain goal info) and plots agent
paths on the 2D grid, comparing expert vs learner behavior.

Usage:
    # Compare expert (step 0) vs learner (step 1) for a specific run
    python visualize_trajectories.py \
        --expert_path history_dagger_results/history_dagger-darkroom-easy-seed42/dagger_step_0/train_dataset.pkl \
        --learner_path history_dagger_results/history_dagger-darkroom-easy-seed42/dagger_step_1/train_dataset.pkl \
        --num_trajs 6 --save_path trajectory_vis.png

    # Generate fresh expert trajectories on the fly
    python visualize_trajectories.py --env_name darkroom-easy --generate --num_trajs 6
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


def extract_path(states):
    """Extract (x, y) path from states array of shape (T, 2)."""
    return states[:, 0], states[:, 1]


def plot_trajectory_on_grid(ax, states, actions, goal, grid_dim, color, label,
                            alpha=0.8, linewidth=1.5):
    """
    Plot a single trajectory on a grid axis.

    Args:
        ax: matplotlib Axes
        states: (T, 2) array of (x, y) positions
        actions: (T, action_dim) one-hot actions
        goal: (2,) goal position
        grid_dim: size of the grid (e.g. 10 for 10x10)
        color: line color
        label: legend label
        alpha: line transparency
        linewidth: line width
    """
    xs, ys = extract_path(states)

    ax.plot(xs, ys, color=color, alpha=alpha, linewidth=linewidth, label=label,
            zorder=3)

    # Arrow every few steps to show direction
    step = max(1, len(xs) // 15)
    for i in range(0, len(xs) - 1, step):
        dx = xs[i + 1] - xs[i]
        dy = ys[i + 1] - ys[i]
        if abs(dx) > 0.01 or abs(dy) > 0.01:
            ax.annotate(
                "", xy=(xs[i + 1], ys[i + 1]), xytext=(xs[i], ys[i]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2,
                                alpha=alpha * 0.9),
                zorder=4,
            )

    ax.plot(xs[0], ys[0], "s", color=color, markersize=8, zorder=5,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.plot(goal[0], goal[1], "*", color="gold", markersize=14, zorder=6,
            markeredgecolor="white", markeredgewidth=0.8)

    _format_grid(ax, grid_dim)


def _format_grid(ax, grid_dim):
    """Set up grid lines and axis limits."""
    ax.set_facecolor("black")
    ax.set_xlim(-0.5, grid_dim - 0.5)
    ax.set_ylim(-0.5, grid_dim - 0.5)
    ax.set_xticks(range(grid_dim))
    ax.set_yticks(range(grid_dim))
    ax.grid(True, color="gray", alpha=0.3, linewidth=0.5)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7, colors="white")
    for spine in ax.spines.values():
        spine.set_color("gray")


def plot_comparison(expert_trajs, learner_trajs, grid_dim, num_trajs,
                    save_path=None, title_suffix=""):
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
            learner_trajs, grid_dim, num_trajs, color="dodgerblue",
            set_label="Learner", save_path=save_path,
        )
        return

    n = len(expert_sel)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows),
                             squeeze=False)
    fig.patch.set_facecolor("#1a1a1a")

    for idx in range(n):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        et = expert_sel[idx]
        lt = learner_sel[idx]
        goal = lt["goal"]

        plot_trajectory_on_grid(
            ax, et["states"], et["expert_actions"], goal, grid_dim,
            color="limegreen", label="Expert",
        )
        plot_trajectory_on_grid(
            ax, lt["states"], lt["actions"], goal, grid_dim,
            color="dodgerblue", label="Learner",
        )

        total_expert_reward = float(np.sum(et["rewards"]))
        total_learner_reward = float(np.sum(lt["rewards"]))
        ax.set_title(
            f"Goal ({int(goal[0])}, {int(goal[1])})\n"
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
                   linestyle="None", label="Start (0,0)"),
        plt.Line2D([0], [0], marker="*", color="gold", markersize=12,
                   markeredgecolor="black", linestyle="None", label="Goal"),
    ]
    leg = fig.legend(handles=legend_handles, loc="lower center", ncol=4,
                     fontsize=10, frameon=True, facecolor="#1a1a1a",
                     edgecolor="gray", labelcolor="white")

    fig.suptitle(f"Expert vs Learner Trajectories{title_suffix}", fontsize=13,
                 y=1.01, color="white")
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved comparison plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_single_set(trajs, grid_dim, num_trajs, color, set_label,
                    save_path=None):
    """Plot trajectories from a single dataset (expert or learner)."""
    trajs = sample_diverse_trajectories(trajs, num_trajs)
    n = len(trajs)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows),
                             squeeze=False)
    fig.patch.set_facecolor("#1a1a1a")

    for idx, traj in enumerate(trajs):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        goal = traj["goal"]
        plot_trajectory_on_grid(
            ax, traj["states"], traj["actions"], goal, grid_dim,
            color=color, label=set_label,
        )
        total_reward = float(np.sum(traj["rewards"]))
        ax.set_title(
            f"Goal ({int(goal[0])}, {int(goal[1])})  R={total_reward:.0f}",
            fontsize=9, color="white",
        )

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"{set_label} Trajectories", fontsize=13, color="white")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved {set_label.lower()} plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _get_cmap_for_color(base_color, n=256):
    """Create a colormap that fades from white to the given color."""
    rgba = mcolors.to_rgba(base_color)
    colors = np.ones((n, 4))
    for i in range(3):
        colors[:, i] = np.linspace(1.0, rgba[i], n)
    return mcolors.ListedColormap(colors)


def render_trajectory_video(expert_trajs_list, learner_trajs_list,
                            grid_dim, save_path, fps=4):
    """
    Render a step-by-step video with a subplot grid of trajectory pairs.

    All panels animate in sync: each frame advances every learner
    trajectory by one timestep, with the full expert path shown as a
    faded background.  This makes backtracking clearly visible.

    Args:
        expert_trajs_list: list of expert traj dicts, or None to skip
            the expert overlay.
        learner_trajs_list: list of learner traj dicts
        grid_dim: size of the darkroom grid (e.g. 10)
        save_path: output .mp4 path
        fps: frames per second
    """
    from matplotlib.animation import FuncAnimation

    n = len(learner_trajs_list)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    T = max(len(lt["states"]) for lt in learner_trajs_list)

    cmap = _get_cmap_for_color("dodgerblue")
    norm = plt.Normalize(vmin=0, vmax=max(T - 2, 1))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows),
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

            _format_grid(ax, grid_dim)

            # Faded expert path as background reference (only if available)
            if expert_trajs_list is not None:
                ex_states = expert_trajs_list[idx]["states"]
                ax.plot(ex_states[:, 0], ex_states[:, 1], color="limegreen",
                        alpha=0.3, linewidth=4, zorder=3)

            # Goal star
            ax.plot(goal[0], goal[1], "*", color="gold", markersize=14,
                    zorder=12, markeredgecolor="white", markeredgewidth=0.8)

            # Progressive learner path with time-gradient coloring
            if t_i > 0:
                pts = np.column_stack([lr_states[:t_i + 1, 0],
                                       lr_states[:t_i + 1, 1]]).reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                lc = LineCollection(segs, cmap=cmap, norm=norm,
                                    linewidths=2.0, alpha=0.85, zorder=5)
                lc.set_array(np.arange(t_i, dtype=float))
                ax.add_collection(lc)

            # Start marker
            ax.plot(lr_states[0, 0], lr_states[0, 1], "s", color="dodgerblue",
                    markersize=8, zorder=11, markeredgecolor="white",
                    markeredgewidth=0.8)

            # Current position marker (outer white ring + inner colored)
            mc = cmap(norm(min(t_i, T_i - 2)))
            ax.plot(lr_states[t_i, 0], lr_states[t_i, 1], "o", color="white",
                    markersize=18, zorder=13)
            ax.plot(lr_states[t_i, 0], lr_states[t_i, 1], "o", color=mc,
                    markersize=14, zorder=14, markeredgecolor="white",
                    markeredgewidth=2.0)

            cum_reward = float(np.sum(lr_rewards[:t_i]))
            ax.set_title(
                f"Goal ({int(goal[0])}, {int(goal[1])})\n"
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


# def generate_fresh_trajs(env_name, num_trajs):
#     """Generate fresh expert trajectories on the fly."""
#     from create_envs import create_env
#     from collect_data import get_dagger_data
#     from get_rollout_policy import get_rollout_policy

#     n_envs = min(num_trajs, 100)
#     train_envs, _, _ = create_env(env_name, num_trajs, n_envs)
#     env_horizon = train_envs[0]._envs[0].horizon

#     expert_policy = get_rollout_policy("expert")
#     trajs = get_dagger_data(train_envs, expert_policy, env_horizon)
#     return trajs, train_envs[0]._envs[0].dim


def get_grid_dim(env_name):
    """Return grid dimension for a given env name."""
    if "easy-small" in env_name:
        return 5
    elif "easy" in env_name:
        return 10
    elif "hard" in env_name:
        return 20
    return 10


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize Darkroom expert/learner trajectories"
    )
    parser.add_argument("--expert_path", type=str, default=None,
                        help="Path to expert SequenceDataset pickle (step 0)")
    parser.add_argument("--learner_path", type=str, default=None,
                        help="Path to learner SequenceDataset pickle (step 1+)")
    parser.add_argument("--env_name", type=str, default="darkroom-easy")
    # parser.add_argument("--generate", action="store_true",
                        # help="Generate fresh expert trajectories on the fly")
    parser.add_argument("--num_trajs", type=int, default=6)
    parser.add_argument("--positive_only", action="store_true",
                        help="Only show learner trajectories with positive return")
    parser.add_argument("--video", action="store_true",
                        help="Render a step-by-step MP4 video instead of a static plot")
    parser.add_argument("--fps", type=int, default=4,
                        help="Frames per second for video mode (default 4)")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save figure/video (shows interactively if omitted)")

    args = parser.parse_args()

    grid_dim = get_grid_dim(args.env_name)

    # if args.generate:
    #     print(f"Generating fresh expert trajectories for {args.env_name}...")
    #     trajs, grid_dim = generate_fresh_trajs(args.env_name, args.num_trajs)
    #     plot_single_set(
    #         trajs, grid_dim, args.num_trajs, color="green",
    #         set_label="Expert", save_path=args.save_path,
    #     )

    if args.video:
        save = args.save_path or "darkroom_trajectory.mp4"

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

            print(f"Rendering video with {len(l_sel)} trajectory panels ...")
            render_trajectory_video(e_sel, l_sel, grid_dim, save, fps=args.fps)

        elif args.learner_path:
            learner_trajs = load_sequence_dataset(args.learner_path,
                                                  positive_only=args.positive_only)
            l_sel = sample_diverse_trajectories(learner_trajs, args.num_trajs)
            print(f"Rendering learner-only video with {len(l_sel)} panels ...")
            render_trajectory_video(None, l_sel, grid_dim, save, fps=args.fps)

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
            expert_trajs, learner_trajs, grid_dim, args.num_trajs,
            save_path=args.save_path,
        )

    elif args.expert_path:
        print(f"Loading data from {args.expert_path}")
        trajs = load_sequence_dataset(args.expert_path)
        print(f"  {len(trajs)} trajectories")
        plot_single_set(
            trajs, grid_dim, args.num_trajs, color="green",
            set_label="Expert", save_path=args.save_path,
        )

    elif args.learner_path:
        print(f"Loading data from {args.learner_path}")
        trajs = load_sequence_dataset(args.learner_path,
                                      positive_only=args.positive_only)
        print(f"  {len(trajs)} trajectories")
        plot_single_set(
            trajs, grid_dim, args.num_trajs, color="royalblue",
            set_label="Learner", save_path=args.save_path,
        )

    else:
        parser.error(
            "Provide --expert_path and/or --learner_path, or use --generate"
        )
