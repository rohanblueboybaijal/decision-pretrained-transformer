"""
Shared utilities for trajectory visualization scripts.

Contains data-loading helpers, trajectory sampling, and goal-matching
logic used by both darkroom and junction visualizers.
"""

import pickle
import numpy as np


ACTION_NAMES = {0: "right", 1: "left", 2: "down", 3: "up", 4: "stay"}
ACTION_DELTAS = {
    0: (1, 0),   # right: +x
    1: (-1, 0),  # left:  -x
    2: (0, 1),   # down:  +y
    3: (0, -1),  # up:    -y
    4: (0, 0),   # stay
}


def _unstack_merged_dict(merged):
    """Convert a merged trajectory dict (stacked arrays) into a list of per-traj dicts."""
    n = merged["states"].shape[0]
    trajs = []
    has_goals = "goals" in merged
    for i in range(n):
        traj = {
            "states": merged["states"][i],
            "actions": merged["actions"][i],
            "expert_actions": merged["expert_actions"][i],
            "rewards": merged["rewards"][i],
            "dones": merged["dones"][i],
        }
        if has_goals:
            traj["goal"] = merged["goals"][i]
        trajs.append(traj)
    return trajs


def load_sequence_dataset(path, positive_only=False):
    """Load trajectory data from a pickle, handling multiple formats.

    Supported formats:
        - SequenceDataset object (has .trajs attribute)
        - Plain list of trajectory dicts
        - Merged dict of stacked arrays (from save_dagger_data / eval_trajs.pkl)

    Args:
        path: Path to pickle file.
        positive_only: If True, keep only trajectories with total reward > 0.
    """
    with open(path, "rb") as f:
        ds = pickle.load(f)

    if hasattr(ds, "trajs"):
        trajs = ds.trajs
    elif isinstance(ds, list):
        trajs = ds
    elif isinstance(ds, dict) and "states" in ds:
        print(f"  Detected merged-dict format, unstacking ...")
        trajs = _unstack_merged_dict(ds)
    else:
        raise ValueError(f"Unexpected data format in {path}: {type(ds)}")

    if positive_only:
        before = len(trajs)
        trajs = [t for t in trajs if np.sum(t["rewards"]) > 0]
        print(f"  Filtered to positive-return trajectories: {before} -> {len(trajs)}")

    return trajs


def sample_diverse_trajectories(trajs, num_trajs, rng=None):
    """Sample trajectories spanning the full range of returns.

    Divides [min_return, max_return] into ``num_trajs`` equal-width bins
    and randomly picks one trajectory per bin.  Empty bins fall back to
    the trajectory whose return is nearest to the bin centre.

    Returns a list of selected trajectory dicts (length <= num_trajs).
    """
    if len(trajs) <= num_trajs:
        return list(trajs)

    if rng is None:
        rng = np.random.default_rng()

    returns = np.array([float(np.sum(t["rewards"])) for t in trajs])
    r_min, r_max = returns.min(), returns.max()

    if r_min == r_max:
        indices = rng.choice(len(trajs), size=num_trajs, replace=False)
        selected = [trajs[i] for i in sorted(indices)]
        print(f"  Diverse sampling: all returns identical ({r_min:.1f}), "
              f"sampled {num_trajs} randomly")
        return selected

    bin_edges = np.linspace(r_min, r_max, num_trajs + 1)
    selected_indices = []
    for b in range(num_trajs):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == num_trajs - 1:
            mask = (returns >= lo) & (returns <= hi)
        else:
            mask = (returns >= lo) & (returns < hi)
        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            centre = (lo + hi) / 2
            nearest = int(np.argmin(np.abs(returns - centre)))
            selected_indices.append(nearest)
        else:
            selected_indices.append(int(rng.choice(candidates)))

    selected = [trajs[i] for i in selected_indices]
    sel_returns = [float(np.sum(t["rewards"])) for t in selected]
    print(f"  Diverse sampling: bins [{r_min:.1f}, {r_max:.1f}] -> "
          f"selected returns {[f'{r:.1f}' for r in sel_returns]}")
    return selected


def match_by_goal(expert_trajs, learner_trajs, num_trajs):
    """Match expert and learner trajectories by shared goals.

    Goals are rounded before comparison so that float-valued goal
    coordinates are handled safely.

    Returns (expert_sel, learner_sel, matched).  If no goals overlap,
    matched is False and both lists are empty.
    """
    has_expert_goals = all("goal" in t for t in expert_trajs[:1])
    has_learner_goals = all("goal" in t for t in learner_trajs[:1])

    if not (has_expert_goals and has_learner_goals):
        return [], [], False

    goal_to_expert = {}
    for t in expert_trajs:
        key = tuple(np.round(t["goal"]).astype(int))
        goal_to_expert.setdefault(key, []).append(t)

    goal_to_learner = {}
    for t in learner_trajs:
        key = tuple(np.round(t["goal"]).astype(int))
        goal_to_learner.setdefault(key, []).append(t)

    shared = sorted(set(goal_to_expert) & set(goal_to_learner))
    if not shared:
        return [], [], False

    learner_pool = [t for g in shared for t in goal_to_learner[g]]
    learner_sel = sample_diverse_trajectories(learner_pool, num_trajs)

    expert_sel = []
    for lt in learner_sel:
        key = tuple(np.round(lt["goal"]).astype(int))
        expert_sel.append(goal_to_expert[key][0])

    return expert_sel, learner_sel, True
