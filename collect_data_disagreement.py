"""
Data collection for disagreement-based DAgger.

- Selective expert querying: expert is queried only when ensemble disagreement > threshold.
- Expert actions are stored for ALL states; loss_mask marks where expert was queried.
"""

import numpy as np
import scipy.special
import torch
import tqdm


@torch.no_grad()
def get_action_and_disagreement(policy, states, device):
    """
    Get ensemble action and per-env disagreement.

    Disagreement is: sum_k KL(p_k || p_mean).

    Returns:
        actions: (B, A) one-hot sampled from p_mean
        disagreement: (B,) float
    """
    if not hasattr(policy, "models"):
        actions = policy.get_action(states)
        disagreement = np.zeros(states.shape[0], dtype=np.float32)
        return actions, disagreement

    for model in policy.models:
        model.eval()

    current_states = torch.from_numpy(states).float().to(device)

    if len(policy.context_states) < 1:
        all_logits = [
            m.get_action(current_states, None, None, None, None)
            for m in policy.models
        ]
    else:
        ctx_states, ctx_actions, ctx_rewards, ctx_dones = policy._get_context_tensors()
        all_logits = [
            m.get_action(current_states, ctx_states, ctx_actions, ctx_rewards, ctx_dones)
            for m in policy.models
        ]

    all_probs = np.stack([
        scipy.special.softmax(logits.cpu().numpy() / policy.temp, axis=1)
        for logits in all_logits
    ])  # (K, B, A)
    mean_probs = all_probs.mean(axis=0)  # (B, A)

    eps = 1e-8
    disagreement = np.sum(
        all_probs * (np.log(all_probs + eps) - np.log(mean_probs[None, :, :] + eps)),
        axis=(0, 2),
    ).astype(np.float32)  # (B,)

    batch_size, num_actions = mean_probs.shape
    action_ids = np.array([
        np.random.choice(num_actions, p=mean_probs[i])
        for i in range(batch_size)
    ])
    actions = np.zeros((batch_size, num_actions), dtype=np.float32)
    actions[np.arange(batch_size), action_ids] = 1.0

    return actions, disagreement


def selective_dagger_rollout(env, rollout_policy, horizon, disagreement_threshold, device):
    """
    Rollout with selective expert querying.

    Expert is queried only where disagreement > threshold.
    Expert actions are stored for ALL timesteps; loss_mask is 1 where expert was queried.
    """
    rollout_policy.set_env(env)
    state = env.reset()
    rollout_policy.reset()
    n_envs = env.num_envs

    states = []
    actions = []
    expert_actions = []
    rewards = []
    dones_list = []
    loss_masks = []
    disagreements = []

    for _ in range(horizon):
        action, disagreement = get_action_and_disagreement(rollout_policy, state, device)
        query_mask = disagreement > disagreement_threshold

        # Always get expert actions for all states (for storage)
        if hasattr(env, "have_keys"):
            expert_action_all = env.opt_action(state, env.have_keys)
        else:
            expert_action_all = env.opt_action(state)

        next_state, reward, done, _ = env.step(action)

        states.append(state)
        actions.append(action)
        expert_actions.append(expert_action_all)
        rewards.append(reward)
        dones_list.append(done)
        loss_masks.append(query_mask.astype(np.float32))
        disagreements.append(disagreement)

        rollout_policy.update_context(state, action, reward, done)

        if np.any(done):
            next_state = env.reset()
        state = next_state

    data = {
        "states": np.stack(states, axis=1),
        "actions": np.stack(actions, axis=1),
        "expert_actions": np.stack(expert_actions, axis=1),
        "rewards": np.stack(rewards, axis=1),
        "dones": np.stack(dones_list, axis=1),
        "loss_mask": np.stack(loss_masks, axis=1),
        "disagreement": np.stack(disagreements, axis=1),
    }

    assert data["states"].shape == (n_envs, horizon, env.state_dim)
    assert data["actions"].shape == (n_envs, horizon, env.action_dim)
    assert data["expert_actions"].shape == (n_envs, horizon, env.action_dim)
    assert data["loss_mask"].shape == (n_envs, horizon)
    assert data["disagreement"].shape == (n_envs, horizon)

    return data


def get_selective_dagger_data(envs, rollout_policy, horizon, disagreement_threshold, device):
    """Collect selective-query DAgger trajectories from a list of envs."""
    from dataset import SequenceDataset

    trajs = []
    query_rates = []
    for env in tqdm.tqdm(envs, desc="Collecting selective DAgger data"):
        data = selective_dagger_rollout(
            env=env,
            rollout_policy=rollout_policy,
            horizon=horizon,
            disagreement_threshold=disagreement_threshold,
            device=device,
        )
        n_envs = env.num_envs

        for k in range(n_envs):
            traj = {
                "states": data["states"][k],
                "actions": data["actions"][k],
                "expert_actions": data["expert_actions"][k],
                "rewards": data["rewards"][k],
                "dones": data["dones"][k],
                "loss_mask": data["loss_mask"][k],
                "disagreement": data["disagreement"][k],
            }
            if hasattr(env, "_envs") and hasattr(env._envs[k], "goal"):
                traj["goal"] = env._envs[k].goal
            query_rates.append(traj["loss_mask"].mean())
            trajs.append(traj)

    mean_query_rate = float(np.mean(query_rates)) if query_rates else 0.0
    print(
        f"Collected {len(trajs)} trajectories, "
        f"mean expert query rate: {mean_query_rate:.2%}"
    )
    return trajs


def get_selective_dagger_dataset(train_envs, test_envs, rollout_policy, horizon, threshold, device):
    """Build train and test SequenceDatasets for selective-query DAgger."""
    from dataset import SequenceDataset

    train_trajs = get_selective_dagger_data(
        train_envs, rollout_policy, horizon, threshold, device
    )
    test_trajs = get_selective_dagger_data(
        test_envs, rollout_policy, horizon, threshold, device
    )

    state_dim = getattr(train_envs[0], "state_dim", train_envs[0]._envs[0].state_dim)
    action_dim = getattr(train_envs[0], "action_dim", train_envs[0]._envs[0].action_dim)

    config = {
        "horizon": horizon,
        "store_gpu": False,
        "state_dim": state_dim,
        "action_dim": action_dim,
    }

    train_dataset = SequenceDataset(train_trajs, {**config, "shuffle": True})
    test_dataset = SequenceDataset(test_trajs, {**config, "shuffle": False})
    return train_dataset, test_dataset
