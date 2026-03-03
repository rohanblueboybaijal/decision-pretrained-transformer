"""
Disagreement-Informed Labeling (DIL) DAgger with ensemble policies.

Algorithm:
1. Roll out an ensemble learner.
2. At each history, compute disagreement:
      sum_k KL(p_k || p_mean)
   where p_k is member-k action distribution and p_mean is the ensemble mean.
3. Query the expert only when disagreement is above a threshold.
4. Train on queried timesteps only (loss_mask).
"""

import torch.multiprocessing as mp

if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method("spawn", force=True)

import argparse
import os
import pickle
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import torch
import torch.nn.functional as F
import tqdm
import wandb

from create_envs import create_env
from collect_data import merge_sequence_datasets
from dataset import SequenceDataset, collate_fn
from ensemble_policy import EnsembleTransformerPolicy
from eval_policy import evaluate_policy_on_envs
from get_rollout_policy import get_rollout_policy
from models import DecisionTransformer


def get_optimizer_scheduler(model, total_steps, lr, warmup_ratio):
    """Create optimizer with warmup + cosine schedule."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    warmup_steps = int(total_steps * warmup_ratio)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_steps]
    )
    return optimizer, scheduler


@torch.no_grad()
def get_action_and_disagreement(policy, states):
    """
    Get ensemble action and per-env disagreement.

    Disagreement is:
        sum_k KL(p_k || p_mean)

    Returns:
        actions: (B, A) one-hot sampled from p_mean
        disagreement: (B,) float
    """
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
    )  # (B,)

    batch_size, num_actions = mean_probs.shape
    action_ids = np.array([
        np.random.choice(num_actions, p=mean_probs[i])
        for i in range(batch_size)
    ])
    actions = np.zeros((batch_size, num_actions))
    actions[np.arange(batch_size), action_ids] = 1.0

    return actions, disagreement


def selective_dagger_rollout(env, rollout_policy, horizon, disagreement_threshold):
    """
    Rollout with selective expert querying.

    Expert is queried only where disagreement > threshold.
    Unqueried timesteps have loss_mask=0 and do not contribute to training.
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
        action, disagreement = get_action_and_disagreement(rollout_policy, state)
        query_mask = disagreement > disagreement_threshold

        queried_expert_actions = np.zeros_like(action)
        if np.any(query_mask):
            if hasattr(env, "have_keys"):
                expert_action_all = env.opt_action(state, env.have_keys)
            else:
                expert_action_all = env.opt_action(state)
            queried_expert_actions[query_mask] = expert_action_all[query_mask]

        next_state, reward, done, _ = env.step(action)

        states.append(state)
        actions.append(action)
        expert_actions.append(queried_expert_actions)
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

    return data


def get_selective_dagger_data(envs, rollout_policy, horizon, disagreement_threshold):
    """Collect selective-query DAgger trajectories from a list of envs."""
    trajs = []
    query_rates = []
    for env in tqdm.tqdm(envs, desc="Collecting selective DAgger data"):
        data = selective_dagger_rollout(
            env=env,
            rollout_policy=rollout_policy,
            horizon=horizon,
            disagreement_threshold=disagreement_threshold,
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
                "goal": env._envs[k].goal,
            }
            query_rates.append(traj["loss_mask"].mean())
            trajs.append(traj)

    mean_query_rate = float(np.mean(query_rates)) if query_rates else 0.0
    print(
        f"Collected {len(trajs)} trajectories, "
        f"mean expert query rate: {mean_query_rate:.2%}"
    )
    return trajs


def get_selective_dagger_dataset(train_envs, test_envs, rollout_policy, horizon, threshold):
    """Build SequenceDataset objects for selective-query DAgger."""
    train_trajs = get_selective_dagger_data(
        train_envs, rollout_policy, horizon, threshold
    )
    test_trajs = get_selective_dagger_data(
        test_envs, rollout_policy, horizon, threshold
    )

    config = {
        "horizon": horizon,
        "store_gpu": False,
        "state_dim": train_envs[0].state_dim,
        "action_dim": train_envs[0].action_dim,
    }

    train_dataset = SequenceDataset(train_trajs, {**config, "shuffle": True})
    test_dataset = SequenceDataset(test_trajs, {**config, "shuffle": False})
    return train_dataset, test_dataset


def data_step(save_dir, step_id, train_envs, test_envs, rollout_policy, horizon, threshold):
    """Collect/load selective-query data for one DAgger step."""
    step_save_dir = os.path.join(save_dir, f"dagger_step_{step_id}")
    os.makedirs(step_save_dir, exist_ok=True)

    train_path = os.path.join(step_save_dir, "train_dataset.pkl")
    test_path = os.path.join(step_save_dir, "test_dataset.pkl")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"Loading existing data from {step_save_dir}")
        with open(train_path, "rb") as f:
            train_dataset = pickle.load(f)
        with open(test_path, "rb") as f:
            test_dataset = pickle.load(f)
    else:
        print(f"Collecting new selective-query data for step {step_id}")
        train_dataset, test_dataset = get_selective_dagger_dataset(
            train_envs=train_envs,
            test_envs=test_envs,
            rollout_policy=rollout_policy,
            horizon=horizon,
            threshold=threshold,
        )
        with open(train_path, "wb") as f:
            pickle.dump(train_dataset, f)
        with open(test_path, "wb") as f:
            pickle.dump(test_dataset, f)

    return train_dataset, test_dataset


def train_step(
    step_id,
    ensemble,
    optimizers,
    schedulers,
    train_loader,
    test_loader,
    save_dir,
    args,
    action_dim,
):
    """Train ensemble on selectively queried labels (loss_mask)."""
    step_save_dir = os.path.join(save_dir, f"dagger_step_{step_id}")
    os.makedirs(step_save_dir, exist_ok=True)

    eval_freq = max(1, int(args.eval_interval * args.num_epochs))
    save_freq = max(1, int(args.save_interval * args.num_epochs))

    for epoch in tqdm.tqdm(range(args.num_epochs), desc=f"Training DAgger Step {step_id}"):
        if epoch % eval_freq == 0 or epoch == args.num_epochs - 1:
            eval_stats = defaultdict(list)
            with torch.no_grad():
                for batch in test_loader:
                    batch_dev = {k: v.to(device) for k, v in batch.items()}
                    true_actions = batch_dev["expert_actions"]
                    mask = batch_dev["loss_mask"]
                    denom = mask.sum().clamp(min=1.0)

                    losses = []
                    for model in ensemble:
                        model.eval()
                        pred, _ = model(batch_dev)
                        action_loss = F.cross_entropy(
                            pred.reshape(-1, action_dim),
                            true_actions.reshape(-1, action_dim),
                            reduction="none",
                        )
                        loss = (action_loss.reshape(mask.shape) * mask).sum() / denom
                        losses.append(loss.item())

                    eval_stats["loss"].append(np.mean(losses))
                    eval_stats["expert_query_rate"].append(mask.mean().item())

                    if "disagreement" in batch_dev:
                        valid = batch_dev["attention_mask"].float()
                        disagreement_mean = (
                            (batch_dev["disagreement"] * valid).sum()
                            / valid.sum().clamp(min=1.0)
                        )
                        eval_stats["disagreement_mean"].append(disagreement_mean.item())

            for k in eval_stats:
                eval_stats[k] = np.mean(eval_stats[k]) if eval_stats[k] else 0.0

            if args.log_wandb:
                for k, v in eval_stats.items():
                    wandb.log({f"dagger-{step_id}/test_{k}": v})

            print(
                f"Epoch {epoch} - Test Loss: {eval_stats.get('loss', 0.0):.4f}, "
                f"Expert Rate: {eval_stats.get('expert_query_rate', 0.0):.2%}, "
                f"Disagreement: {eval_stats.get('disagreement_mean', 0.0):.4f}"
            )

        train_stats = defaultdict(list)
        for batch in train_loader:
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            true_actions = batch_dev["expert_actions"]
            mask = batch_dev["loss_mask"]
            denom = mask.sum()
            if denom <= 0:
                continue

            for k, (model, opt, sched) in enumerate(
                zip(ensemble, optimizers, schedulers)
            ):
                model.train()
                pred, _ = model(batch_dev)
                action_loss = F.cross_entropy(
                    pred.reshape(-1, action_dim),
                    true_actions.reshape(-1, action_dim),
                    reduction="none",
                )
                loss = (action_loss.reshape(mask.shape) * mask).sum() / denom

                opt.zero_grad()
                loss.backward()

                if args.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                opt.step()
                sched.step()
                train_stats[f"loss_model_{k}"].append(loss.item())

                if args.log_wandb and k == 0:
                    wandb.log({
                        f"dagger-{step_id}/lr": opt.param_groups[0]["lr"],
                    })

            avg_loss = np.mean([
                train_stats[f"loss_model_{k}"][-1]
                for k in range(len(ensemble))
            ])
            train_stats["loss"].append(avg_loss)
            train_stats["expert_query_rate"].append(mask.mean().item())

        if args.log_wandb and train_stats["loss"]:
            for k, v in train_stats.items():
                wandb.log({f"dagger-{step_id}/{k}": np.mean(v)})

        if epoch % save_freq == 0:
            for k, model in enumerate(ensemble):
                torch.save(
                    model.state_dict(),
                    os.path.join(step_save_dir, f"model_{k}_epoch_{epoch}.pth"),
                )

    return ensemble


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DIL-DAgger: selective expert querying with ensemble disagreement"
    )

    # Experiment
    parser.add_argument("--exp_name", type=str, default="dil_dagger")
    parser.add_argument("--env_name", type=str, default="darkroom-easy")
    parser.add_argument("--seed", type=int, default=42)

    # Data
    parser.add_argument("--dataset_size", type=int, default=10000)
    parser.add_argument("--dagger_steps", type=int, default=3)
    parser.add_argument("--n_envs", type=int, default=10000)

    # Evaluation
    parser.add_argument("--eval_episodes", type=int, default=40)

    # Model
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Ensemble + querying
    parser.add_argument("--num_ensemble", type=int, default=3)
    parser.add_argument(
        "--disagreement_threshold", type=float, default=0.1,
        help="Query expert when sum_k KL(p_k || p_mean) exceeds this threshold",
    )

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--gradient_clip", action="store_true")
    parser.add_argument("--eval_interval", type=float, default=0.1)
    parser.add_argument("--save_interval", type=float, default=0.1)

    # Logging
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dpt-sweep")
    parser.add_argument("--wandb_entity", type=str, default=None)

    # Paths
    parser.add_argument("--save_dir", type=str, default="./ensemble_results")

    args = parser.parse_args()

    if args.log_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"{args.exp_name}-{args.env_name}-seed{args.seed}",
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    save_dir = os.path.join(
        args.save_dir,
        f"{args.exp_name}-{args.env_name}-seed{args.seed}",
    )
    os.makedirs(save_dir, exist_ok=True)

    print(f"Creating environments: {args.env_name}")
    train_envs, test_envs, eval_envs = create_env(
        args.env_name, args.dataset_size, args.n_envs
    )

    state_dim = train_envs[0]._envs[0].state_dim
    action_dim = train_envs[0]._envs[0].action_dim
    env_horizon = train_envs[0]._envs[0].horizon
    print(f"State dim: {state_dim}, Action dim: {action_dim}, Env horizon: {env_horizon}")

    model_horizon = env_horizon * args.dagger_steps
    model_args = {
        "horizon": model_horizon,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "n_layer": args.num_layers,
        "n_head": args.num_heads,
        "n_embd": 128,
        "dropout": args.dropout,
        "shuffle": True,
        "test": False,
        "continuous_action": False,
        "gmm_heads": 1,
    }

    with open(os.path.join(save_dir, "model_args.pkl"), "wb") as f:
        pickle.dump(model_args, f)

    ensemble = []
    for k in range(args.num_ensemble):
        torch.manual_seed(args.seed + k + 1)
        model = DecisionTransformer(model_args).to(device)
        ensemble.append(model)
    print(
        f"Ensemble: {args.num_ensemble} models, "
        f"{sum(p.numel() for p in ensemble[0].parameters()):,} params each"
    )

    torch.manual_seed(args.seed)

    # Initial step uses full expert data.
    init_rollout_policy = get_rollout_policy("expert")
    train_dataset, test_dataset = data_step(
        save_dir, 0, train_envs, test_envs, init_rollout_policy, env_horizon, -1.0
    )
    for traj in train_dataset.trajs:
        if "loss_mask" not in traj:
            traj["loss_mask"] = np.ones(len(traj["states"]), dtype=np.float32)
    for traj in test_dataset.trajs:
        if "loss_mask" not in traj:
            traj["loss_mask"] = np.ones(len(traj["states"]), dtype=np.float32)

    current_horizon = env_horizon

    total_steps = len(train_dataset) // args.batch_size * args.num_epochs
    optimizers, schedulers = [], []
    for model in ensemble:
        opt, sched = get_optimizer_scheduler(model, total_steps, args.lr, args.warmup_ratio)
        optimizers.append(opt)
        schedulers.append(sched)

    for step_idx in range(args.dagger_steps):
        print(f"\n{'='*60}")
        print(f"DAgger Step {step_idx}/{args.dagger_steps}")
        print(f"Training horizon: {current_horizon}")
        print(f"Dataset size - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        print(f"Disagreement threshold: {args.disagreement_threshold}")
        print(f"{'='*60}")

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        ensemble = train_step(
            step_idx=step_idx,
            ensemble=ensemble,
            optimizers=optimizers,
            schedulers=schedulers,
            train_loader=train_loader,
            test_loader=test_loader,
            save_dir=save_dir,
            args=args,
            action_dim=action_dim,
        )

        print(f"\nEvaluating after DAgger step {step_idx}...")
        eval_policy = EnsembleTransformerPolicy(
            ensemble,
            context_horizon=current_horizon,
            env_horizon=env_horizon,
            sliding_window="nonepisodic" in args.env_name,
        )

        eval_save_dir = os.path.join(save_dir, f"dagger_step_{step_idx}", "eval")
        eval_horizon = args.eval_episodes * env_horizon
        eval_results = evaluate_policy_on_envs(
            eval_envs=eval_envs,
            policy=eval_policy,
            eval_horizon=eval_horizon,
            env_horizon=env_horizon,
            save_dir=eval_save_dir,
            env_name=args.env_name,
            plot=True,
        )

        if args.log_wandb:
            final_mean = eval_results["mean_returns"][-1]
            final_std = eval_results["std_returns"][-1]
            wandb.log({
                f"eval/step_{step_idx}_final_return": final_mean,
                f"eval/step_{step_idx}_final_return_std": final_std,
                f"eval/step_{step_idx}_mean_return": np.mean(eval_results["mean_returns"]),
            })

            fig, ax = plt.subplots(figsize=(10, 6))
            episodes = np.arange(len(eval_results["mean_returns"]))
            ax.plot(episodes, eval_results["mean_returns"], label="Mean Return", linewidth=2)
            ax.fill_between(
                episodes,
                eval_results["mean_returns"] - eval_results["std_returns"],
                eval_results["mean_returns"] + eval_results["std_returns"],
                alpha=0.2,
            )
            ax.set_xlabel("Episode")
            ax.set_ylabel("Return")
            ax.set_title(f"Eval Returns - DAgger Step {step_idx}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            wandb.log({f"eval/step_{step_idx}_returns_plot": wandb.Image(fig)})
            plt.close(fig)

        print(
            f"Evaluation complete - Final return: "
            f"{eval_results['mean_returns'][-1]:.2f} +/- "
            f"{eval_results['std_returns'][-1]:.2f}"
        )

        current_horizon = env_horizon * (step_idx + 2)

        if step_idx < args.dagger_steps - 1:
            step_policy = EnsembleTransformerPolicy(
                ensemble,
                context_horizon=current_horizon,
                env_horizon=env_horizon,
            )

            step_train_dataset, step_test_dataset = data_step(
                save_dir=save_dir,
                step_id=step_idx + 1,
                train_envs=train_envs,
                test_envs=test_envs,
                rollout_policy=step_policy,
                horizon=current_horizon,
                threshold=args.disagreement_threshold,
            )

            train_dataset = merge_sequence_datasets(train_dataset, step_train_dataset)
            test_dataset = merge_sequence_datasets(test_dataset, step_test_dataset)

            total_steps = len(train_dataset) // args.batch_size * args.num_epochs
            optimizers, schedulers = [], []
            for model in ensemble:
                opt, sched = get_optimizer_scheduler(
                    model, total_steps, args.lr, args.warmup_ratio
                )
                optimizers.append(opt)
                schedulers.append(sched)

    for k, model in enumerate(ensemble):
        torch.save(model.state_dict(), os.path.join(save_dir, f"final_model_{k}.pth"))
    print(f"\nTraining complete! Results saved to {save_dir}")

    if args.log_wandb:
        wandb.finish()
