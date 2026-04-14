"""
History-Based DAgger with Disagreement-Based Expert Querying.

Extends train_history_dagger.py with:
- Ensemble of Decision Transformers (context length = env_horizon).
- Expert queried only when ensemble disagreement > threshold.
- Expert actions stored for all states; loss_mask marks where expert was queried.
- Loss: masked by loss_mask (Strategy 1 — run on all data, mask in loss).
- Step 0 (BC): expert-only data; per-timestep disagreement is the constant
  ``default_disagreement`` (no ensemble at collection). Logged as data/step_kind=bc and
  mirrored in metrics.json for wandb/API alignment.
- metrics.json: full training log + per-dagger-step snapshots with disagreement arrays.
  Each snapshot includes ``expert_query_rate`` (full merged buffer) and
  ``expert_query_rate_this_step`` (only ``dagger_step_k/train_dataset.pkl``).

Checkpoints: ``--no-save_model`` skips per-epoch and final ensemble ``.pth`` saves (same flag as ``train_history_dagger.py``).
"""

import torch.multiprocessing as mp

if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method("spawn", force=True)

import argparse
import json
import os
from typing import Optional
import pickle
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from create_envs import create_env
from collect_data import get_dagger_dataset, merge_sequence_datasets
from collect_data_disagreement import get_selective_dagger_dataset
from dataset import SequenceDataset, collate_fn
from eval_policy import evaluate_policy_on_envs
from get_rollout_policy import get_rollout_policy
from ensemble_policy import EnsembleTransformerPolicy
from models import DecisionTransformer
from viz.viz_common import sample_diverse_trajectories, plot_eval_returns_histogram

METRICS_JSON_NAME = "metrics.json"


def _persist_run_metrics(save_dir, run_metrics):
    path = os.path.join(save_dir, METRICS_JSON_NAME)
    with open(path, "w") as f:
        json.dump(run_metrics, f, indent=2, default=str)


def init_run_metrics(save_dir, args):
    """Initialize on-disk metrics mirror (wandb + richer fields like per-timestep disagreement)."""
    run_metrics = {
        "created_at": datetime.now().isoformat(),
        "save_dir": os.path.abspath(save_dir),
        "config": vars(args).copy(),
        "epochs": [],
        "dagger_steps": [],
    }
    _persist_run_metrics(save_dir, run_metrics)
    return run_metrics


def expert_query_rate_for_step_dataset(save_dir: str, step_idx: int) -> Optional[float]:
    """
    Expert query rate (% of tokens with loss_mask) using only
    ``dagger_step_{step_idx}/train_dataset.pkl`` — data for this DAgger iteration,
    not the merged replay buffer.

    Returns None if that pickle is missing.
    """
    train_path = os.path.join(save_dir, f"dagger_step_{step_idx}", "train_dataset.pkl")
    if not os.path.isfile(train_path):
        return None
    with open(train_path, "rb") as f:
        step_dataset = pickle.load(f)
    trajs = step_dataset.trajs
    if not trajs:
        return 0.0
    expert_queried_tokens = sum(int(t["loss_mask"].sum()) for t in trajs)
    total_tokens = sum(t["states"].shape[0] for t in trajs)
    if total_tokens <= 0:
        return 0.0
    return expert_queried_tokens / total_tokens * 100.0


def compute_disagreement_per_timestep_stats(step_trajs, horizon):
    """
    Mean and standard error of ensemble disagreement at each env timestep across trajectories.

    Returns None if trajectories lack 'disagreement' (should not happen in normal runs).
    """
    if not step_trajs or "disagreement" not in step_trajs[0]:
        return None
    stacked = np.stack([t["disagreement"] for t in step_trajs])
    mean_disagreement = stacked.mean(axis=0)
    n = stacked.shape[0]
    std_error = stacked.std(axis=0) / np.sqrt(n) if n > 1 else np.zeros_like(mean_disagreement)
    return {
        "mean_per_timestep": mean_disagreement.astype(np.float64).tolist(),
        "stderr_per_timestep": std_error.astype(np.float64).tolist(),
        "n_trajectories": int(n),
        "horizon": int(horizon),
    }


def generate_eval_video(eval_trajs, env_name, save_path, num_trajs=9):
    """Render an MP4 video of sampled eval trajectories."""
    learner_trajs = sample_diverse_trajectories(eval_trajs, num_trajs)

    if "darkroom" in env_name:
        from viz.viz_darkroom import render_trajectory_video, get_grid_dim
        grid_dim = get_grid_dim(env_name)
        render_trajectory_video(None, learner_trajs, grid_dim, save_path)
    elif "junction" in env_name:
        from viz.viz_junction import render_trajectory_video, get_corridor_length
        corridor_length = get_corridor_length(env_name)
        render_trajectory_video(None, learner_trajs, corridor_length, save_path)
    elif "navigation" in env_name:
        from viz.viz_navigation import render_trajectory_video, get_radius
        radius = get_radius(env_name)
        render_trajectory_video(None, learner_trajs, radius, save_path)
    else:
        print(f"Video visualization not supported for {env_name}, skipping.")
        return None

    return save_path


def get_optimizer_scheduler(model, total_steps, lr, warmup_ratio):
    """Create optimizer with warmup + cosine decay schedule."""
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


def data_step_expert(save_dir, step_id, train_envs, test_envs, horizon, default_disagreement):
    """
    Step 0: collect expert-only data and add loss_mask (all 1) and disagreement (default).
    """
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
        print(f"Collecting expert data for step {step_id}")
        init_rollout_policy = get_rollout_policy("expert")
        train_dataset, test_dataset = get_dagger_dataset(
            train_envs, test_envs, init_rollout_policy, horizon
        )
        for traj in train_dataset.trajs:
            traj["loss_mask"] = np.ones(horizon, dtype=np.float32)
            traj["disagreement"] = np.full(horizon, default_disagreement, dtype=np.float32)
        for traj in test_dataset.trajs:
            traj["loss_mask"] = np.ones(horizon, dtype=np.float32)
            traj["disagreement"] = np.full(horizon, default_disagreement, dtype=np.float32)
        with open(train_path, "wb") as f:
            pickle.dump(train_dataset, f)
        with open(test_path, "wb") as f:
            pickle.dump(test_dataset, f)

    return train_dataset, test_dataset


def data_step_ensemble(save_dir, step_id, train_envs, test_envs, rollout_policy, horizon, threshold, device):
    """Step k>0: collect data with ensemble rollout and selective expert querying."""
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
        print(f"Collecting selective DAgger data for step {step_id}")
        train_dataset, test_dataset = get_selective_dagger_dataset(
            train_envs, test_envs, rollout_policy, horizon, threshold, device
        )
        with open(train_path, "wb") as f:
            pickle.dump(train_dataset, f)
        with open(test_path, "wb") as f:
            pickle.dump(test_dataset, f)

    return train_dataset, test_dataset


def plot_disagreement_by_timestep(step_trajs, horizon, step_idx):
    """
    Compute mean and std error of disagreement per env timestep; return a matplotlib figure.
    step_trajs: list of trajectory dicts with 'disagreement' (horizon,).
    """
    stats = compute_disagreement_per_timestep_stats(step_trajs, horizon)
    if stats is None:
        return None
    mean_disagreement = np.array(stats["mean_per_timestep"], dtype=np.float64)
    std_error = np.array(stats["stderr_per_timestep"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(horizon), mean_disagreement)
    ax.fill_between(
        np.arange(horizon),
        mean_disagreement - std_error,
        mean_disagreement + std_error,
        alpha=0.3,
    )
    ax.set_xlabel("Env timestep")
    ax.set_ylabel("Mean disagreement")
    ax.set_title(f"Disagreement by timestep (DAgger step {step_idx})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def train_step(
    step_id,
    ensemble,
    optimizers,
    schedulers,
    train_loader,
    test_loader,
    save_dir,
    args,
    device,
    action_dim,
    global_epoch_offset,
    run_metrics,
):
    """
    Train ensemble for one DAgger step. Loss is masked by loss_mask (Strategy 1).
    Model runs on all data; only positions where loss_mask=1 contribute to the loss.
    """
    step_save_dir = os.path.join(save_dir, f"dagger_step_{step_id}")
    os.makedirs(step_save_dir, exist_ok=True)

    eval_freq = max(1, int(args.eval_interval * args.num_epochs))
    save_freq = max(1, int(args.save_interval * args.num_epochs))

    def forward_batch(batch, model):
        batch = {k: v.to(device) for k, v in batch.items()}
        pred_actions, _ = model(batch)

        if args.label_strategy == "blend":
            loss_mask_expanded = batch["loss_mask"].unsqueeze(-1)
            true_actions = (
                batch["expert_actions"] * loss_mask_expanded
                + batch["actions"] * (1.0 - loss_mask_expanded)
            )
            mask = batch["attention_mask"].float()
        else:
            true_actions = batch["expert_actions"]
            mask = batch["attention_mask"].float() * batch["loss_mask"]

        per_token_loss = F.cross_entropy(
            pred_actions.reshape(-1, action_dim),
            true_actions.reshape(-1, action_dim),
            reduction="none",
        )
        per_token_loss = per_token_loss.reshape(mask.shape)
        denom = mask.sum().clamp(min=1e-8)
        loss = (per_token_loss * mask).sum() / denom
        return loss, batch

    for epoch in tqdm.tqdm(range(args.num_epochs), desc=f"Training DAgger Step {step_id}"):
        global_epoch = global_epoch_offset + epoch

        # --- Evaluation ---
        epoch_test_loss = None
        if epoch % eval_freq == 0 or epoch == args.num_epochs - 1:
            test_losses = []
            for model in ensemble:
                model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    loss, _ = forward_batch(batch, ensemble[0])
                    test_losses.append(loss.item())
            epoch_test_loss = np.mean(test_losses)
            print(f"Epoch {epoch} - Test Loss: {epoch_test_loss:.4f}")

        # --- Training ---
        batch_losses = []
        batch_grad_norms = []

        for batch in train_loader:
            for k, (model, optimizer, scheduler) in enumerate(zip(ensemble, optimizers, schedulers)):
                model.train()
                loss, batch_dev = forward_batch(batch, model)

                optimizer.zero_grad()
                loss.backward()

                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                if k == 0:
                    batch_grad_norms.append(total_norm ** 0.5)

                if args.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                batch_losses.append(loss.item())

        # --- Logging ---
        log_dict = {
            "global_epoch": global_epoch,
            "dagger_step": step_id,
            "train/loss": float(np.mean(batch_losses)),
            "train/grad_norm": float(np.mean(batch_grad_norms)) if batch_grad_norms else 0.0,
            "train/lr": float(optimizers[0].param_groups[0]["lr"]),
        }
        if epoch_test_loss is not None:
            log_dict["test/loss"] = float(epoch_test_loss)
        run_metrics["epochs"].append(log_dict.copy())
        _persist_run_metrics(save_dir, run_metrics)

        if args.log_wandb:
            wandb.log(log_dict)

        if args.save_model and epoch % save_freq == 0:
            for k, model in enumerate(ensemble):
                torch.save(
                    model.state_dict(),
                    os.path.join(step_save_dir, f"model_{k}_epoch_{epoch}.pth"),
                )

    return ensemble


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="History-Based DAgger with Disagreement")

    parser.add_argument("--exp_name", type=str, default="history_dagger_disagreement")
    parser.add_argument("--env_name", type=str, default="darkroom-easy")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dataset_size", type=int, default=5000)
    parser.add_argument("--dagger_steps", type=int, default=3)
    parser.add_argument("--n_envs", type=int, default=5000)
    parser.add_argument(
        "--eval_ood",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set (default), test/eval goals are disjoint from train. "
             "Use --no-eval_ood for in-distribution test/eval.",
    )

    parser.add_argument("--num_ensemble", type=int, default=3)
    parser.add_argument(
        "--disagreement_threshold",
        type=float,
        default=0.1,
        help="Query expert when ensemble disagreement exceeds this",
    )
    parser.add_argument(
        "--default_disagreement",
        type=float,
        default=0.0,
        help="Disagreement value stored for step 0 (no ensemble yet)",
    )
    parser.add_argument(
        "--label_strategy",
        type=str,
        default="mask",
        choices=["mask", "blend"],
        help="'mask': loss only on expert-queried positions. "
             "'blend': expert labels where queried, learner labels elsewhere, loss on all.",
    )

    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--gradient_clip", action="store_true")
    parser.add_argument("--eval_interval", type=float, default=0.1)
    parser.add_argument("--save_interval", type=float, default=1)
    parser.add_argument(
        "--save_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-epoch and final checkpoints (disable with --no-save_model to save disk when "
        "using large ensembles).",
    )

    parser.add_argument(
        "--num_eval_trajs",
        type=int,
        default=9,
        help="Number of eval trajectories for video and for returns histogram bins",
    )

    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dpt-sweep")
    parser.add_argument("--wandb_entity", type=str, default=None)

    parser.add_argument("--save_dir", type=str, default="./history_dagger_disagreement_results")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    if args.log_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"{args.exp_name}-{args.env_name}-thresh{args.disagreement_threshold}-{args.label_strategy}-seed{args.seed}",
        )
        wandb.define_metric("global_epoch")
        wandb.define_metric("*", step_metric="global_epoch")
        wandb.define_metric("disagreement_by_timestep", step_metric="dagger_step")
        wandb.define_metric("data/disagreement_per_timestep_table", step_metric="dagger_step")
        wandb.define_metric("eval_returns_histogram", step_metric="dagger_step")

    # ------------------------------------------------------------------
    # Seed
    # ------------------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(
        args.save_dir,
        f"{args.exp_name}-{args.env_name}-thresh{args.disagreement_threshold}-seed{args.seed}-{timestamp}",
    )
    os.makedirs(save_dir, exist_ok=True)
    run_metrics = init_run_metrics(save_dir, args)

    # ------------------------------------------------------------------
    # Environments
    # ------------------------------------------------------------------
    print(f"Creating environments: {args.env_name} (eval_ood={args.eval_ood})")
    train_envs, test_envs, eval_envs = create_env(
        args.env_name, args.dataset_size, args.n_envs, eval_ood=args.eval_ood
    )

    state_dim = train_envs[0]._envs[0].state_dim
    action_dim = train_envs[0]._envs[0].action_dim
    env_horizon = train_envs[0]._envs[0].horizon
    print(f"State dim: {state_dim}, Action dim: {action_dim}, Env horizon: {env_horizon}")

    # ------------------------------------------------------------------
    # Ensemble — context length = env_horizon (fixed)
    # ------------------------------------------------------------------
    model_horizon = env_horizon
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

    # ------------------------------------------------------------------
    # Step 0: expert data with loss_mask=1 and default disagreement
    # ------------------------------------------------------------------
    train_dataset, test_dataset = data_step_expert(
        save_dir, 0, train_envs, test_envs, env_horizon, args.default_disagreement
    )

    total_steps = len(train_dataset) // args.batch_size * args.num_epochs
    optimizers = []
    schedulers = []
    for model in ensemble:
        opt, sched = get_optimizer_scheduler(model, total_steps, args.lr, args.warmup_ratio)
        optimizers.append(opt)
        schedulers.append(sched)

    global_epoch_offset = 0

    # ------------------------------------------------------------------
    # DAgger loop
    # ------------------------------------------------------------------
    for step_idx in range(args.dagger_steps):
        expert_queried_tokens = sum(int(traj["loss_mask"].sum()) for traj in train_dataset.trajs)
        total_tokens = sum(traj["states"].shape[0] for traj in train_dataset.trajs)
        expert_query_rate = (expert_queried_tokens / total_tokens * 100.0) if total_tokens > 0 else 100.0
        expert_query_rate_this_step = expert_query_rate_for_step_dataset(save_dir, step_idx)
        if args.label_strategy == "blend":
            effective_size_pct = 100.0
        else:
            effective_size_pct = expert_query_rate

        print(f"\n{'='*60}")
        print(f"DAgger Step {step_idx}/{args.dagger_steps}")
        print(f"Dataset size - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        print(f"Expert query rate (full buffer): {expert_query_rate:.1f}%")
        if expert_query_rate_this_step is not None:
            print(f"Expert query rate (this DAgger iteration only): {expert_query_rate_this_step:.1f}%")
        print(f"Effective dataset size (tokens contributing to loss): {effective_size_pct:.1f}%")
        print(f"Disagreement threshold: {args.disagreement_threshold}")
        print(f"Label strategy: {args.label_strategy}")
        print(f"{'='*60}")

        step_kind = "bc" if step_idx == 0 else "dagger"
        dataset_log = {
            "global_epoch": global_epoch_offset,
            "dagger_step": step_idx,
            "data/step_kind": step_kind,
            "data/is_bc_step": step_idx == 0,
            "dataset/train_trajectories": len(train_dataset),
            "dataset/test_trajectories": len(test_dataset),
            "dataset/expert_query_rate": float(expert_query_rate),
            "dataset/effective_size": float(effective_size_pct),
        }
        if expert_query_rate_this_step is not None:
            dataset_log["dataset/expert_query_rate_this_step"] = float(expert_query_rate_this_step)
        if args.log_wandb:
            wandb.log(dataset_log)

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
            step_id=step_idx,
            ensemble=ensemble,
            optimizers=optimizers,
            schedulers=schedulers,
            train_loader=train_loader,
            test_loader=test_loader,
            save_dir=save_dir,
            args=args,
            device=device,
            action_dim=action_dim,
            global_epoch_offset=global_epoch_offset,
            run_metrics=run_metrics,
        )

        eval_global_epoch = global_epoch_offset + args.num_epochs - 1
        global_epoch_offset += args.num_epochs

        # --- Disagreement-by-timestep plot (use this step's data only)
        step_save_dir = os.path.join(save_dir, f"dagger_step_{step_idx}")
        train_path = os.path.join(step_save_dir, "train_dataset.pkl")
        if os.path.exists(train_path):
            with open(train_path, "rb") as f:
                step_dataset = pickle.load(f)
            step_trajs = step_dataset.trajs
        else:
            step_trajs = train_dataset.trajs
        disc_stats = compute_disagreement_per_timestep_stats(step_trajs, env_horizon)
        fig = plot_disagreement_by_timestep(step_trajs, env_horizon, step_idx)
        if fig is not None:
            if args.log_wandb:
                log_disc = {
                    "dagger_step": step_idx,
                    "global_epoch": eval_global_epoch,
                    "data/step_kind": step_kind,
                    "data/is_bc_step": step_idx == 0,
                    "disagreement_by_timestep": wandb.Image(fig),
                }
                if disc_stats is not None:
                    tbl = wandb.Table(columns=["timestep", "mean", "stderr"])
                    for t in range(env_horizon):
                        tbl.add_data(
                            t,
                            disc_stats["mean_per_timestep"][t],
                            disc_stats["stderr_per_timestep"][t],
                        )
                    log_disc["data/disagreement_per_timestep_table"] = tbl
                wandb.log(log_disc)
            plt.close(fig)

        # --- Evaluate ---
        print(f"\nEvaluating after DAgger step {step_idx}...")
        eval_policy = EnsembleTransformerPolicy(
            ensemble,
            context_horizon=env_horizon,
            env_horizon=env_horizon,
            sliding_window=True,
        )

        eval_save_dir = os.path.join(save_dir, f"dagger_step_{step_idx}", "eval")
        eval_results = evaluate_policy_on_envs(
            eval_envs=eval_envs,
            policy=eval_policy,
            eval_horizon=env_horizon,
            env_horizon=env_horizon,
            save_dir=eval_save_dir,
            env_name=args.env_name,
            plot=True,
        )

        if args.log_wandb:
            num_eval_trajs = args.num_eval_trajs
            final_mean = eval_results["mean_returns"][-1]
            final_std = eval_results["std_returns"][-1]
            eval_log = {
                "global_epoch": eval_global_epoch,
                "eval/return": final_mean,
                "eval/return_std": final_std,
                "eval/dagger_step": step_idx,
            }
            video_path = os.path.join(eval_save_dir, "eval_video.mp4")
            video_result = generate_eval_video(
                eval_results["trajs"], args.env_name, video_path, num_trajs=num_eval_trajs
            )
            if video_result:
                eval_log["eval/video"] = wandb.Video(video_path, fps=4, format="mp4")
            wandb.log(eval_log)

            # Eval returns histogram (same step metric as disagreement_by_timestep)
            episode_returns = eval_results["episode_returns"]
            hist_fig = plot_eval_returns_histogram(episode_returns, num_eval_trajs, step_idx)
            if hist_fig is not None:
                wandb.log({
                    "dagger_step": step_idx,
                    "eval_returns_histogram": wandb.Image(hist_fig),
                })
                plt.close(hist_fig)

        ep_ret = eval_results["episode_returns"].reshape(-1)
        dagger_snapshot = {
            "dagger_step": step_idx,
            "global_epoch_end": eval_global_epoch,
            "data/step_kind": step_kind,
            "data/is_bc_step": step_idx == 0,
            "dataset": {
                "train_trajectories": len(train_dataset),
                "test_trajectories": len(test_dataset),
                "expert_query_rate": float(expert_query_rate),
                "expert_query_rate_this_step": (
                    float(expert_query_rate_this_step)
                    if expert_query_rate_this_step is not None
                    else None
                ),
                "effective_size": float(effective_size_pct),
            },
            "disagreement_per_timestep": disc_stats,
            "disagreement_note": (
                "BC (step 0): expert-only rollout; values are constant "
                f"default_disagreement={args.default_disagreement} (no ensemble at collection). "
                "When pulling from wandb, filter rows with data/is_bc_step or data/step_kind=='bc'."
                if step_idx == 0
                else None
            ),
            "eval": {
                "mean_return_final": float(eval_results["mean_returns"][-1]),
                "std_return_final": float(eval_results["std_returns"][-1]),
                "mean_returns_curve": eval_results["mean_returns"].astype(float).tolist(),
                "std_returns_curve": eval_results["std_returns"].astype(float).tolist(),
                "episode_return_summary": {
                    "min": float(ep_ret.min()),
                    "max": float(ep_ret.max()),
                    "mean": float(ep_ret.mean()),
                    "num_rollouts": int(ep_ret.size),
                },
            },
        }
        run_metrics["dagger_steps"].append(dagger_snapshot)
        _persist_run_metrics(save_dir, run_metrics)

        print(
            f"Evaluation complete - Return: "
            f"{eval_results['mean_returns'][0]:.2f} +/- "
            f"{eval_results['std_returns'][0]:.2f}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Collect new data for next step ---
        if step_idx < args.dagger_steps - 1:
            step_policy = EnsembleTransformerPolicy(
                ensemble,
                context_horizon=env_horizon,
                env_horizon=env_horizon,
                sliding_window=False,
            )
            step_train_dataset, step_test_dataset = data_step_ensemble(
                save_dir, step_idx + 1, train_envs, test_envs,
                step_policy, env_horizon, args.disagreement_threshold, device,
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

    # ------------------------------------------------------------------
    # Save final models
    # ------------------------------------------------------------------
    if args.save_model:
        for k, model in enumerate(ensemble):
            torch.save(model.state_dict(), os.path.join(save_dir, f"final_model_{k}.pth"))
    print(f"\nTraining complete! Results saved to {save_dir}")

    if args.log_wandb:
        wandb.finish()
