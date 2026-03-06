"""
History-Based DAgger Training.

Simple DAgger with a single Decision Transformer learner whose context length
equals the environment horizon (fixed throughout all DAgger steps).

Algorithm:
1. Expert collects an initial dataset; run behavior cloning.
2. For each subsequent DAgger step:
   a. Learner rolls out for one episode (env_horizon steps).
   b. Expert labels every history the learner visited.
   c. New data is merged into the dataset; supervised learning is repeated.

No curriculum over context horizons, no loss masks, no ensemble.
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
import torch
import torch.nn.functional as F
import tqdm
import wandb

from create_envs import create_env
from collect_data import get_dagger_dataset, merge_sequence_datasets
from dataset import collate_fn
from eval_policy import evaluate_policy_on_envs
from get_rollout_policy import get_rollout_policy
from models import DecisionTransformer


def get_optimizer_scheduler(model, total_steps, lr, warmup_ratio):
    """
    Create optimizer with warmup + cosine decay schedule.

    Args:
        model: nn.Module whose parameters will be optimized.
        total_steps: Total number of optimizer steps across all epochs.
        lr: Peak learning rate.
        warmup_ratio: Fraction of total_steps used for linear warmup.

    Returns:
        optimizer: AdamW optimizer.
        scheduler: SequentialLR (linear warmup -> cosine decay).
    """
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


def data_step(save_dir, step_id, train_envs, test_envs, rollout_policy, horizon):
    """
    Collect (or load cached) DAgger data for one step.

    Uses dagger_rollout() under the hood: the rollout_policy executes actions
    while the expert labels every visited state.

    Args:
        save_dir: Root directory for caching datasets.
        step_id: Current DAgger iteration index (0 = expert-only).
        train_envs: List of vectorized training environments.
        test_envs: List of vectorized test environments.
        rollout_policy: Policy used to act in the environment.
            - Step 0: ExpertPolicy.
            - Step k>0: TransformerPolicy (the current learner).
        horizon: Number of steps per trajectory (= env_horizon, fixed).

    Returns:
        train_dataset: SequenceDataset with trajectories shaped
            {
                "states":         (horizon, state_dim),
                "actions":        (horizon, action_dim),   # rollout policy
                "expert_actions": (horizon, action_dim),   # expert labels
                "rewards":        (horizon,),
                "dones":          (horizon,),
            }
        test_dataset: Same structure, from test environments.
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
        print(f"Collecting new data for step {step_id}")
        train_dataset, test_dataset = get_dagger_dataset(
            train_envs, test_envs, rollout_policy, horizon
        )
        with open(train_path, "wb") as f:
            pickle.dump(train_dataset, f)
        with open(test_path, "wb") as f:
            pickle.dump(test_dataset, f)

    return train_dataset, test_dataset


def train_step(
    step_id,
    model,
    optimizer,
    scheduler,
    train_loader,
    test_loader,
    save_dir,
    args,
    device,
    action_dim,
):
    """
    Train the model for one DAgger step (multiple epochs of supervised learning).

    The inner forward() computes cross-entropy loss between the model's
    predicted action logits and the expert-labeled actions over ALL valid
    timesteps (no extra loss mask).

    Batch shapes after collate_fn:
        states:         (B, H, state_dim)
        actions:        (B, H, action_dim)
        expert_actions: (B, H, action_dim)
        rewards:        (B, H)
        dones:          (B, H)
        attention_mask: (B, H)       — 1 for valid positions, 0 for padding

    Model output:
        pred_actions:   (B, H, action_dim)   — logits (discrete)

    Loss:
        cross_entropy(pred.reshape(-1, A), expert.reshape(-1, A), reduction='none')
        masked by attention_mask, averaged over valid positions.

    Args:
        step_id: Current DAgger iteration.
        model: DecisionTransformer to train.
        optimizer: AdamW optimizer.
        scheduler: LR scheduler.
        train_loader: DataLoader over training SequenceDataset.
        test_loader: DataLoader over test SequenceDataset.
        save_dir: Directory for saving checkpoints.
        args: Namespace with num_epochs, eval_interval, save_interval,
              gradient_clip, log_wandb.
        device: torch device.
        action_dim: Number of discrete actions.

    Returns:
        model: The trained model (same object, updated in-place).
    """
    step_save_dir = os.path.join(save_dir, f"dagger_step_{step_id}")
    os.makedirs(step_save_dir, exist_ok=True)

    eval_freq = max(1, int(args.eval_interval * args.num_epochs))
    save_freq = max(1, int(args.save_interval * args.num_epochs))

    def forward(batch):
        """
        Compute cross-entropy loss over all valid timesteps.

        batch keys (all on device after move):
            states:         (B, H, state_dim)
            actions:        (B, H, action_dim)
            expert_actions: (B, H, action_dim)   — targets (one-hot)
            rewards:        (B, H)
            dones:          (B, H)
            attention_mask: (B, H)

        Returns:
            loss: scalar tensor.
            stats: dict with "loss" float for logging.
        """
        batch = {k: v.to(device) for k, v in batch.items()}
        true_actions = batch["expert_actions"]          # (B, H, A)
        pred_actions, _ = model(batch)                  # (B, H, A)

        per_token_loss = F.cross_entropy(
            pred_actions.reshape(-1, action_dim),
            true_actions.reshape(-1, action_dim),
            reduction="none",
        )  # (B*H,)

        mask = batch["attention_mask"]                  # (B, H)
        per_token_loss = per_token_loss.reshape(mask.shape)
        loss = (per_token_loss * mask).sum() / mask.sum()

        return loss, {"loss": loss.item()}

    for epoch in tqdm.tqdm(range(args.num_epochs), desc=f"Training DAgger Step {step_id}"):
        # --- Evaluation ---
        if epoch % eval_freq == 0 or epoch == args.num_epochs - 1:
            model.eval()
            eval_stats = defaultdict(list)

            with torch.no_grad():
                for batch in test_loader:
                    loss, stats = forward(batch)
                    for k, v in stats.items():
                        eval_stats[k].append(v)

            for k in eval_stats:
                eval_stats[k] = np.mean(eval_stats[k])

            if args.log_wandb:
                for k, v in eval_stats.items():
                    wandb.log({f"dagger-{step_id}/test_{k}": v})

            print(f"Epoch {epoch} - Test Loss: {eval_stats['loss']:.4f}")

        # --- Training ---
        model.train()
        train_stats = defaultdict(list)

        for batch in train_loader:
            loss, stats = forward(batch)

            optimizer.zero_grad()
            loss.backward()

            if args.log_wandb:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                wandb.log({
                    f"dagger-{step_id}/grad_norm": total_norm,
                    f"dagger-{step_id}/lr": optimizer.param_groups[0]["lr"],
                })

            if args.gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            for k, v in stats.items():
                train_stats[k].append(v)

        if args.log_wandb:
            for k, v in train_stats.items():
                wandb.log({f"dagger-{step_id}/{k}": np.mean(v)})

        if epoch % save_freq == 0:
            torch.save(
                model.state_dict(),
                os.path.join(step_save_dir, f"model_epoch_{epoch}.pth"),
            )

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="History-Based DAgger Training")

    parser.add_argument("--exp_name", type=str, default="history_dagger")
    parser.add_argument("--env_name", type=str, default="darkroom-easy")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dataset_size", type=int, default=10000)
    parser.add_argument("--dagger_steps", type=int, default=3)
    parser.add_argument("--n_envs", type=int, default=10000)

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
    parser.add_argument("--save_interval", type=float, default=0.1)

    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dpt-sweep")
    parser.add_argument("--wandb_entity", type=str, default=None)

    parser.add_argument("--save_dir", type=str, default="./history_dagger_results")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    if args.log_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"{args.exp_name}-{args.env_name}-seed{args.seed}",
        )

    # ------------------------------------------------------------------
    # Seed everything
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

    save_dir = os.path.join(
        args.save_dir, f"{args.exp_name}-{args.env_name}-seed{args.seed}"
    )
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Environments
    # ------------------------------------------------------------------
    print(f"Creating environments: {args.env_name}")
    train_envs, test_envs, eval_envs = create_env(
        args.env_name, args.dataset_size, args.n_envs
    )

    state_dim = train_envs[0]._envs[0].state_dim
    action_dim = train_envs[0]._envs[0].action_dim
    env_horizon = train_envs[0]._envs[0].horizon
    print(f"State dim: {state_dim}, Action dim: {action_dim}, Env horizon: {env_horizon}")

    # ------------------------------------------------------------------
    # Model — context length = env_horizon (fixed)
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

    model = DecisionTransformer(model_args).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # Step 0: expert collects initial dataset
    # ------------------------------------------------------------------
    init_rollout_policy = get_rollout_policy("expert")
    train_dataset, test_dataset = data_step(
        save_dir, 0, train_envs, test_envs, init_rollout_policy, env_horizon
    )

    total_steps = len(train_dataset) // args.batch_size * args.num_epochs
    optimizer, scheduler = get_optimizer_scheduler(
        model, total_steps, args.lr, args.warmup_ratio
    )

    # ------------------------------------------------------------------
    # DAgger loop
    # ------------------------------------------------------------------
    for step_idx in range(args.dagger_steps):
        print(f"\n{'='*60}")
        print(f"DAgger Step {step_idx}/{args.dagger_steps}")
        print(f"Dataset size - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
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

        model = train_step(
            step_id=step_idx,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            test_loader=test_loader,
            save_dir=save_dir,
            args=args,
            device=device,
            action_dim=action_dim,
        )

        # --- Evaluate ---
        print(f"\nEvaluating after DAgger step {step_idx}...")
        eval_policy = get_rollout_policy(
            "decision_transformer",
            model=model,
            context_horizon=env_horizon,
            env_horizon=env_horizon,
            sliding_window=True,
        )

        eval_save_dir = os.path.join(save_dir, f"dagger_step_{step_idx}", "eval")
        eval_horizon = env_horizon

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
                f"eval/step_{step_idx}_return": final_mean,
                f"eval/step_{step_idx}_return_std": final_std,
            })

        print(
            f"Evaluation complete - Return: "
            f"{eval_results['mean_returns'][0]:.2f} +/- "
            f"{eval_results['std_returns'][0]:.2f}"
        )

        # Free GPU memory cached by eval before large-batch data collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Collect new data for next step (learner rollout, expert labels) ---
        if step_idx < args.dagger_steps - 1:
            step_policy = get_rollout_policy(
                "decision_transformer",
                model=model,
                context_horizon=env_horizon,
                env_horizon=env_horizon,
                sliding_window=False,
            )

            step_train_dataset, step_test_dataset = data_step(
                save_dir, step_idx + 1, train_envs, test_envs,
                step_policy, env_horizon,
            )

            train_dataset = merge_sequence_datasets(train_dataset, step_train_dataset)
            test_dataset = merge_sequence_datasets(test_dataset, step_test_dataset)

            total_steps = len(train_dataset) // args.batch_size * args.num_epochs
            optimizer, scheduler = get_optimizer_scheduler(
                model, total_steps, args.lr, args.warmup_ratio
            )

    # ------------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------------
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))
    print(f"\nTraining complete! Results saved to {save_dir}")

    if args.log_wandb:
        wandb.finish()
