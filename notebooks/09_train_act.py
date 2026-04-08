"""
09_train_act.py — train an ACT policy on the LeRobot dataset built by 08.

Self-contained training loop. Doesn't depend on lerobot's `train.py` CLI
(which uses `accelerate` + `draccus` config injection — too much extra
machinery for a small replication run that needs ablation flexibility).

What it does
------------
1. Apply the PyAV monkey-patch BEFORE importing lerobot (D-012).
2. Load the LeRobotDataset, filter to a train episode subset, get the
   per-feature stats lerobot computed at finalize time.
3. Wire `delta_timestamps` so each frame yields an action chunk of
   length `chunk_size` (10 by default).
4. Build `ACTConfig` for our 13-D action + 224x224 image + 13-D state.
5. Build the ACT policy and the matching pre/post processors via
   `make_act_pre_post_processors` (these handle the per-feature
   normalization based on the dataset stats).
6. Manual training loop on cuda: AdamW + autocast bf16 + grad clip +
   periodic val pass + checkpoint saving.

Output
------
`<ckpt-dir>/`
    config.json            — the ACTConfig dump
    train_log.jsonl        — per-step train metrics
    val_log.jsonl          — per-eval val metrics
    ckpt_step{N}.pt        — periodic checkpoints (also `latest.pt`)
    args.json              — the CLI args used

CLI
---
    uv run python notebooks/09_train_act.py \\
        --steps 20000 --batch-size 64 \\
        --ckpt-dir outputs/stage4/act_full_pipeline
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# Apply lerobot's PyAV-only video decoder patch BEFORE importing lerobot.
from mimicdreamer_egodex.lerobot_pyav_patch import apply as apply_pyav_patch

apply_pyav_patch()

import torch
import torch.nn.functional as F  # noqa: F401
from torch.utils.data import DataLoader

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: F401
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO / "outputs/lerobot/mimicdreamer_egodex_basic_pick_place_full"
DEFAULT_REPO_ID = "mimicdreamer-egodex/basic_pick_place_full"
DEFAULT_CKPT_DIR = REPO / "outputs/stage4/act_full_pipeline"


def split_episodes(num_episodes: int, val_frac: float, seed: int = 42) -> tuple[list[int], list[int]]:
    """Deterministic 80/20 split: last `val_frac * num_episodes` episodes go to val.
    Sequential split is fine here because EgoDex episode order is not correlated
    with task or wearer (manually checked: distribution looks well-mixed)."""
    n_val = max(1, int(round(num_episodes * val_frac)))
    val_idx = list(range(num_episodes - n_val, num_episodes))
    train_idx = list(range(num_episodes - n_val))
    return train_idx, val_idx


def make_features(action_dim: int, img_h: int, img_w: int) -> tuple[dict, dict]:
    input_features = {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, img_h, img_w)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(action_dim,)),
    }
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
    }
    return input_features, output_features


def move_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


@torch.no_grad()
def evaluate(policy, preprocessor, val_loader, device, max_batches: int = 30) -> dict:
    """Validation pass.

    NOTE: we keep the policy in `train()` mode here, NOT `eval()`. ACT's
    forward() in `eval()` mode skips the VAE encoder and then crashes on
    `log_sigma_x2_hat = None` when computing the KL loss term — that's an
    upstream ACT bug. Wrapping with `torch.no_grad()` is enough to keep
    val pass cheap and gradient-free, and the per-frame loss numbers are
    still comparable across train and val splits.
    """
    losses = []
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        batch = move_to_device(batch, device)
        batch = preprocessor(batch)
        loss, _ = policy.forward(batch)
        losses.append(loss.item())
    return {"val_loss": float(sum(losses) / max(1, len(losses))), "val_n_batches": len(losses)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train ACT on the LeRobot dataset")
    ap.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    ap.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip-norm", type=float, default=10.0)
    ap.add_argument("--chunk-size", type=int, default=10)
    ap.add_argument("--n-action-steps", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.20)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-amp", action="store_true", help="Disable bf16 autocast")
    args = ap.parse_args(argv)

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    (args.ckpt_dir / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load dataset metadata to get episode count + stats + fps ──────────
    print(f"Loading dataset metadata from {args.dataset_root}")
    meta_only = LeRobotDataset(repo_id=args.repo_id, root=args.dataset_root)
    num_episodes = meta_only.num_episodes
    fps = meta_only.fps
    print(f"  num_episodes: {num_episodes}  num_frames: {meta_only.num_frames}  fps: {fps}")

    # Pull stats per feature for the normalizer
    dataset_stats = meta_only.meta.stats
    print(f"  dataset_stats keys: {list(dataset_stats.keys())}")

    # ── Train/val split ───────────────────────────────────────────────────
    train_eps, val_eps = split_episodes(num_episodes, args.val_frac, seed=args.seed)
    print(f"  train episodes: {len(train_eps)}  val episodes: {len(val_eps)}")

    # ── Delta timestamps for action chunking ──────────────────────────────
    action_delta = [i / fps for i in range(args.chunk_size)]
    delta_timestamps = {"action": action_delta}

    # ── Build train + val datasets with the action delta ──────────────────
    train_ds = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.dataset_root,
        episodes=train_eps,
        delta_timestamps=delta_timestamps,
    )
    val_ds = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.dataset_root,
        episodes=val_eps,
        delta_timestamps=delta_timestamps,
    )
    print(f"  train_ds: {len(train_ds)} samples  val_ds: {len(val_ds)} samples")

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    # ── Build policy + processors ─────────────────────────────────────────
    input_features, output_features = make_features(action_dim=13, img_h=224, img_w=224)
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        n_obs_steps=1,
        device=str(device),
    )
    policy = ACTPolicy(cfg).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"ACTPolicy: {n_params / 1e6:.1f}M params")

    preprocessor, postprocessor = make_act_pre_post_processors(
        config=cfg, dataset_stats=dataset_stats
    )

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    use_amp = (not args.no_amp) and device.type == "cuda"
    autocast_dtype = torch.bfloat16
    print(f"Mixed precision: bf16 autocast = {use_amp}")

    # Save the config + initial CLI args
    (args.ckpt_dir / "config.json").write_text(
        json.dumps({"action_dim": 13, "img": [224, 224], "fps": fps,
                    "chunk_size": args.chunk_size, "n_action_steps": args.n_action_steps,
                    "n_train_episodes": len(train_eps), "n_val_episodes": len(val_eps),
                    "n_train_samples": len(train_ds), "n_val_samples": len(val_ds)}, indent=2)
    )

    # ── Training loop ─────────────────────────────────────────────────────
    train_log_path = args.ckpt_dir / "train_log.jsonl"
    val_log_path = args.ckpt_dir / "val_log.jsonl"
    train_log = train_log_path.open("w")
    val_log = val_log_path.open("w")

    step = 0
    train_iter = iter(train_loader)
    losses_window: list[float] = []
    t_start = time.time()
    print(f"\nStarting training: {args.steps} steps, batch_size={args.batch_size}")

    while step < args.steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = move_to_device(batch, device)
        batch = preprocessor(batch)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                loss, output_dict = policy.forward(batch)
        else:
            loss, output_dict = policy.forward(batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip_norm)
        optimizer.step()

        step += 1
        losses_window.append(loss.item())

        if step % args.log_every == 0:
            mean_loss = sum(losses_window) / len(losses_window)
            losses_window.clear()
            elapsed = time.time() - t_start
            it_per_s = step / elapsed
            eta = (args.steps - step) / max(1e-9, it_per_s)
            row = {
                "step": step,
                "loss": mean_loss,
                "grad_norm": float(grad_norm),
                "lr": optimizer.param_groups[0]["lr"],
                "it_per_s": it_per_s,
            }
            train_log.write(json.dumps(row) + "\n")
            train_log.flush()
            print(
                f"  step {step:>6d}/{args.steps}  loss={mean_loss:.4f}  "
                f"grad={grad_norm:.2f}  it/s={it_per_s:5.1f}  eta={eta / 60:5.1f} min",
                flush=True,
            )

        if step % args.eval_every == 0 or step == args.steps:
            val = evaluate(policy, preprocessor, val_loader, device)
            val["step"] = step
            val_log.write(json.dumps(val) + "\n")
            val_log.flush()
            print(f"  ----- val @ step {step}: loss={val['val_loss']:.4f}  ({val['val_n_batches']} batches)")

        if step % args.save_every == 0 or step == args.steps:
            ckpt_path = args.ckpt_dir / f"ckpt_step{step}.pt"
            torch.save(
                {
                    "step": step,
                    "policy_state": policy.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                ckpt_path,
            )
            torch.save(policy.state_dict(), args.ckpt_dir / "latest.pt")
            print(f"  ----- saved {ckpt_path.name}")

    train_log.close()
    val_log.close()
    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
