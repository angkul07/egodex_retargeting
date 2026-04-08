"""
10_eval_act.py — load a trained ACT checkpoint and run rollouts in the
MuJoCo eval env.

For each rollout:
  1. `EvalEnv.reset()` with a randomized object position
  2. `policy.reset()` (clears the internal action queue)
  3. Loop up to `episode_length` env steps:
     - get observation
     - `policy.select_action(obs)` → 13-D action (uses ACT's chunked
       inference internally)
     - `env.step(action)`
  4. Record success / final object z

Reports success rate, mean object lift, and writes per-rollout details
to a JSONL log.

Run
---
    uv run python notebooks/10_eval_act.py \\
        --ckpt-dir outputs/stage4/act_full_pipeline \\
        --n-rollouts 50

Stage 4 §4.4 ablation runs use this with `--ckpt-dir` pointed at each
condition's training output.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Headless OpenGL must come before importing mujoco. eval_env.py also sets
# this, but ensure it's set BEFORE we import torch / lerobot too in case
# anything pre-loads OpenGL.
os.environ.setdefault("MUJOCO_GL", "egl")

# Apply lerobot PyAV patch (required even for inference, since ACT model
# instantiation goes through lerobot's processor pipeline which imports
# lerobot.datasets.video_utils).
from mimicdreamer_egodex.lerobot_pyav_patch import apply as apply_pyav_patch  # noqa: E402

apply_pyav_patch()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from lerobot.policies.act.processor_act import make_act_pre_post_processors  # noqa: E402

from mimicdreamer_egodex.eval_env import EvalEnv  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO / "outputs/lerobot/mimicdreamer_egodex_basic_pick_place_full"
DEFAULT_REPO_ID = "mimicdreamer-egodex/basic_pick_place_full"


def build_policy(ckpt_dir: Path, dataset_root: Path, repo_id: str, device: torch.device):
    """Reconstruct an ACTPolicy with the same config used at training time
    and load weights from `ckpt_dir/latest.pt`. Pulls dataset stats from
    the on-disk LeRobotDataset to wire the pre/post processors."""
    # Pull stats from dataset (no episodes filter — we just need stats here)
    ds = LeRobotDataset(repo_id=repo_id, root=dataset_root)
    dataset_stats = ds.meta.stats

    input_features = {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(13,)),
    }
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(13,)),
    }

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=10,
        n_action_steps=8,
        n_obs_steps=1,
        device=str(device),
    )
    policy = ACTPolicy(cfg).to(device)

    ckpt_path = ckpt_dir / "latest.pt"
    if not ckpt_path.exists():
        # Try the most recent step file as fallback
        steps = sorted(ckpt_dir.glob("ckpt_step*.pt"),
                       key=lambda p: int(p.stem.replace("ckpt_step", "")))
        if not steps:
            raise FileNotFoundError(f"no checkpoint in {ckpt_dir}")
        ckpt_path = steps[-1]
    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    policy.load_state_dict(state)
    policy.eval()  # eval() is fine at INFERENCE time — it's only the loss-mode
                   # forward() that crashes (see notebooks/09 evaluate())

    preprocessor, postprocessor = make_act_pre_post_processors(
        config=cfg, dataset_stats=dataset_stats
    )
    return policy, preprocessor, postprocessor, cfg


def obs_to_batch(obs: dict, device: torch.device) -> dict:
    """Convert a dict of numpy obs into a batched-by-1 torch dict on device."""
    out = {}
    for k, v in obs.items():
        t = torch.as_tensor(v).unsqueeze(0).to(device)
        out[k] = t
    return out


def run_rollouts(
    policy,
    preprocessor,
    postprocessor,
    env: EvalEnv,
    n_rollouts: int,
    device: torch.device,
    log_path: Path | None = None,
) -> dict:
    successes = 0
    object_lifts = []
    per_rollout = []

    for r in range(n_rollouts):
        obs = env.reset()
        if hasattr(policy, "reset"):
            policy.reset()
        traj_obj_z = []
        success = False
        for t in range(env.episode_length):
            batch = obs_to_batch(obs, device)
            batch = preprocessor(batch)
            with torch.no_grad():
                action_t = policy.select_action(batch)
            # postprocessor unnormalizes the action tensor back to dataset
            # units (matches the canonical lerobot_eval.py pattern)
            action_t = postprocessor(action_t)
            action = action_t[0].cpu().numpy().astype(np.float32)
            res = env.step(action)
            traj_obj_z.append(res.object_z)
            obs = res.obs
            if res.done:
                success = res.success
                break

        lift = float(max(traj_obj_z) - env._obj_init_z) if traj_obj_z else 0.0
        object_lifts.append(lift)
        if success:
            successes += 1
        per_rollout.append({
            "rollout": r,
            "success": bool(success),
            "max_lift_m": lift,
            "n_steps": len(traj_obj_z),
        })
        print(
            f"  [{r + 1:>3}/{n_rollouts}] success={int(success)}  "
            f"max_lift={lift * 1000:6.1f} mm  n_steps={len(traj_obj_z)}",
            flush=True,
        )

    summary = {
        "n_rollouts": n_rollouts,
        "n_successes": successes,
        "success_rate": successes / max(1, n_rollouts),
        "mean_max_lift_m": float(np.mean(object_lifts)) if object_lifts else 0.0,
        "median_max_lift_m": float(np.median(object_lifts)) if object_lifts else 0.0,
        "max_max_lift_m": float(np.max(object_lifts)) if object_lifts else 0.0,
    }
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as f:
            for r in per_rollout:
                f.write(json.dumps(r) + "\n")
            f.write(json.dumps({"summary": summary}) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rollout a trained ACT policy in the MuJoCo eval env")
    ap.add_argument("--ckpt-dir", type=Path, default=REPO / "outputs/stage4/act_full_pipeline")
    ap.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    ap.add_argument("--n-rollouts", type=int, default=20)
    ap.add_argument("--episode-length", type=int, default=120)  # 4 s @ 30 Hz
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    policy, preprocessor, postprocessor, cfg = build_policy(
        args.ckpt_dir, args.dataset_root, args.repo_id, device
    )
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"ACT loaded: {n_params / 1e6:.1f}M params")

    env = EvalEnv(episode_length=args.episode_length, seed=args.seed)
    print(f"EvalEnv built: episode_length={args.episode_length}")

    out_dir = args.out_dir or (args.ckpt_dir / "eval")
    log_path = out_dir / "rollouts.jsonl"

    print(f"\nRolling out {args.n_rollouts} episodes...")
    t_start = time.time()
    summary = run_rollouts(
        policy, preprocessor, postprocessor, env, args.n_rollouts, device, log_path
    )
    elapsed = time.time() - t_start

    print(f"\n=== Summary ({args.n_rollouts} rollouts in {elapsed:.1f}s) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  rollout log: {log_path}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
