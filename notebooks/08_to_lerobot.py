"""
08_to_lerobot.py — convert Stage 1+2+3 outputs into a LeRobot dataset.

For each episode:
  - load the stabilized MP4 from `outputs/stage1/<idx>_stabilized.mp4`
    and resize each frame to 224x224 RGB (lerobot/ACT convention)
  - load `q_arm (T,6)` + `gripper (T,)` from `outputs/stage2/<idx>_actions.npz`
  - load `q_finger (T,6)` from `outputs/stage3/<idx>_fingers.npz`
  - read the language description from the EgoDex HDF5 attrs
  - build the 13-D action vector `[arm_6, gripper_1, finger_6]`
  - feed each frame into a `LeRobotDataset` writer

State vs action: this is an offline-converted dataset, not a live robot
recording, so we don't have a separate "current state" measurement —
the IK targets ARE the trajectory the robot would have followed. We use
`observation.state[t] = action[t]` for simplicity. ACT uses the state
input as a proprioceptive anchor and the action as the prediction
target; both being equal at training time gives the policy a clean
"current pose → next pose chunk" mapping.

The dataset is built with the FULL pipeline (stabilized frames, smooth-IK
arm, fingertip-spread gripper, retargeted Inspire fingers). Ablation
variants for §4.4 will be derived later by either:
  - re-running this script with different inputs (no-stab, no-smooth-IK,
    no-fingers), or
  - filtering at training time by zeroing out specific action dimensions.

CLI
---
    uv run python notebooks/08_to_lerobot.py --max-episodes 3   # smoke test
    uv run python notebooks/08_to_lerobot.py                    # full 277 eps
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import cv2
import h5py
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO = Path(__file__).resolve().parents[1]
TASK_DIR = REPO / "data/test/basic_pick_place"
STAGE1_DIR = REPO / "outputs/stage1"
STAGE2_DIR = REPO / "outputs/stage2"
STAGE3_DIR = REPO / "outputs/stage3"
DEFAULT_ROOT = REPO / "outputs/lerobot/mimicdreamer_egodex_basic_pick_place_full"
REPO_ID = "mimicdreamer-egodex/basic_pick_place_full"

IMG_H, IMG_W = 224, 224
FPS = 30
ACTION_DIM = 13  # 6 arm + 1 gripper + 6 finger


def load_episode(idx: int) -> dict | None:
    """Pull stage1 mp4 path + 13-D action vector + task description for one
    episode. Returns None if any input is missing or shapes disagree."""
    s1_mp4 = STAGE1_DIR / f"{idx}_stabilized.mp4"
    s2_npz = STAGE2_DIR / f"{idx}_actions.npz"
    s3_npz = STAGE3_DIR / f"{idx}_fingers.npz"
    h5 = TASK_DIR / f"{idx}.hdf5"
    if not (s1_mp4.exists() and s2_npz.exists() and s3_npz.exists() and h5.exists()):
        return None

    s2 = np.load(s2_npz, allow_pickle=True)
    s3 = np.load(s3_npz, allow_pickle=True)

    q_arm = s2["q"]                 # (T, 6)
    gripper = s2["gripper"]         # (T,)
    q_finger = s3["q_finger"]       # (T, 6)
    T = len(q_arm)
    if not (len(gripper) == T and len(q_finger) == T):
        return None

    action = np.concatenate(
        [q_arm, gripper[:, None], q_finger], axis=1
    ).astype(np.float32)  # (T, 13)

    with h5py.File(h5, "r") as f:
        desc = f.attrs.get("description", "")
    desc = desc.decode() if isinstance(desc, bytes) else str(desc)
    if not desc:
        desc = "pick and place"

    return {
        "idx": idx,
        "mp4": s1_mp4,
        "T": T,
        "action": action,
        "task": desc,
    }


def load_video_frames_resized(mp4_path: Path, n_frames: int, h: int, w: int) -> np.ndarray:
    """Read `n_frames` BGR frames from `mp4_path`, resize to (h, w),
    convert to RGB. Returns (n_frames, h, w, 3) uint8."""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise FileNotFoundError(mp4_path)
    out = np.zeros((n_frames, h, w, 3), dtype=np.uint8)
    actually_read = 0
    while actually_read < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        out[actually_read] = rgb
        actually_read += 1
    cap.release()
    if actually_read < n_frames:
        # Pad missing frames at the end with the last good frame (rare; only
        # if the MP4 is truncated)
        if actually_read > 0:
            out[actually_read:] = out[actually_read - 1]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Convert Stage 1+2+3 to LeRobotDataset")
    ap.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Cap on episodes processed (quick test). Default = all.",
    )
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument(
        "--force",
        action="store_true",
        help="Delete `--root` first if it exists.",
    )
    args = ap.parse_args(argv)

    if args.root.exists():
        if args.force:
            print(f"deleting existing {args.root}")
            shutil.rmtree(args.root)
        else:
            print(
                f"ERROR: dataset root {args.root} already exists. Re-run with "
                f"--force to delete and rebuild."
            )
            return 1

    paths = sorted(TASK_DIR.glob("*.hdf5"), key=lambda p: int(p.stem))
    if args.max_episodes is not None:
        paths = paths[: args.max_episodes]
    print(f"Building LeRobotDataset from {len(paths)} episodes -> {args.root}")

    # Feature schema. lerobot expects (H, W, C) for image/video shapes.
    features = {
        "observation.image": {
            "dtype": "video",
            "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": None,
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        root=args.root,
        fps=FPS,
        features=features,
        use_videos=True,
        robot_type="ur5e+inspire",
    )

    t_start = time.time()
    skipped: list[int] = []
    n_done = 0
    for i, h5p in enumerate(paths):
        idx = int(h5p.stem)
        ep = load_episode(idx)
        if ep is None:
            skipped.append(idx)
            continue

        frames = load_video_frames_resized(ep["mp4"], ep["T"], IMG_H, IMG_W)
        for t in range(ep["T"]):
            dataset.add_frame(
                {
                    "observation.image": frames[t],
                    "observation.state": ep["action"][t],
                    "action": ep["action"][t],
                    "task": ep["task"],
                }
            )
        dataset.save_episode()
        n_done += 1

        if (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (len(paths) - i - 1)
            print(
                f"  [{i + 1:>3}/{len(paths)}] ep{idx:>3}  "
                f"T={ep['T']:>4}  task={ep['task'][:60]!r}  "
                f"elapsed={elapsed:5.1f}s  eta={eta:5.0f}s",
                flush=True,
            )

    dataset.finalize()
    elapsed = time.time() - t_start
    print(
        f"\nDone. {n_done} episodes saved, {len(skipped)} skipped "
        f"in {elapsed:.1f}s."
    )
    if skipped:
        print(f"Skipped indices: {skipped}")
    print(f"Dataset root: {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
