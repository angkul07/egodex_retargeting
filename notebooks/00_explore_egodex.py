"""
Stage 0.3 — EgoDex data exploration script.

NOTE: The HDF5 schema documented in `initial_plan.md` §0.3 turned out to be
incorrect for the actual `test.zip` (Apple, downloaded 2026-04-07). The real
layout, discovered by reading `data/test/basic_pick_place/0.hdf5`, is:

    /camera/intrinsic                  (3, 3)         camera intrinsics
    /transforms/camera                 (T, 4, 4)      camera-to-world per frame
    /transforms/leftHand               (T, 4, 4)      left wrist SE(3)
    /transforms/rightHand              (T, 4, 4)      right wrist SE(3)
    /transforms/<jointName>            (T, 4, 4)      every other body/finger joint
    /confidences/<jointName>           (T,)           ARKit confidence per joint
    (no /hand/{l,r}/joints, no /hand/{l,r}/confidence, no /language/annotation)

There are 25 joints per hand (24 finger + 1 wrist) split across many top-level
datasets — the script enumerates them rather than relying on a packed array.

Usage:
    uv run python notebooks/00_explore_egodex.py [path/to/episode.hdf5]

If no path is given, the script auto-discovers the first `basic_pick_place`
HDF5 under `data/test/`.

Per CLAUDE.md, run with tee so the run-log is captured:
    uv run python notebooks/00_explore_egodex.py 2>&1 \\
        | tee logs/runs/$(date +%Y-%m-%d_%H%M%S)_explore.log
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

# 25 joints per hand: 1 wrist (`<side>Hand`) + 4 fingers x 5 joints (Metacarpal,
# Knuckle, IntermediateBase, IntermediateTip, Tip) + thumb x 4 joints (no
# Metacarpal in ARKit's thumb model). 1 + 4*5 + 4 = 25.
THUMB_JOINTS = ["ThumbKnuckle", "ThumbIntermediateBase", "ThumbIntermediateTip", "ThumbTip"]
FINGER_JOINTS_TEMPLATE = [
    "{f}Metacarpal", "{f}Knuckle", "{f}IntermediateBase", "{f}IntermediateTip", "{f}Tip",
]
NON_THUMB_FINGERS = ["IndexFinger", "MiddleFinger", "RingFinger", "LittleFinger"]
FINGERTIPS = ["ThumbTip", "IndexFingerTip", "MiddleFingerTip", "RingFingerTip", "LittleFingerTip"]


def hand_joint_names(side: str) -> list[str]:
    """Return all 25 joint dataset names for one hand, e.g. 'leftHand', 'leftThumbTip'."""
    assert side in ("left", "right")
    out = [f"{side}Hand"]                                   # wrist
    out.extend(f"{side}{j}" for j in THUMB_JOINTS)          # 4 thumb joints
    for finger in NON_THUMB_FINGERS:                        # 4 fingers x 5 joints
        out.extend(f"{side}{tmpl.format(f=finger)}" for tmpl in FINGER_JOINTS_TEMPLATE)
    return out


def find_default_episode() -> Path | None:
    """Walk data/test/ and return the first basic_pick_place HDF5 if any, else any HDF5."""
    root = Path("data/test")
    if not root.exists():
        return None
    preferred = sorted(root.rglob("basic_pick_place/*.hdf5"))
    if preferred:
        return preferred[0]
    any_hdf5 = sorted(root.rglob("*.hdf5"))
    return any_hdf5[0] if any_hdf5 else None


def report_hand(f: h5py.File, side: str) -> None:
    """Variance + confidence report for one hand using the real schema."""
    print(f"\n=== {side.upper()} HAND ===")
    joint_names = hand_joint_names(side)

    wrist_key = f"transforms/{side}Hand"
    if wrist_key not in f:
        print(f"  no {wrist_key} dataset — skipping")
        return

    wrist = f[wrist_key][:]                # (T, 4, 4)
    wrist_pos = wrist[:, :3, 3]
    T = wrist_pos.shape[0]
    print(f"  episode length: {T} frames (~{T / 30.0:.2f} s @ 30 Hz)")
    for axis, name in zip(range(3), "xyz"):
        lo, hi = wrist_pos[:, axis].min(), wrist_pos[:, axis].max()
        print(f"  wrist {name}: {lo: .3f} -> {hi: .3f}  (range {hi - lo:.3f} m)")

    # Stack all 25 joint positions for this hand: (T, 25, 3)
    missing = [n for n in joint_names if f"transforms/{n}" not in f]
    if missing:
        print(f"  WARNING: {len(missing)} expected joints missing: {missing[:3]}...")
        joint_names = [n for n in joint_names if f"transforms/{n}" in f]

    joints = np.stack([f[f"transforms/{n}"][:, :3, 3] for n in joint_names], axis=1)
    print(f"  joints shape: {joints.shape} (T, J, 3)")

    # Hand-openness proxy (Stage 2.4): mean fingertip-to-wrist distance
    tip_indices = [joint_names.index(f"{side}{tip}") for tip in FINGERTIPS if f"{side}{tip}" in joint_names]
    wrist_idx = joint_names.index(f"{side}Hand")
    tips = joints[:, tip_indices, :]                    # (T, 5, 3)
    wrist_pts = joints[:, wrist_idx:wrist_idx + 1, :]    # (T, 1, 3)
    spread = np.linalg.norm(tips - wrist_pts, axis=2).mean(axis=1)
    print(
        f"  fingertip-to-wrist mean spread: "
        f"min {spread.min():.3f}  max {spread.max():.3f}  "
        f"range {spread.max() - spread.min():.3f} m"
    )

    # Confidence — average across this hand's joints
    conf_keys = [f"confidences/{n}" for n in joint_names if f"confidences/{n}" in f]
    if conf_keys:
        conf = np.stack([f[k][:] for k in conf_keys], axis=1)  # (T, J)
        per_frame_mean = conf.mean(axis=1)
        print(
            f"  ARKit confidence (mean over {len(conf_keys)} joints): "
            f"mean {per_frame_mean.mean():.3f}  "
            f"min {per_frame_mean.min():.3f}  max {per_frame_mean.max():.3f}  "
            f"frac>0.8 {(per_frame_mean > 0.8).mean():.3f}"
        )
    else:
        print("  no confidence datasets found for this hand")


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        episode = Path(argv[1])
    else:
        found = find_default_episode()
        if found is None:
            print(
                "No HDF5 file given and none found under data/test/. Pass a "
                "path explicitly: `uv run python notebooks/00_explore_egodex.py "
                "path/to/episode.hdf5`",
                file=sys.stderr,
            )
            return 2
        episode = found

    if not episode.exists():
        print(f"Episode not found: {episode}", file=sys.stderr)
        return 2

    print(f"Episode: {episode}")
    print(f"Size on disk: {episode.stat().st_size / 1e6:.2f} MB")

    with h5py.File(episode, "r") as f:
        if "camera/intrinsic" in f:
            K = f["camera/intrinsic"][:]
            print("\n--- Camera intrinsics (3x3) ---")
            print(K)

        if "transforms/camera" in f:
            cam = f["transforms/camera"][:]   # (T, 4, 4)
            cam_pos = cam[:, :3, 3]
            print(f"\n--- Camera extrinsics: shape {cam.shape} ---")
            for axis, name in zip(range(3), "xyz"):
                lo, hi = cam_pos[:, axis].min(), cam_pos[:, axis].max()
                print(f"  cam {name}: {lo: .3f} -> {hi: .3f}  (range {hi - lo:.3f} m)")

        # Top-level groups summary
        groups = sorted(f.keys())
        print(f"\nTop-level groups: {groups}")
        if "transforms" in f:
            tf_keys = sorted(f["transforms"].keys())
            print(f"transforms/: {len(tf_keys)} datasets (e.g. {tf_keys[:4]} ... {tf_keys[-4:]})")
        if "confidences" in f:
            cf_keys = sorted(f["confidences"].keys())
            print(f"confidences/: {len(cf_keys)} datasets")

        # File-level attributes (EgoDex stores task/language metadata as attrs)
        if f.attrs:
            print("\n--- File attributes ---")
            for k, v in f.attrs.items():
                if isinstance(v, bytes):
                    v = v.decode("utf-8", errors="replace")
                print(f"  {k}: {v!r}")

        report_hand(f, "left")
        report_hand(f, "right")

    print("\n--- VARIANCE CHECK ---")
    print(
        "Expectation per initial_plan.md §0.3: wrist ranges should be 0.2–0.5 m. "
        "If every axis is < 0.05 m, this is the FIVER v1 collapse signature and "
        "the rest of the pipeline will not save us."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
