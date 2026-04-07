"""
Pre-Stage-1 calibration — answers two open questions from `plan.md`:

  1. ARKit confidence distribution. The §2.1 plan to filter at
     `min_confidence=0.5` looked wrong on episode 0 (no frame above 0.8). What
     does the real distribution look like across many episodes and joints?
     What's a sensible threshold?

  2. Camera-to-table distance. `initial_plan.md` §1.1 hard-codes `table_dist=0.5`
     in the homography formula H = K (R - t n^T / d) K^-1. We can estimate the
     true distance from the wrist's lowest 5th percentile (proxy for table
     surface) projected into the camera frame. If it's stable across episodes
     near 0.5 m, the constant is fine. If it varies, we need per-episode
     estimation.

Usage:
    uv run python notebooks/02_calibrate_open_questions.py [task_dir]

Defaults to `data/test/basic_pick_place/`. Capture stdout to logs/runs/ as
usual. Writes per-episode CSV to outputs/calibration_<task>.csv.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import h5py
import numpy as np


# 25 hand-joint dataset names per side. Same definition as in
# `00_explore_egodex.py::hand_joint_names`, copied here so this script is
# self-contained.
THUMB = ["ThumbKnuckle", "ThumbIntermediateBase", "ThumbIntermediateTip", "ThumbTip"]
NON_THUMB = ["IndexFinger", "MiddleFinger", "RingFinger", "LittleFinger"]
NON_THUMB_SUB = ["Metacarpal", "Knuckle", "IntermediateBase", "IntermediateTip", "Tip"]


def hand_joint_names(side: str) -> list[str]:
    out = [f"{side}Hand"]
    out.extend(f"{side}{j}" for j in THUMB)
    for f in NON_THUMB:
        out.extend(f"{side}{f}{s}" for s in NON_THUMB_SUB)
    return out


HAND_JOINT_NAMES = {"left": hand_joint_names("left"), "right": hand_joint_names("right")}


def collect_episode(path: Path) -> dict | None:
    with h5py.File(path, "r") as f:
        cam = f["transforms/camera"][:]                  # (T, 4, 4) — assume cam-to-world
        right_wrist = f["transforms/rightHand"][:, :3, 3]
        left_wrist = f["transforms/leftHand"][:, :3, 3]
        T = right_wrist.shape[0]

        # Active hand: bigger total wrist excursion
        right_total = float((right_wrist.max(0) - right_wrist.min(0)).sum())
        left_total = float((left_wrist.max(0) - left_wrist.min(0)).sum())
        active = "right" if right_total > left_total else "left"
        active_wrist = right_wrist if active == "right" else left_wrist

        # ----- (1) Confidence distribution: collect every per-joint,
        # per-frame value for the active hand's 25 joints.
        active_conf_values = []
        for jn in HAND_JOINT_NAMES[active]:
            key = f"confidences/{jn}"
            if key in f:
                active_conf_values.append(f[key][:])
        if not active_conf_values:
            return None
        active_conf = np.concatenate(active_conf_values)  # (T*J,)

        # ----- (2) Camera-to-table distance estimate.
        # ARKit uses **y-up** (verified empirically on episode 0: camera y is
        # higher than wrist y, and camera y barely changes within an episode
        # while wrist y varies). The `initial_plan.md` §1.1 snippet using
        # `n = [0,0,1]` and z-up was wrong on both the axis and the magnitude.
        #
        # Estimate the table plane as the 5th-percentile wrist *world-y*
        # (lowest the hand goes — assumes the wrist briefly rests on or near
        # the table during pick/place). Then perpendicular camera-to-table
        # distance is |cam_y_world - table_y|.
        table_y = float(np.percentile(active_wrist[:, 1], 5))
        cam_y_world = cam[:, 1, 3]                        # (T,)
        cam_to_table = np.abs(cam_y_world - table_y)      # (T,)

    return {
        "episode": path.name,
        "frames": T,
        "active": active,
        # Confidence stats over the active hand's 25 joints x T frames
        "conf_n": int(active_conf.size),
        "conf_min": float(active_conf.min()),
        "conf_p05": float(np.percentile(active_conf, 5)),
        "conf_p25": float(np.percentile(active_conf, 25)),
        "conf_p50": float(np.percentile(active_conf, 50)),
        "conf_p75": float(np.percentile(active_conf, 75)),
        "conf_p95": float(np.percentile(active_conf, 95)),
        "conf_max": float(active_conf.max()),
        "conf_mean": float(active_conf.mean()),
        # Table-distance stats
        "table_y_world": table_y,
        "cam_to_table_min": float(cam_to_table.min()),
        "cam_to_table_mean": float(cam_to_table.mean()),
        "cam_to_table_max": float(cam_to_table.max()),
        # Raw confidence array — kept for global histogram
        "_conf_raw": active_conf,
    }


def main(argv: list[str]) -> int:
    task_dir = Path(argv[1]) if len(argv) > 1 else Path("data/test/basic_pick_place")
    episodes = sorted(task_dir.glob("*.hdf5"))
    if not episodes:
        print(f"No episodes in {task_dir}", file=sys.stderr)
        return 2

    print(f"Task dir: {task_dir}")
    print(f"Episodes: {len(episodes)}")

    rows = []
    all_conf = []
    for p in episodes:
        r = collect_episode(p)
        if r is None:
            continue
        all_conf.append(r.pop("_conf_raw"))
        rows.append(r)

    print(f"Episodes processed: {len(rows)}")

    # ----- Question 1: confidence distribution -----
    conf = np.concatenate(all_conf)
    print("\n=== Q1: ARKit confidence distribution (active hand, all 25 joints, all frames) ===")
    print(f"  total samples: {conf.size:,}")
    print(f"  min:  {conf.min():.3f}")
    print(f"  max:  {conf.max():.3f}")
    print(f"  mean: {conf.mean():.3f}   std: {conf.std():.3f}")
    pct_targets = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pcts = np.percentile(conf, pct_targets)
    for p, v in zip(pct_targets, pcts):
        print(f"  p{p:02d}: {v:.3f}")
    print(f"  unique values (rounded to 0.001): {len(np.unique(np.round(conf, 3)))}")

    # Distinct value counts — ARKit confidence in some Apple APIs is a 3-level
    # enum (low/medium/high) coded as 0.0/0.5/1.0 or similar. Check.
    uniq, counts = np.unique(np.round(conf, 4), return_counts=True)
    if len(uniq) <= 20:
        print("\n  full discrete value distribution:")
        for u, c in zip(uniq, counts):
            print(f"    {u:.4f}  ->  {c:>10,}  ({100 * c / conf.size:5.2f}%)")
    else:
        print(f"\n  not a small discrete set ({len(uniq)} unique values); showing top 10 by frequency:")
        order = np.argsort(-counts)[:10]
        for i in order:
            print(f"    {uniq[i]:.4f}  ->  {counts[i]:>10,}  ({100 * counts[i] / conf.size:5.2f}%)")

    print("\n  fraction of samples above various thresholds:")
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        keep = float((conf > thr).mean())
        print(f"    > {thr:.1f}:  {100 * keep:5.1f}%")

    # ----- Question 2: camera-to-table distance distribution -----
    print("\n=== Q2: camera-to-table distance ===")
    cam_means = np.array([r["cam_to_table_mean"] for r in rows])
    cam_mins = np.array([r["cam_to_table_min"] for r in rows])
    cam_maxs = np.array([r["cam_to_table_max"] for r in rows])
    print(f"  per-episode mean: median {np.median(cam_means):.3f} m  "
          f"min {cam_means.min():.3f}  max {cam_means.max():.3f}  "
          f"std {cam_means.std():.3f}")
    print(f"  global min over all frames: {cam_mins.min():.3f} m")
    print(f"  global max over all frames: {cam_maxs.max():.3f} m")
    print(f"  fraction of episodes within 20% of 0.5 m: "
          f"{100 * np.mean((cam_means > 0.4) & (cam_means < 0.6)):.1f}%")
    p10, p50, p90 = np.percentile(cam_means, [10, 50, 90])
    print(f"  per-episode-mean p10/p50/p90: {p10:.3f} / {p50:.3f} / {p90:.3f} m")

    # CSV out
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"calibration_{task_dir.name}.csv"
    keys = [k for k in rows[0].keys() if not k.startswith("_")]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
    print(f"\nwrote {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
