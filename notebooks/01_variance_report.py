"""
Stage 0 deliverable — variance report across many EgoDex episodes.

Loops over every `*.hdf5` in a chosen task folder, extracts both wrist
trajectories, and aggregates per-axis range statistics. The point is to confirm
the FIVER v1 collapse signature (sub-cm joint excursions) is absent at the
*dataset* level and to spot which hand is active per episode.

Usage:
    uv run python notebooks/01_variance_report.py [task_dir]

Defaults to `data/test/basic_pick_place/`.

The script writes both human-readable lines to stdout and a CSV summary to
`outputs/variance_report_<task>.csv` for later inspection.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import h5py
import numpy as np


def per_episode_stats(hdf5_path: Path) -> dict:
    """One row of stats: episode + per-hand wrist ranges + active-hand label."""
    with h5py.File(hdf5_path, "r") as f:
        T = f["transforms/leftHand"].shape[0]
        left = f["transforms/leftHand"][:, :3, 3]    # (T, 3)
        right = f["transforms/rightHand"][:, :3, 3]  # (T, 3)
        hand_attr = ""
        for key in ("environment", "description"):
            if key in f.attrs:
                v = f.attrs[key]
                hand_attr = v.decode() if isinstance(v, bytes) else str(v)
                break

    def ranges(arr: np.ndarray) -> tuple[float, float, float]:
        r = arr.max(axis=0) - arr.min(axis=0)
        return float(r[0]), float(r[1]), float(r[2])

    lx, ly, lz = ranges(left)
    rx, ry, rz = ranges(right)
    # Active hand: whichever has the larger total wrist excursion
    left_total = lx + ly + lz
    right_total = rx + ry + rz
    active = "right" if right_total > left_total else "left"

    return {
        "episode": hdf5_path.name,
        "frames": T,
        "left_x": lx, "left_y": ly, "left_z": lz,
        "right_x": rx, "right_y": ry, "right_z": rz,
        "active_hand": active,
        "active_total_m": max(left_total, right_total),
        "env_attr": hand_attr[:80],
    }


def main(argv: list[str]) -> int:
    task_dir = Path(argv[1]) if len(argv) > 1 else Path("data/test/basic_pick_place")
    if not task_dir.is_dir():
        print(f"Not a directory: {task_dir}", file=sys.stderr)
        return 2

    episodes = sorted(task_dir.glob("*.hdf5"))
    if not episodes:
        print(f"No HDF5 files in {task_dir}", file=sys.stderr)
        return 2

    print(f"Task dir: {task_dir}")
    print(f"Episodes: {len(episodes)}")

    rows = [per_episode_stats(p) for p in episodes]

    # Aggregate using whichever hand is active per episode
    active_max_axis = np.array([
        max(r[f"{r['active_hand']}_x"], r[f"{r['active_hand']}_y"], r[f"{r['active_hand']}_z"])
        for r in rows
    ])
    active_total = np.array([r["active_total_m"] for r in rows])
    frames = np.array([r["frames"] for r in rows])

    print("\n--- AGGREGATE (active-hand wrist ranges) ---")
    print(f"frames per episode: median {np.median(frames):.0f}  "
          f"min {frames.min()}  max {frames.max()}  "
          f"total {frames.sum()} ({frames.sum() / 30 / 60:.1f} min @30Hz)")
    print(f"max-axis range:     mean {active_max_axis.mean():.3f} m  "
          f"median {np.median(active_max_axis):.3f}  "
          f"min {active_max_axis.min():.3f}  max {active_max_axis.max():.3f}")
    print(f"sum-of-axes range:  mean {active_total.mean():.3f} m  "
          f"median {np.median(active_total):.3f}  "
          f"min {active_total.min():.3f}  max {active_total.max():.3f}")

    # Variance check vs FIVER v1 collapse threshold (sub-5cm on max axis)
    collapsed = (active_max_axis < 0.05).sum()
    in_target = ((active_max_axis >= 0.2) & (active_max_axis <= 0.5)).sum()
    print(f"\nepisodes with active-hand max-axis range < 0.05 m (collapse signature): "
          f"{collapsed}/{len(rows)} ({100 * collapsed / len(rows):.1f}%)")
    print(f"episodes with active-hand max-axis range in [0.2, 0.5] m (plan target): "
          f"{in_target}/{len(rows)} ({100 * in_target / len(rows):.1f}%)")

    active_split = {h: sum(1 for r in rows if r["active_hand"] == h) for h in ("left", "right")}
    print(f"active-hand split: {active_split}")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"variance_report_{task_dir.name}.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
