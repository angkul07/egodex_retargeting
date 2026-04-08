"""
04_stage3_batch.py — batch-run Stage 3 finger retargeting over the entire
`basic_pick_place` test split.

Writes per-episode artifacts under `outputs/stage3/` via `process_episode`,
a flat per-episode summary CSV, and a headline aggregate JSON. Mirrors the
shape of `03_stage2_batch.py` but only handles Stage 3 — Stage 1 and 2
summaries already exist from the previous batch run.

Run
---
    uv run python notebooks/04_stage3_batch.py 2>&1 | \\
        tee logs/runs/$(date +%Y-%m-%d_%H%M%S)_stage3_batch.log
"""

from __future__ import annotations

import csv
import json
import time
import traceback
from pathlib import Path

import numpy as np

from mimicdreamer_egodex.finger_retargeting import (
    INSPIRE_TARGET_FINGER_JOINTS,
    process_episode,
)

REPO = Path(__file__).resolve().parents[1]
TASK_DIR = REPO / "data/test/basic_pick_place"
STAGE3_DIR = REPO / "outputs/stage3"
OUT_DIR = REPO / "outputs"
STAGE3_CSV = OUT_DIR / "stage3_summary_basic_pick_place.csv"
STAGE3_AGG = OUT_DIR / "stage3_aggregate.json"


# One CSV column per target joint range so downstream analysis is trivial.
PER_JOINT_RANGE_FIELDS = [f"range_{n}" for n in INSPIRE_TARGET_FINGER_JOINTS]
STAGE3_FIELDS = [
    "idx",
    "episode",
    "task",
    "active_hand",
    "n_frames",
    "mean_tip_to_wrist_m",
    "retarget_wall_s",
    "retarget_ms_per_frame",
    "n_target_joints_range_gt_0_1rad",
    "min_target_range_rad",
    *PER_JOINT_RANGE_FIELDS,
    "wall_s",
]


def _stats(a: np.ndarray) -> dict:
    if len(a) == 0:
        return {"n": 0}
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "p05": float(np.percentile(a, 5)),
        "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def _fmt(s: dict, scale: float = 1.0, unit: str = "") -> str:
    if s.get("n", 0) == 0:
        return "(empty)"
    return (
        f"p05={s['p05'] * scale:.3f}{unit}  "
        f"p50={s['p50'] * scale:.3f}{unit}  "
        f"mean={s['mean'] * scale:.3f}{unit}  "
        f"p95={s['p95'] * scale:.3f}{unit}  "
        f"max={s['max'] * scale:.3f}{unit}"
    )


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    episode_paths = sorted(TASK_DIR.glob("*.hdf5"), key=lambda p: int(p.stem))
    print(f"Found {len(episode_paths)} episodes under {TASK_DIR}")
    STAGE3_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    failures: list[tuple[str, str]] = []
    t_start = time.time()

    for i, h5 in enumerate(episode_paths):
        t0 = time.time()
        try:
            res = process_episode(h5, STAGE3_DIR)
        except Exception as exc:
            failures.append((h5.stem, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()
            continue
        wall = time.time() - t0

        var = res.variance
        per_joint = dict(zip(PER_JOINT_RANGE_FIELDS, var["target_joint_range_rad"]))
        row = {
            "idx": int(h5.stem),
            "episode": res.episode,
            "task": res.task,
            "active_hand": res.active_hand,
            "n_frames": res.n_frames,
            "mean_tip_to_wrist_m": res.mean_tip_to_wrist_m,
            "retarget_wall_s": res.retarget_wall_s,
            "retarget_ms_per_frame": res.retarget_ms_per_frame,
            "n_target_joints_range_gt_0_1rad": var["n_target_joints_range_gt_0_1rad"],
            "min_target_range_rad": float(min(var["target_joint_range_rad"])),
            **per_joint,
            "wall_s": wall,
        }
        rows.append(row)
        if (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (len(episode_paths) - i - 1)
            print(
                f"  [{i + 1:>3}/{len(episode_paths)}] {h5.stem:<6} "
                f"n_frames={res.n_frames:>4} "
                f"ms/frame={res.retarget_ms_per_frame:5.2f} "
                f"var={var['n_target_joints_range_gt_0_1rad']}/6 "
                f"min_range={row['min_target_range_rad']:.3f}  "
                f"elapsed={elapsed:5.1f}s  eta={eta:5.0f}s",
                flush=True,
            )

    rows.sort(key=lambda r: r["idx"])
    _write_csv(STAGE3_CSV, rows, STAGE3_FIELDS)

    def _arr(key: str) -> np.ndarray:
        vals = [r[key] for r in rows if r[key] is not None and np.isfinite(r[key])]
        return np.array(vals, dtype=float)

    ms_per_frame = _arr("retarget_ms_per_frame")
    n_frames_arr = _arr("n_frames")
    mean_spread = _arr("mean_tip_to_wrist_m")
    n_over = _arr("n_target_joints_range_gt_0_1rad")
    min_range = _arr("min_target_range_rad")

    per_joint_stats = {
        name: _stats(_arr(f"range_{name}")) for name in INSPIRE_TARGET_FINGER_JOINTS
    }

    hands: dict[str, int] = {}
    for r in rows:
        hands[r["active_hand"]] = hands.get(r["active_hand"], 0) + 1

    agg = {
        "n_episodes": len(rows),
        "n_failures": len(failures),
        "failures": failures,
        "wall_s_total": time.time() - t_start,
        "active_hand_counts": hands,
        "n_frames": _stats(n_frames_arr),
        "retarget_ms_per_frame": _stats(ms_per_frame),
        "mean_tip_to_wrist_m": _stats(mean_spread),
        "n_target_joints_range_gt_0_1rad": _stats(n_over),
        "min_target_range_rad": _stats(min_range),
        "per_target_joint_range_rad": per_joint_stats,
        "pct_all_6_target_joints_above_0_1rad":
            float((n_over >= 6).mean() * 100) if len(n_over) else 0.0,
        "pct_min_target_range_above_0_2rad":
            float((min_range > 0.2).mean() * 100) if len(min_range) else 0.0,
        "pct_min_target_range_above_0_3rad":
            float((min_range > 0.3).mean() * 100) if len(min_range) else 0.0,
    }
    STAGE3_AGG.write_text(json.dumps(agg, indent=2))

    print(f"\n  wrote {STAGE3_CSV} ({len(rows)} rows)")
    print(f"  wrote {STAGE3_AGG}")
    print(f"  n_episodes   : {agg['n_episodes']}  (failures: {agg['n_failures']})")
    print(f"  wall total   : {agg['wall_s_total']:.1f} s")
    print(f"  active_hand  : {agg['active_hand_counts']}")
    print(f"  n_frames     : {_fmt(agg['n_frames'])}")
    print(f"  ms/frame     : {_fmt(agg['retarget_ms_per_frame'], unit=' ms')}")
    print(f"  tip→wrist (m): {_fmt(agg['mean_tip_to_wrist_m'], unit=' m')}")
    print(f"  n>0.1 rad    : {_fmt(agg['n_target_joints_range_gt_0_1rad'])}")
    print(f"  min_range    : {_fmt(agg['min_target_range_rad'], unit=' rad')}")
    print("  per-target-joint range (rad) across episodes:")
    for name, s in per_joint_stats.items():
        print(f"    {name:34s}  {_fmt(s, unit=' rad')}")
    print(
        f"  % episodes with 6/6 target joints > 0.1 rad: "
        f"{agg['pct_all_6_target_joints_above_0_1rad']:.1f}"
    )
    print(
        f"  % episodes with min target range > 0.2 rad : "
        f"{agg['pct_min_target_range_above_0_2rad']:.1f}"
    )
    print(
        f"  % episodes with min target range > 0.3 rad : "
        f"{agg['pct_min_target_range_above_0_3rad']:.1f}"
    )
    if failures:
        print(f"\n  Failures:")
        for idx, msg in failures[:10]:
            print(f"    {idx}: {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
