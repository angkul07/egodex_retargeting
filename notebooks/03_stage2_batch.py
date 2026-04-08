"""
03_stage2_batch.py — batch-run Stage 2 action alignment over the entire
`basic_pick_place` test split, and aggregate the existing Stage 1 metrics
JSONs in the same pass so both stages get a single summary CSV + a single
aggregate printout.

Why one script for two stages
-----------------------------
Stage 1 is already done (277 `outputs/stage1/<idx>_metrics.json` files).
Stage 2 needs running. The interesting thing for doc updates is the
per-episode *distribution* of each stage's metrics (FIVER-collapse guard,
pixel-disp reduction, IK error tails), so producing both summaries in one
shot keeps the doc update fast and makes it trivial to join the two CSVs
by episode index later.

Outputs
-------
- `outputs/stage1_summary_basic_pick_place.csv` (aggregated from existing JSONs)
- `outputs/stage2_summary_basic_pick_place.csv` (written as we go)
- `outputs/stage1_aggregate.json` / `outputs/stage2_aggregate.json` — headline
  stats for each stage, the thing we quote in doc.md.

Run
---
    uv run python notebooks/03_stage2_batch.py 2>&1 | \\
        tee logs/runs/$(date +%Y-%m-%d_%H%M%S)_stage2_batch.log
"""

from __future__ import annotations

import csv
import json
import time
import traceback
from pathlib import Path

import numpy as np

from mimicdreamer_egodex.action_alignment import process_episode

# --- Paths -----------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
TASK_DIR = REPO / "data/test/basic_pick_place"
STAGE1_DIR = REPO / "outputs/stage1"
STAGE2_DIR = REPO / "outputs/stage2"
OUT_DIR = REPO / "outputs"
STAGE1_CSV = OUT_DIR / "stage1_summary_basic_pick_place.csv"
STAGE2_CSV = OUT_DIR / "stage2_summary_basic_pick_place.csv"
STAGE1_AGG = OUT_DIR / "stage1_aggregate.json"
STAGE2_AGG = OUT_DIR / "stage2_aggregate.json"


# ---------------------------------------------------------------------------
# Stage 1 aggregation (metrics already on disk)
# ---------------------------------------------------------------------------
STAGE1_FIELDS = [
    "idx",
    "episode",
    "method",
    "n_frames",
    "active_hand",
    "table_y_world",
    "fps",
    "raw_mean_interframe_camera_angle_deg",
    "raw_mean_interframe_pixel_disp",
    "stab_mean_interframe_pixel_disp",
    "pixel_disp_reduction_ratio",
    "homography_rmse_px_median",
]


def aggregate_stage1(episode_paths: list[Path]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    missing: list[str] = []
    for h5 in episode_paths:
        j = STAGE1_DIR / f"{h5.stem}_metrics.json"
        if not j.exists():
            missing.append(h5.stem)
            continue
        m = json.loads(j.read_text())
        row = {k: m.get(k) for k in STAGE1_FIELDS if k != "idx"}
        row["idx"] = int(h5.stem)
        rows.append(row)
    rows.sort(key=lambda r: r["idx"])

    def _arr(key: str) -> np.ndarray:
        vals = [r[key] for r in rows if r[key] is not None and np.isfinite(r[key])]
        return np.array(vals, dtype=float)

    raw_disp = _arr("raw_mean_interframe_pixel_disp")
    stab_disp = _arr("stab_mean_interframe_pixel_disp")
    ratio = _arr("pixel_disp_reduction_ratio")
    hrmse = _arr("homography_rmse_px_median")
    raw_angle = _arr("raw_mean_interframe_camera_angle_deg")
    n_frames = _arr("n_frames")

    methods = {}
    for r in rows:
        methods[r["method"]] = methods.get(r["method"], 0) + 1
    hands = {}
    for r in rows:
        hands[r["active_hand"]] = hands.get(r["active_hand"], 0) + 1

    agg = {
        "n_episodes": len(rows),
        "n_missing": len(missing),
        "missing_idx": missing,
        "methods": methods,
        "active_hand_counts": hands,
        "n_frames": _stats(n_frames),
        "raw_interframe_camera_angle_deg": _stats(raw_angle),
        "raw_interframe_pixel_disp": _stats(raw_disp),
        "stab_interframe_pixel_disp": _stats(stab_disp),
        "pixel_disp_reduction_ratio": _stats(ratio),
        "homography_rmse_px_median": _stats(hrmse),
        "pct_reduction_gt_2x": float((ratio > 2.0).mean() * 100) if len(ratio) else 0.0,
        "pct_reduction_gt_4x": float((ratio > 4.0).mean() * 100) if len(ratio) else 0.0,
        "pct_hrmse_lt_10px": float((hrmse < 10.0).mean() * 100) if len(hrmse) else 0.0,
    }
    return rows, agg


# ---------------------------------------------------------------------------
# Stage 2 batch
# ---------------------------------------------------------------------------
STAGE2_FIELDS = [
    "idx",
    "episode",
    "task",
    "active_hand",
    "n_frames",
    "n_tracked_wrist",
    "table_y_world",
    "h2r_scale",
    "pos_err_m_median",
    "pos_err_m_p95",
    "ori_err_deg_median",
    "ori_err_deg_p95",
    "iters_mean",
    "iters_max",
    "n_joints_range_gt_0_3rad",
    "min_joint_range_rad",
    "gripper_open_frac",
    "gripper_threshold",
    "wall_s",
]


def run_stage2(episode_paths: list[Path]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    failures: list[tuple[str, str]] = []
    t_start = time.time()
    for i, h5 in enumerate(episode_paths):
        t0 = time.time()
        try:
            res = process_episode(h5, STAGE2_DIR)
        except Exception as exc:
            failures.append((h5.stem, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()
            continue
        wall = time.time() - t0
        var = res.variance
        min_range = float(min(var["per_joint_range_rad"]))
        row = {
            "idx": int(h5.stem),
            "episode": res.episode,
            "task": res.task,
            "active_hand": res.active_hand,
            "n_frames": res.n_frames,
            "n_tracked_wrist": res.n_tracked_wrist,
            "table_y_world": res.table_y_world,
            "h2r_scale": res.h2r_scale,
            "pos_err_m_median": res.pos_err_m_median,
            "pos_err_m_p95": res.pos_err_m_p95,
            "ori_err_deg_median": res.ori_err_deg_median,
            "ori_err_deg_p95": res.ori_err_deg_p95,
            "iters_mean": res.iters_mean,
            "iters_max": res.iters_max,
            "n_joints_range_gt_0_3rad": var["n_joints_range_gt_0_3rad"],
            "min_joint_range_rad": min_range,
            "gripper_open_frac": res.gripper_open_frac,
            "gripper_threshold": res.gripper_threshold,
            "wall_s": wall,
        }
        rows.append(row)
        if (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (len(episode_paths) - i - 1)
            print(
                f"  [{i + 1:>3}/{len(episode_paths)}] {h5.stem:<6} "
                f"pos_med={res.pos_err_m_median * 1000:6.2f}mm  "
                f"var={var['n_joints_range_gt_0_3rad']}/{var['n_joints_total']}  "
                f"wall={wall:4.2f}s  elapsed={elapsed:5.1f}s  eta={eta:5.0f}s",
                flush=True,
            )
    rows.sort(key=lambda r: r["idx"])

    def _arr(key: str) -> np.ndarray:
        vals = [r[key] for r in rows if r[key] is not None and np.isfinite(r[key])]
        return np.array(vals, dtype=float)

    pe_med = _arr("pos_err_m_median")
    pe_p95 = _arr("pos_err_m_p95")
    oe_med = _arr("ori_err_deg_median")
    oe_p95 = _arr("ori_err_deg_p95")
    iters_mean = _arr("iters_mean")
    n_over = _arr("n_joints_range_gt_0_3rad")
    min_rng = _arr("min_joint_range_rad")
    n_tracked = _arr("n_tracked_wrist")
    n_frames = _arr("n_frames")
    hands = {}
    for r in rows:
        hands[r["active_hand"]] = hands.get(r["active_hand"], 0) + 1

    agg = {
        "n_episodes": len(rows),
        "n_failures": len(failures),
        "failures": failures,
        "active_hand_counts": hands,
        "wall_s_total": time.time() - t_start,
        "n_frames": _stats(n_frames),
        "pos_err_m_median": _stats(pe_med),
        "pos_err_m_p95": _stats(pe_p95),
        "ori_err_deg_median": _stats(oe_med),
        "ori_err_deg_p95": _stats(oe_p95),
        "iters_mean": _stats(iters_mean),
        "n_joints_range_gt_0_3rad": _stats(n_over),
        "min_joint_range_rad": _stats(min_rng),
        "pct_full_tracked": float((n_tracked == n_frames).mean() * 100) if len(n_tracked) else 0.0,
        # FIVER-collapse guard fractions
        "pct_all_6_joints_above_0_3rad": float((n_over >= 6).mean() * 100) if len(n_over) else 0.0,
        "pct_at_least_5_joints_above_0_3rad": float((n_over >= 5).mean() * 100) if len(n_over) else 0.0,
        # IK-quality fractions
        "pct_pos_med_lt_5mm": float((pe_med < 0.005).mean() * 100) if len(pe_med) else 0.0,
        "pct_pos_med_lt_10mm": float((pe_med < 0.010).mean() * 100) if len(pe_med) else 0.0,
        "pct_pos_p95_lt_20mm": float((pe_p95 < 0.020).mean() * 100) if len(pe_p95) else 0.0,
        "pct_pos_p95_lt_50mm": float((pe_p95 < 0.050).mean() * 100) if len(pe_p95) else 0.0,
    }
    return rows, agg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _fmt_stats(s: dict, scale: float = 1.0, unit: str = "") -> str:
    if s.get("n", 0) == 0:
        return "(empty)"
    return (
        f"p05={s['p05'] * scale:.3f}{unit}  "
        f"p50={s['p50'] * scale:.3f}{unit}  "
        f"mean={s['mean'] * scale:.3f}{unit}  "
        f"p95={s['p95'] * scale:.3f}{unit}  "
        f"max={s['max'] * scale:.3f}{unit}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    episode_paths = sorted(TASK_DIR.glob("*.hdf5"), key=lambda p: int(p.stem))
    print(f"Found {len(episode_paths)} episodes under {TASK_DIR}")
    STAGE2_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Stage 1 aggregation (read-only; fast) -----------------------------
    print("\n== Stage 1: aggregating existing metrics JSONs ==")
    s1_rows, s1_agg = aggregate_stage1(episode_paths)
    _write_csv(STAGE1_CSV, s1_rows, STAGE1_FIELDS)
    STAGE1_AGG.write_text(json.dumps(s1_agg, indent=2))
    print(f"  wrote {STAGE1_CSV} ({len(s1_rows)} rows)")
    print(f"  wrote {STAGE1_AGG}")
    print(f"  n_episodes       : {s1_agg['n_episodes']}  (missing: {s1_agg['n_missing']})")
    print(f"  methods          : {s1_agg['methods']}")
    print(f"  active_hand      : {s1_agg['active_hand_counts']}")
    print(f"  raw cam angle deg: {_fmt_stats(s1_agg['raw_interframe_camera_angle_deg'])}")
    print(f"  raw disp (px)    : {_fmt_stats(s1_agg['raw_interframe_pixel_disp'])}")
    print(f"  stab disp (px)   : {_fmt_stats(s1_agg['stab_interframe_pixel_disp'])}")
    print(f"  reduction ratio  : {_fmt_stats(s1_agg['pixel_disp_reduction_ratio'], unit='x')}")
    print(f"  H-RMSE (px)      : {_fmt_stats(s1_agg['homography_rmse_px_median'])}")
    print(f"  % reduction > 2x : {s1_agg['pct_reduction_gt_2x']:.1f}")
    print(f"  % reduction > 4x : {s1_agg['pct_reduction_gt_4x']:.1f}")
    print(f"  % H-RMSE < 10 px : {s1_agg['pct_hrmse_lt_10px']:.1f}")

    # --- Stage 2 batch (the slow part) -------------------------------------
    print("\n== Stage 2: running IK on every episode ==")
    s2_rows, s2_agg = run_stage2(episode_paths)
    _write_csv(STAGE2_CSV, s2_rows, STAGE2_FIELDS)
    STAGE2_AGG.write_text(json.dumps(s2_agg, indent=2))
    print(f"\n  wrote {STAGE2_CSV} ({len(s2_rows)} rows)")
    print(f"  wrote {STAGE2_AGG}")
    print(f"  n_episodes  : {s2_agg['n_episodes']}  (failures: {s2_agg['n_failures']})")
    print(f"  wall total  : {s2_agg['wall_s_total']:.1f} s")
    print(f"  active_hand : {s2_agg['active_hand_counts']}")
    print(f"  pos_err med : {_fmt_stats(s2_agg['pos_err_m_median'], scale=1000.0, unit='mm')}")
    print(f"  pos_err p95 : {_fmt_stats(s2_agg['pos_err_m_p95'], scale=1000.0, unit='mm')}")
    print(f"  ori_err med : {_fmt_stats(s2_agg['ori_err_deg_median'], unit='°')}")
    print(f"  ori_err p95 : {_fmt_stats(s2_agg['ori_err_deg_p95'], unit='°')}")
    print(f"  iters mean  : {_fmt_stats(s2_agg['iters_mean'])}")
    print(f"  n_joints>0.3: {_fmt_stats(s2_agg['n_joints_range_gt_0_3rad'])}")
    print(f"  min_joint_range_rad: {_fmt_stats(s2_agg['min_joint_range_rad'])}")
    print(f"  % full-tracked wrist           : {s2_agg['pct_full_tracked']:.1f}")
    print(f"  % episodes with 6/6 > 0.3 rad  : {s2_agg['pct_all_6_joints_above_0_3rad']:.1f}")
    print(f"  % episodes with >=5/6 > 0.3 rad: {s2_agg['pct_at_least_5_joints_above_0_3rad']:.1f}")
    print(f"  % pos_err median < 5 mm        : {s2_agg['pct_pos_med_lt_5mm']:.1f}")
    print(f"  % pos_err median < 10 mm       : {s2_agg['pct_pos_med_lt_10mm']:.1f}")
    print(f"  % pos_err p95    < 20 mm       : {s2_agg['pct_pos_p95_lt_20mm']:.1f}")
    print(f"  % pos_err p95    < 50 mm       : {s2_agg['pct_pos_p95_lt_50mm']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
