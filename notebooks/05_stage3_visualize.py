"""
05_stage3_visualize.py — static visualizations of Stage 3 finger retargeting.

Produces four plots per episode, answering four different questions:

1. `<idx>_joint_angles.png`     : What signal does Stage 4 actually get?
2. `<idx>_fingertips_3d.png`    : How closely does the Inspire hand follow
                                  the human hand in 3D space?
3. `<idx>_retarget_error.png`   : Where (in time) does the retargeting
                                  struggle, per finger?
4. `<idx>_overview.png`         : Compact 2x2 summary (joint angles, mean
                                  fingertip spread, per-finger error,
                                  per-joint range bar chart).

Relies on the per-side `SeqRetargeting` cache in `finger_retargeting.py`
to compute forward kinematics on the retargeted `q_full` without having
to load the URDF into MuJoCo (which fails on relative mesh paths). All
plots are matplotlib-only → works headless on RunPod.

Run
---
    uv run python notebooks/05_stage3_visualize.py --episodes 0 1 2
        --out-dir outputs/stage3/viz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

from mimicdreamer_egodex.finger_retargeting import (
    INSPIRE_TARGET_FINGER_JOINTS,
    get_inspire_retargeter,
)

# --- Shared colors: one per finger (thumb..pinky) -------------------------
FINGER_COLORS = {
    "thumb": "#d62728",   # red
    "index": "#ff7f0e",   # orange
    "middle": "#2ca02c",  # green
    "ring": "#1f77b4",    # blue
    "pinky": "#9467bd",   # purple
}
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")  # matches tips_rel
TIP_LINK_NAMES = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")


# ---------------------------------------------------------------------------
# Forward kinematics — robot fingertips in the hand-root frame per frame
# ---------------------------------------------------------------------------
def robot_fingertip_trajectory(q_full: np.ndarray, side: str) -> np.ndarray:
    """Run pinocchio FK on each q_full row and return the 5 Inspire
    fingertip positions in the same (dummy-free-joint-absorbed) frame the
    retargeter used. Shape: (T, 5, 3)."""
    retargeter = get_inspire_retargeter(side)
    robot = retargeter.optimizer.robot
    link_names = list(robot.link_names)
    tip_indices = [link_names.index(name) for name in TIP_LINK_NAMES]

    T = len(q_full)
    out = np.zeros((T, 5, 3), dtype=np.float32)
    for t in range(T):
        robot.compute_forward_kinematics(q_full[t].astype(np.float32))
        for j, idx in enumerate(tip_indices):
            out[t, j] = robot.get_link_pose(idx)[:3, 3]
    return out


# ---------------------------------------------------------------------------
# Plot 1 — Joint-angle time series
# ---------------------------------------------------------------------------
def plot_joint_angles(
    q_finger: np.ndarray, joint_names: list[str], title: str, out_path: Path
) -> None:
    T = len(q_finger)
    t = np.arange(T) / 30.0  # seconds @ 30 Hz
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    axes = axes.flatten()
    for i, (ax, name) in enumerate(zip(axes, joint_names)):
        ax.plot(t, q_finger[:, i], lw=1.5)
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylabel("angle (rad)")
        if i >= 3:
            ax.set_xlabel("time (s)")
        rng = q_finger[:, i].max() - q_finger[:, i].min()
        ax.text(
            0.98,
            0.03,
            f"range = {rng:.3f} rad ({np.degrees(rng):.1f}°)",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7),
        )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2 — Human vs robot fingertip trajectories in 3D
# ---------------------------------------------------------------------------
def plot_fingertips_3d(
    human_tips: np.ndarray,  # (T, 5, 3)
    robot_tips: np.ndarray,  # (T, 5, 3)
    title: str,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(11, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    # Wrist marker at origin (both frames are wrist-relative)
    for ax in (ax1, ax2):
        ax.scatter([0], [0], [0], color="black", s=60, marker="o", label="wrist", zorder=5)

    for j, finger in enumerate(FINGER_ORDER):
        c = FINGER_COLORS[finger]
        ax1.plot(
            human_tips[:, j, 0],
            human_tips[:, j, 1],
            human_tips[:, j, 2],
            color=c,
            lw=1.5,
            label=finger,
        )
        ax2.plot(
            robot_tips[:, j, 0],
            robot_tips[:, j, 1],
            robot_tips[:, j, 2],
            color=c,
            lw=1.5,
            label=finger,
        )
        # Start + end markers
        ax1.scatter(*human_tips[0, j], color=c, s=30, marker="^")  # start
        ax1.scatter(*human_tips[-1, j], color=c, s=30, marker="s")  # end
        ax2.scatter(*robot_tips[0, j], color=c, s=30, marker="^")
        ax2.scatter(*robot_tips[-1, j], color=c, s=30, marker="s")

    ax1.set_title("Input: EgoDex fingertips (wrist-relative)")
    ax2.set_title("Output: Inspire fingertips (via pinocchio FK on q_full)")
    for ax in (ax1, ax2):
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.legend(loc="upper left", fontsize=7)
    # Shared axis limits so visual comparison is fair
    all_pts = np.concatenate([human_tips.reshape(-1, 3), robot_tips.reshape(-1, 3)], axis=0)
    pad = 0.02
    for ax in (ax1, ax2):
        ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
        ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
        ax.set_zlim(all_pts[:, 2].min() - pad, all_pts[:, 2].max() + pad)
    fig.suptitle(title + "   (▲ = start, ■ = end)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3 — Per-frame retargeting error (per finger)
# ---------------------------------------------------------------------------
def plot_retarget_error(
    human_tips: np.ndarray,  # (T, 5, 3)
    robot_tips: np.ndarray,  # (T, 5, 3)
    title: str,
    out_path: Path,
) -> None:
    err = np.linalg.norm(human_tips - robot_tips, axis=-1) * 1000.0  # (T, 5) mm
    T = len(err)
    t = np.arange(T) / 30.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]})

    for j, finger in enumerate(FINGER_ORDER):
        ax1.plot(t, err[:, j], color=FINGER_COLORS[finger], lw=1.2, label=finger)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("‖human tip − robot tip‖ (mm)")
    ax1.set_title("Per-frame retargeting error")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)

    for j, finger in enumerate(FINGER_ORDER):
        ax2.hist(
            err[:, j],
            bins=30,
            color=FINGER_COLORS[finger],
            alpha=0.45,
            label=f"{finger} (med {np.median(err[:, j]):.1f})",
        )
    ax2.set_xlabel("error (mm)")
    ax2.set_ylabel("frames")
    ax2.set_title("Error histogram")
    ax2.legend(loc="upper right", fontsize=7)
    ax2.grid(alpha=0.3)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    # Return summary for printing
    return {
        "median_mm": [float(np.median(err[:, j])) for j in range(5)],
        "p95_mm": [float(np.percentile(err[:, j], 95)) for j in range(5)],
        "max_mm": [float(err[:, j].max()) for j in range(5)],
    }


# ---------------------------------------------------------------------------
# Plot 4 — 2x2 overview
# ---------------------------------------------------------------------------
def plot_overview(
    q_finger: np.ndarray,
    joint_names: list[str],
    human_tips: np.ndarray,
    robot_tips: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    T = len(q_finger)
    t = np.arange(T) / 30.0
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (0,0) joint angles
    ax = axes[0, 0]
    for i, name in enumerate(joint_names):
        ax.plot(t, q_finger[:, i], lw=1.2, label=name.replace("_joint", "").replace("_proximal", ""))
    ax.set_title("Inspire target joint angles (the q_finger action vector)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("angle (rad)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right", ncol=2)

    # (0,1) mean fingertip-to-wrist distance — openness proxy
    ax = axes[0, 1]
    openness_human = np.linalg.norm(human_tips, axis=-1).mean(axis=1) * 1000
    openness_robot = np.linalg.norm(robot_tips, axis=-1).mean(axis=1) * 1000
    ax.plot(t, openness_human, color="black", lw=1.5, label="human (input)")
    ax.plot(t, openness_robot, color="tab:red", lw=1.2, ls="--", label="Inspire (FK)")
    ax.set_title("Mean fingertip-to-wrist distance (hand openness proxy)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("distance (mm)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (1,0) per-finger retarget error
    ax = axes[1, 0]
    err = np.linalg.norm(human_tips - robot_tips, axis=-1) * 1000.0
    for j, finger in enumerate(FINGER_ORDER):
        ax.plot(t, err[:, j], color=FINGER_COLORS[finger], lw=1.2, label=finger)
    ax.set_title("Retargeting error: ‖human tip − robot tip‖")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("error (mm)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (1,1) per-joint range bar chart
    ax = axes[1, 1]
    ranges_deg = np.degrees(q_finger.max(0) - q_finger.min(0))
    short_names = [
        n.replace("_joint", "").replace("_proximal", "").replace("thumb_", "th_")
        for n in joint_names
    ]
    x_pos = np.arange(len(short_names))
    bars = ax.bar(x_pos, ranges_deg, color="steelblue")
    for bar, val in zip(bars, ranges_deg):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.0f}°",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title("Per-joint range over episode")
    ax.set_ylabel("range (deg)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_names, rotation=20, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-episode driver
# ---------------------------------------------------------------------------
def visualize_episode(idx: int, stage3_dir: Path, out_dir: Path) -> None:
    npz_path = stage3_dir / f"{idx}_fingers.npz"
    meta_path = stage3_dir / f"{idx}_metrics.json"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing {npz_path} — run Stage 3 batch first")

    d = np.load(npz_path, allow_pickle=True)
    q_finger = d["q_finger"]
    q_full = d["q_full"]
    tips_rel = d["tips_rel"]
    joint_names = list(d["joint_names_target"])
    active = str(d["active_hand"])

    import json

    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    task = meta.get("task", "unknown")
    desc = meta.get("description", "")

    # Per-side FK (uses the cached retargeter)
    robot_tips = robot_fingertip_trajectory(q_full, active)

    title_base = f"Episode {idx} — {task}, {active} hand\n“{desc}”"

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_joint_angles(
        q_finger, joint_names, f"{title_base}\nInspire target joint angles",
        out_dir / f"{idx}_joint_angles.png",
    )
    plot_fingertips_3d(
        tips_rel, robot_tips,
        f"{title_base}\nFingertip trajectories (wrist-relative)",
        out_dir / f"{idx}_fingertips_3d.png",
    )
    err_summary = plot_retarget_error(
        tips_rel, robot_tips,
        f"{title_base}\nRetargeting error per finger",
        out_dir / f"{idx}_retarget_error.png",
    )
    plot_overview(
        q_finger, joint_names, tips_rel, robot_tips,
        f"{title_base}",
        out_dir / f"{idx}_overview.png",
    )

    # Print a short summary
    print(f"\n=== episode {idx} ({task}, {active}) ===")
    print(f"  n_frames: {len(q_finger)}")
    print(f"  retarget error per finger (median / p95 / max in mm):")
    for j, finger in enumerate(FINGER_ORDER):
        print(
            f"    {finger:7s}  med={err_summary['median_mm'][j]:6.2f}  "
            f"p95={err_summary['p95_mm'][j]:6.2f}  "
            f"max={err_summary['max_mm'][j]:6.2f}"
        )
    print(f"  per-joint range (deg):")
    for name, rng in zip(joint_names, q_finger.max(0) - q_finger.min(0)):
        print(f"    {name:35s}  {np.degrees(rng):6.2f}")
    print(f"  wrote 4 plots to {out_dir}/{idx}_*.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 3 visualizations")
    ap.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Episode indices (e.g. --episodes 0 1 50 100)",
    )
    ap.add_argument("--stage3-dir", type=Path, default=Path("outputs/stage3"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/stage3/viz"))
    args = ap.parse_args(argv)

    for idx in args.episodes:
        visualize_episode(idx, args.stage3_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
