"""
Stage 2 — Action alignment (wrist trajectory → UR5e joint angles).

Takes an EgoDex HDF5 episode, extracts the active-hand wrist trajectory
(plus fingertips for a gripper proxy), aligns the human workspace to the
UR5e robot base frame (H2R), and solves a smoothed DLS-style IK per frame
using `mink` to produce a 6-DOF joint angle trajectory + a binary gripper
signal. Writes an npz + a metrics JSON per episode.

Target robot
------------
UR5e — 6-DOF serial arm (robot_descriptions `ur5e_mj_description`).
The dexterous hand (Stage 3) is decoupled: the arm IK targets the tool-flange
`attachment_site`, and Stage 3's finger retargeting runs independently on the
same episodes. Final action vector (per `initial_plan.md` §3.3):
    a_t = [q_arm (6), gripper (1), q_fingers (N)].

Calibrated conventions (from doc.md §3, §5 and plan.md R-001/R-002/R-003)
------------------------------------------------------------------------
- ARKit world is **y-up**. Robot base is **z-up**. Rotation maps world y → robot z.
- `transforms/<jointName>` are already in world frame — do NOT invert the
  extrinsics (the `initial_plan.md` §2.1 snippet that does `inv(extrinsics) @
  wrist_poses` is wrong for the real schema; see plan.md D-004).
- Confidence floor is **0.10** (hard reject), not 0.5 (plan.md R-002).
- Table plane `table_y = 5th-percentile world-y of active-hand wrist` (R-001).
- Scale defaults to **1.0**. The §2.2 `0.6/0.5` scale hack is not justified
  by the data — human reach vs. UR5e reach are close enough; the §2.2
  approach also fails at the workspace center step. Expose `--scale` as a
  flag for experiments only.

CLI
---
    uv run python -m mimicdreamer_egodex.action_alignment \\
        data/test/basic_pick_place/0.hdf5 \\
        --out-dir outputs/stage2
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import h5py
import mink
import mujoco  # noqa: F401  (imported for side effects; used by mink)
import numpy as np
from robot_descriptions.loaders.mujoco import load_robot_description
from scipy.ndimage import median_filter

# --- Calibrated constants from doc.md §5 (R-002) ---------------------------
CONF_TRACKED = 0.10  # hard reject below this
CONF_HIGH = 0.50     # "high quality" — soft, used as a weight elsewhere

# World up-axis under ARKit y-up.
N_WORLD_UP = np.array([0.0, 1.0, 0.0])

# Rotation: world (y-up) → robot base (z-up).
#   world x → robot x
#   world y → robot z      (up-to-up, the important constraint)
#   world z → robot -y
# Yaw around robot z is *arbitrary* at this stage — ARKit world yaw is not
# fixed across episodes — and is absorbed into `workspace_center_xy`
# centering below. `det(R) = +1`, `R R^T = I`.
R_W2R = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ]
)

# --- Target robot (UR5e) ---------------------------------------------------
ROBOT_DESCRIPTION = "ur5e_mj_description"
EE_FRAME_NAME = "attachment_site"  # UR5e tool flange — mount point for Stage 3 hand
EE_FRAME_TYPE = "site"
HOME_KEYFRAME = "home"

# Where to place the mean wrist position in the robot base frame (meters).
# UR5e reach is ~0.85 m; 0.5 m in front of base, centered on the midline,
# leaves headroom on both sides of the workspace.
WORKSPACE_CENTER_XY = (0.5, 0.0)

# --- EgoDex joint-name tables (doc.md §2.1) --------------------------------
HAND_WRIST = {"left": "leftHand", "right": "rightHand"}
FINGERTIPS = {
    "left": [
        "leftThumbTip",
        "leftIndexFingerTip",
        "leftMiddleFingerTip",
        "leftRingFingerTip",
        "leftLittleFingerTip",
    ],
    "right": [
        "rightThumbTip",
        "rightIndexFingerTip",
        "rightMiddleFingerTip",
        "rightRingFingerTip",
        "rightLittleFingerTip",
    ],
}


# ---------------------------------------------------------------------------
# HDF5 extraction
# ---------------------------------------------------------------------------
def pick_active_hand(rwrist_xyz: np.ndarray, lwrist_xyz: np.ndarray) -> str:
    r_total = float((rwrist_xyz.max(0) - rwrist_xyz.min(0)).sum())
    l_total = float((lwrist_xyz.max(0) - lwrist_xyz.min(0)).sum())
    return "right" if r_total >= l_total else "left"


def load_episode(hdf5_path: Path) -> dict:
    """Pull everything Stage 2 needs out of one HDF5 file."""
    with h5py.File(hdf5_path, "r") as f:
        rwrist_p = f["transforms/rightHand"][:, :3, 3]
        lwrist_p = f["transforms/leftHand"][:, :3, 3]
        active = pick_active_hand(rwrist_p, lwrist_p)

        wrist_key = HAND_WRIST[active]
        wrist_poses = f[f"transforms/{wrist_key}"][:].astype(np.float64)  # (T,4,4)
        wrist_conf = f[f"confidences/{wrist_key}"][:].astype(np.float64)

        tips = np.stack(
            [f[f"transforms/{n}"][:, :3, 3] for n in FINGERTIPS[active]],
            axis=1,
        ).astype(np.float64)  # (T, 5, 3) in world frame
        tips_conf = np.stack(
            [f[f"confidences/{n}"][:] for n in FINGERTIPS[active]],
            axis=1,
        ).astype(np.float64)  # (T, 5)

        task = _attr(f, "task", "unknown")
        env = _attr(f, "environment", "")
        desc = _attr(f, "description", "")

    return {
        "hdf5_path": str(hdf5_path),
        "active_hand": active,
        "wrist_poses_world": wrist_poses,
        "wrist_conf": wrist_conf,
        "fingertips_world": tips,
        "fingertip_conf": tips_conf,
        "task": task,
        "environment": env,
        "description": desc,
    }


def _attr(f: h5py.File, key: str, default):
    if key not in f.attrs:
        return default
    v = f.attrs[key]
    return v.decode() if isinstance(v, bytes) else str(v)


def gap_fill_wrist(
    wrist_poses: np.ndarray,
    wrist_conf: np.ndarray,
    conf_floor: float = CONF_TRACKED,
) -> tuple[np.ndarray, int]:
    """Forward/back-fill `wrist_poses` where confidence is below the floor.

    We do not interpolate on SE(3) — the test split is well-tracked enough
    (doc.md §5) that hold-last is fine; only a handful of frames per episode
    fire this path, and the IK smoothness term hides the step.
    Returns (filled_poses, n_good_frames).
    """
    T = len(wrist_poses)
    good = wrist_conf > conf_floor
    n_good = int(good.sum())
    if n_good == 0:
        return wrist_poses.copy(), 0
    if n_good == T:
        return wrist_poses.copy(), T

    out = wrist_poses.copy()
    first_good = int(np.argmax(good))
    # back-fill leading bad frames with the first good frame
    for t in range(first_good):
        out[t] = wrist_poses[first_good]
    # forward-fill the rest
    last = wrist_poses[first_good]
    for t in range(first_good, T):
        if good[t]:
            last = wrist_poses[t]
        else:
            out[t] = last
    return out, n_good


# ---------------------------------------------------------------------------
# Human-to-robot frame alignment (H2R)
# ---------------------------------------------------------------------------
def estimate_h2r_transform(
    wrist_poses_world: np.ndarray,
    wrist_conf: np.ndarray,
    workspace_center_xy: tuple[float, float] = WORKSPACE_CENTER_XY,
    scale: float = 1.0,
    table_y_pctile: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Build (R_H2R, t_H2R, scale, table_y).

    Applied as: p_robot = scale * (R_H2R @ p_world) + t_H2R.

    - R_H2R is fixed (y-up → z-up; yaw is absorbed into centering).
    - table_y = 5th-pctl wrist world-y over confidence-gated frames.
    - t_H2R.z sends the table to robot z=0.
    - t_H2R.xy centers the mean wrist horizontal position on workspace_center_xy.
    """
    good = wrist_conf > CONF_TRACKED
    if not good.any():
        # fall back to using all frames (caller will have already logged the
        # low confidence); prevents division by zero downstream
        good = np.ones_like(wrist_conf, dtype=bool)

    ys = wrist_poses_world[good, 1, 3]
    table_y = float(np.percentile(ys, table_y_pctile))

    R = R_W2R
    p_world = wrist_poses_world[good, :3, 3]
    p_rot = p_world @ R.T  # equivalent to (R @ p_world.T).T, pre-offset robot frame
    mean_xy = p_rot[:, :2].mean(axis=0)

    tx = float(workspace_center_xy[0] - scale * mean_xy[0])
    ty = float(workspace_center_xy[1] - scale * mean_xy[1])
    tz = float(-scale * table_y)  # after rotation, world-y becomes robot-z
    t = np.array([tx, ty, tz], dtype=np.float64)
    return R, t, float(scale), table_y


def apply_h2r(
    wrist_poses_world: np.ndarray,
    R_H2R: np.ndarray,
    t_H2R: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Map (T,4,4) world-frame wrist SE(3) poses into the robot base frame."""
    T = len(wrist_poses_world)
    out = np.tile(np.eye(4), (T, 1, 1))
    Rw = wrist_poses_world[:, :3, :3]          # (T,3,3)
    pw = wrist_poses_world[:, :3, 3]           # (T,3)
    out[:, :3, :3] = R_H2R @ Rw                 # broadcasted matmul
    out[:, :3, 3] = scale * (pw @ R_H2R.T) + t_H2R
    return out


# ---------------------------------------------------------------------------
# Robot & IK
# ---------------------------------------------------------------------------
_CACHED_UR5E_MODEL = None  # MjModel is read-only after load; safe to share across episodes.


def load_ur5e():
    """Return (model, fresh_configuration_at_home). The MJCF is loaded once
    per process and cached — the per-episode cost is just building a fresh
    `mink.Configuration` (which wraps a mutable mjData)."""
    global _CACHED_UR5E_MODEL
    if _CACHED_UR5E_MODEL is None:
        _CACHED_UR5E_MODEL = load_robot_description(ROBOT_DESCRIPTION)
    model = _CACHED_UR5E_MODEL
    config = mink.Configuration(model)
    config.update_from_keyframe(HOME_KEYFRAME)
    return model, config


def solve_ik_trajectory(
    wrist_targets_robot: np.ndarray,
    dt: float = 1.0 / 30.0,
    position_cost: float = 1.0,
    orientation_cost: float = 0.3,
    lambda_smooth: float = 0.1,
    pos_tol: float = 2e-3,          # 2 mm
    seed_iters: int = 80,
    step_iters: int = 10,
    solver: str = "quadprog",
    ik_damping: float = 1e-3,
) -> dict:
    """Per-frame IK for the UR5e.

    Uses a `FrameTask` on the tool flange + a `PostureTask` whose target is
    reset to the previous frame's solution at the start of each new frame.
    Within a frame we iterate up to `step_iters` (or `seed_iters` on frame 0)
    Newton steps until `||pos_err|| < pos_tol`.

    The PostureTask-anchored-at-previous-q formulation is the smoothness
    penalty the plan (§2.3) actually wants: *within* a frame's iterations the
    anchor is fixed, so as q drifts toward the frame target the posture
    residual grows and pulls back. That trade-off is what prevents the
    per-frame jumps that killed FIVER v1 without smoothness.
    """
    model, config = load_ur5e()
    q_home = config.q.copy()

    frame_task = mink.FrameTask(
        frame_name=EE_FRAME_NAME,
        frame_type=EE_FRAME_TYPE,
        position_cost=position_cost,
        orientation_cost=orientation_cost,
    )
    posture_task = mink.PostureTask(model=model, cost=lambda_smooth)
    posture_task.set_target(q_home)

    T = len(wrist_targets_robot)
    q_out = np.zeros((T, model.nq))
    pos_err_out = np.zeros(T)
    ori_err_out = np.zeros(T)
    iters_used = np.zeros(T, dtype=int)

    prev_q = q_home.copy()
    for t in range(T):
        R_t = wrist_targets_robot[t, :3, :3]
        p_t = wrist_targets_robot[t, :3, 3]
        target_se3 = mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3.from_matrix(R_t),
            translation=p_t,
        )
        frame_task.set_target(target_se3)
        posture_task.set_target(prev_q)  # smoothness anchor

        max_iters = seed_iters if t == 0 else step_iters
        pe = float("inf")
        oe = float("inf")
        for it in range(max_iters):
            vel = mink.solve_ik(
                configuration=config,
                tasks=[frame_task, posture_task],
                dt=dt,
                solver=solver,
                damping=ik_damping,
            )
            config.integrate_inplace(vel, dt)
            err = frame_task.compute_error(config)   # (6,) = [tx,ty,tz,rx,ry,rz]
            pe = float(np.linalg.norm(err[:3]))
            oe = float(np.linalg.norm(err[3:]))
            iters_used[t] = it + 1
            if pe < pos_tol:
                break

        q_out[t] = config.q.copy()
        pos_err_out[t] = pe
        ori_err_out[t] = float(np.degrees(oe))
        prev_q = config.q.copy()

    return {
        "q": q_out,
        "pos_err_m": pos_err_out,
        "ori_err_deg": ori_err_out,
        "iters_per_frame": iters_used,
        "q_home": q_home,
        "joint_names": [model.joint(i).name for i in range(model.njnt)],
    }


# ---------------------------------------------------------------------------
# Gripper signal
# ---------------------------------------------------------------------------
def compute_gripper_signal(
    wrist_poses_world: np.ndarray,
    tips_world: np.ndarray,          # (T, 5, 3)
    window: int = 5,
    threshold: float | None = None,  # None → per-episode median
) -> tuple[np.ndarray, np.ndarray, float]:
    """Binary gripper signal from mean fingertip-to-wrist distance.

    Openness = mean over 5 fingertips of ||tip_i - wrist|| (world frame).
    Threshold defaults to the per-episode *median* of openness (so ~half the
    frames are "open"), then the binary signal is median-filtered. Calibrate
    per task later if needed. Returns (binary, openness, threshold_used).
    """
    wrist_p = wrist_poses_world[:, :3, 3]        # (T, 3)
    diffs = tips_world - wrist_p[:, None, :]     # (T, 5, 3)
    dists = np.linalg.norm(diffs, axis=-1)       # (T, 5)
    openness = dists.mean(axis=1)                # (T,)

    thr = float(np.median(openness)) if threshold is None else float(threshold)
    binary = (openness > thr).astype(np.float32)
    if window and window > 1:
        binary = median_filter(binary, size=window, mode="nearest")
    return binary, openness, thr


# ---------------------------------------------------------------------------
# Variance / sanity reporting
# ---------------------------------------------------------------------------
def variance_report(q: np.ndarray, joint_names: list[str]) -> dict:
    """FIVER-collapse guard: per-joint range + count of primary joints with
    range > 0.3 rad (CLAUDE.md explicitly calls this out)."""
    ranges = (q.max(axis=0) - q.min(axis=0)).astype(float)
    stds = q.std(axis=0).astype(float)
    n_over = int((ranges > 0.3).sum())
    return {
        "joint_names": joint_names,
        "per_joint_range_rad": ranges.tolist(),
        "per_joint_std_rad": stds.tolist(),
        "n_joints_range_gt_0_3rad": n_over,
        "n_joints_total": int(len(ranges)),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
@dataclass
class AlignResult:
    episode: str
    task: str
    description: str
    environment: str
    active_hand: str
    n_frames: int
    n_tracked_wrist: int
    table_y_world: float
    h2r_rotation: list = field(default_factory=list)      # (3,3) row-major
    h2r_translation: list = field(default_factory=list)   # (3,)
    h2r_scale: float = 1.0
    ik_solver: str = "quadprog"
    ik_position_cost: float = 1.0
    ik_orientation_cost: float = 0.3
    ik_lambda_smooth: float = 0.1
    pos_err_m_median: float = 0.0
    pos_err_m_p95: float = 0.0
    ori_err_deg_median: float = 0.0
    ori_err_deg_p95: float = 0.0
    iters_mean: float = 0.0
    iters_max: int = 0
    variance: dict = field(default_factory=dict)
    gripper_threshold: float = 0.0
    gripper_open_frac: float = 0.0
    out_npz: str = ""
    notes: str = ""


def process_episode(
    hdf5_path: Path,
    out_dir: Path,
    scale: float = 1.0,
    workspace_center_xy: tuple[float, float] = WORKSPACE_CENTER_XY,
    lambda_smooth: float = 0.1,
    position_cost: float = 1.0,
    orientation_cost: float = 0.3,
    gripper_threshold: float | None = None,
) -> AlignResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    ep = load_episode(hdf5_path)
    wrist_poses = ep["wrist_poses_world"]
    wrist_conf = ep["wrist_conf"]
    T = len(wrist_poses)

    # Gap-fill sub-threshold wrist frames (rare on the test split).
    filled, n_tracked = gap_fill_wrist(wrist_poses, wrist_conf)
    if n_tracked < T:
        notes.append(
            f"gap-filled {T - n_tracked}/{T} wrist frames below "
            f"CONF_TRACKED={CONF_TRACKED}"
        )

    # H2R alignment on the filled trajectory.
    R_H2R, t_H2R, scale_used, table_y = estimate_h2r_transform(
        filled, wrist_conf, workspace_center_xy=workspace_center_xy, scale=scale
    )
    wrist_robot = apply_h2r(filled, R_H2R, t_H2R, scale_used)

    # IK.
    ik = solve_ik_trajectory(
        wrist_robot,
        lambda_smooth=lambda_smooth,
        position_cost=position_cost,
        orientation_cost=orientation_cost,
    )
    q = ik["q"]
    pos_err = ik["pos_err_m"]
    ori_err = ik["ori_err_deg"]
    iters = ik["iters_per_frame"]

    # Gripper signal.
    binary, openness, thr = compute_gripper_signal(
        filled, ep["fingertips_world"], threshold=gripper_threshold
    )

    # Variance report (FIVER-collapse guard).
    var = variance_report(q, ik["joint_names"])

    # Save trajectory artifact.
    out_npz = out_dir / f"{hdf5_path.stem}_actions.npz"
    np.savez(
        out_npz,
        q=q.astype(np.float32),
        gripper=binary.astype(np.float32),
        gripper_openness=openness.astype(np.float32),
        pos_err_m=pos_err.astype(np.float32),
        ori_err_deg=ori_err.astype(np.float32),
        iters=iters.astype(np.int32),
        wrist_targets_robot=wrist_robot.astype(np.float32),
        h2r_R=R_H2R.astype(np.float32),
        h2r_t=t_H2R.astype(np.float32),
        h2r_scale=np.float32(scale_used),
        table_y_world=np.float32(table_y),
        joint_names=np.array(ik["joint_names"]),
        active_hand=np.array(ep["active_hand"]),
    )

    res = AlignResult(
        episode=str(hdf5_path),
        task=ep["task"],
        description=ep["description"],
        environment=ep["environment"],
        active_hand=ep["active_hand"],
        n_frames=int(T),
        n_tracked_wrist=int(n_tracked),
        table_y_world=float(table_y),
        h2r_rotation=R_H2R.tolist(),
        h2r_translation=t_H2R.tolist(),
        h2r_scale=float(scale_used),
        ik_lambda_smooth=float(lambda_smooth),
        ik_position_cost=float(position_cost),
        ik_orientation_cost=float(orientation_cost),
        pos_err_m_median=float(np.median(pos_err)),
        pos_err_m_p95=float(np.percentile(pos_err, 95)),
        ori_err_deg_median=float(np.median(ori_err)),
        ori_err_deg_p95=float(np.percentile(ori_err, 95)),
        iters_mean=float(iters.mean()),
        iters_max=int(iters.max()),
        variance=var,
        gripper_threshold=float(thr),
        gripper_open_frac=float(binary.mean()),
        out_npz=str(out_npz),
        notes=" | ".join(notes),
    )
    (out_dir / f"{hdf5_path.stem}_metrics.json").write_text(
        json.dumps(asdict(res), indent=2)
    )
    return res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Action alignment (Stage 2)")
    ap.add_argument("hdf5", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/stage2"))
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--lambda-smooth", type=float, default=0.1)
    ap.add_argument("--position-cost", type=float, default=1.0)
    ap.add_argument("--orientation-cost", type=float, default=0.3)
    ap.add_argument(
        "--gripper-threshold",
        type=float,
        default=None,
        help="Override gripper openness threshold (default: per-episode median)",
    )
    args = ap.parse_args(argv)

    res = process_episode(
        args.hdf5,
        args.out_dir,
        scale=args.scale,
        lambda_smooth=args.lambda_smooth,
        position_cost=args.position_cost,
        orientation_cost=args.orientation_cost,
        gripper_threshold=args.gripper_threshold,
    )
    print("=== ActionAlignment result ===")
    d = asdict(res)
    # Pretty-print top-level scalars then a variance summary.
    for k, v in d.items():
        if k == "variance":
            print(f"  variance.n_joints_range_gt_0_3rad: "
                  f"{v['n_joints_range_gt_0_3rad']}/{v['n_joints_total']}")
            print(f"  variance.per_joint_range_rad: "
                  f"{[round(x, 3) for x in v['per_joint_range_rad']]}")
            continue
        if isinstance(v, list) and v and isinstance(v[0], list):
            continue  # skip big matrices in stdout
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
