"""
Stage 3 — Dexterous finger retargeting (EgoDex 25-joint → Inspire 6-DOF).

Takes an EgoDex HDF5 episode, pulls the active-hand fingertip trajectories
(5 tips in the ARKit world frame), converts them to wrist-relative
coordinates, and runs `dex-retargeting`'s `PositionOptimizer` against the
bundled Inspire-hand config to produce a per-frame Inspire joint-angle
trajectory. Writes an npz + a metrics JSON per episode.

Target hand: Inspire (see `plan.md` R-007)
-------------------------------------------
The `dex-retargeting` package ships **both** `offline/inspire_hand_left.yml`
and `offline/inspire_hand_right.yml` out of the box — we do not need to
write a custom config. The referenced URDFs live in the `dex-urdf` repo,
which has to be cloned separately (not a pip artifact); we vendor it under
`third_party/dex-urdf` at commit `7304c7f`.

The 6 optimized ("target") DOFs per hand are:
    index_proximal, middle_proximal, ring_proximal, pinky_proximal,
    thumb_proximal_yaw, thumb_proximal_pitch.
The URDF declares mimic joints that tie the intermediate/distal joints
to their proximal counterparts, so the full robot has 12 non-dummy DOFs
but only these 6 are learned. The config also adds a 6-DOF dummy free
joint at the hand root so the optimizer can absorb any wrist/base offset
in the input frame.

Input contract (confirmed empirically 2026-04-08 — see `plan.md` D-007)
----------------------------------------------------------------------
`SeqRetargeting.retarget(ref_value)` expects a pre-sliced **(5, 3)** array
of fingertip positions, in the order
    [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip],
i.e. already indexed by the YAML's `target_link_human_indices = [4, 8,
12, 16, 20]` (which are MediaPipe landmark indices in the teleop
pathway — irrelevant here because we pre-slice). The optimizer does NOT
index into the 21-landmark array itself; the stored
`target_link_human_indices` is metadata for teleop callers that do their
own slicing. Passing a (21, 3) array directly triggers a silent
SmoothL1Loss broadcast warning and produces garbage.

Coordinate frame: we subtract the EgoDex active-hand wrist world position
from each fingertip (world frame, y-up). The dummy free joint then
absorbs any residual base transform. No scaling: the Inspire hand and
a human adult hand are geometrically similar enough that 1:1 works;
confirmed on `basic_pick_place/0` where the fingertip-to-wrist range
matches `doc.md` §7 exactly.

CLI
---
    uv run python -m mimicdreamer_egodex.finger_retargeting \\
        data/test/basic_pick_place/0.hdf5 \\
        --out-dir outputs/stage3
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import h5py
import numpy as np

# dex_retargeting imports torch at package init, so don't import this module
# from code paths that don't need it.
import dex_retargeting  # noqa: F401
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting

from mimicdreamer_egodex.action_alignment import pick_active_hand  # reuse

# --- Calibrated constants (from doc.md §5, R-002) ------------------------
CONF_TRACKED = 0.10

# --- Inspire hand — config + URDF paths ----------------------------------
# The YAMLs ship inside the `dex_retargeting` wheel.
_DEX_RETARGETING_CFG_DIR = Path(dex_retargeting.__file__).parent / "configs" / "offline"
INSPIRE_CONFIG_PATHS = {
    "left": _DEX_RETARGETING_CFG_DIR / "inspire_hand_left.yml",
    "right": _DEX_RETARGETING_CFG_DIR / "inspire_hand_right.yml",
}

# URDFs live in the separate dex-urdf repo. Default to the vendored clone;
# override with DEX_URDF_DIR if you have it elsewhere. The YAML path is
# relative to this directory: `inspire_hand/inspire_hand_{side}.urdf`.
DEFAULT_DEX_URDF_DIR = Path(
    os.environ.get("DEX_URDF_DIR", "/workspace/third_party/dex-urdf/robots/hands")
)

# Order matches the Inspire YAML's `target_link_names`:
#     [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
FINGERTIPS_ORDER = ("thumb", "index", "middle", "ring", "pinky")

# EgoDex dataset-side joint names, in the same order as FINGERTIPS_ORDER.
# EgoDex calls "pinky" → "LittleFinger" (see doc.md §2.1).
EGODEX_TIP_NAMES = {
    "right": [
        "rightThumbTip",
        "rightIndexFingerTip",
        "rightMiddleFingerTip",
        "rightRingFingerTip",
        "rightLittleFingerTip",
    ],
    "left": [
        "leftThumbTip",
        "leftIndexFingerTip",
        "leftMiddleFingerTip",
        "leftRingFingerTip",
        "leftLittleFingerTip",
    ],
}
EGODEX_WRIST_NAMES = {"left": "leftHand", "right": "rightHand"}

# The 6 "target" (optimized) Inspire DOFs — extracted from the full 18-DOF
# robot qpos by NAME, not by index, so URDF joint-order changes don't
# silently break us. Order here is the canonical action-vector order used
# downstream (index, middle, ring, pinky, thumb_yaw, thumb_pitch).
INSPIRE_TARGET_FINGER_JOINTS = (
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint",
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint",
)


# ---------------------------------------------------------------------------
# Retargeter cache (one per side; building the config is ~100 ms)
# ---------------------------------------------------------------------------
_CACHED_RETARGETERS: dict[str, SeqRetargeting] = {}
_URDF_DIR_SET = False


def _ensure_urdf_dir(urdf_dir: Path = DEFAULT_DEX_URDF_DIR) -> None:
    global _URDF_DIR_SET
    if _URDF_DIR_SET:
        return
    if not urdf_dir.exists():
        raise FileNotFoundError(
            f"dex-urdf directory not found at {urdf_dir}. Clone it with:\n"
            f"  git clone https://github.com/dexsuite/dex-urdf {urdf_dir.parent}/dex-urdf\n"
            f"Or set DEX_URDF_DIR to an existing copy."
        )
    RetargetingConfig.set_default_urdf_dir(urdf_dir)
    _URDF_DIR_SET = True


def get_inspire_retargeter(side: Literal["left", "right"]) -> SeqRetargeting:
    """Return a per-side `SeqRetargeting` instance; cached across episodes
    because building the pinocchio model + nlopt solver is not free."""
    _ensure_urdf_dir()
    if side not in _CACHED_RETARGETERS:
        cfg_path = INSPIRE_CONFIG_PATHS[side]
        if not cfg_path.exists():
            raise FileNotFoundError(f"Inspire config missing: {cfg_path}")
        cfg = RetargetingConfig.load_from_file(cfg_path)
        _CACHED_RETARGETERS[side] = cfg.build()
    return _CACHED_RETARGETERS[side]


def reset_retargeter(side: Literal["left", "right"]) -> None:
    """Reset a retargeter's `last_qpos` to mid-range — call between
    independent episodes so a warm-start from the previous episode doesn't
    bias the first frame of the next one."""
    if side in _CACHED_RETARGETERS:
        _CACHED_RETARGETERS[side].reset()


# ---------------------------------------------------------------------------
# HDF5 extraction
# ---------------------------------------------------------------------------
def load_episode(hdf5_path: Path) -> dict:
    """Pull wrist + 5 fingertip trajectories for the active hand.

    All positions are in the ARKit world frame (y-up) as per doc.md §2 /
    D-004 — `transforms/<joint>` are already world frame.
    """
    with h5py.File(hdf5_path, "r") as f:
        rwrist_p = f["transforms/rightHand"][:, :3, 3]
        lwrist_p = f["transforms/leftHand"][:, :3, 3]
        active: Literal["left", "right"] = pick_active_hand(rwrist_p, lwrist_p)  # type: ignore

        wrist_key = EGODEX_WRIST_NAMES[active]
        wrist_world = f[f"transforms/{wrist_key}"][:, :3, 3].astype(np.float64)
        wrist_conf = f[f"confidences/{wrist_key}"][:].astype(np.float64)

        tip_names = EGODEX_TIP_NAMES[active]
        tips_world = np.stack(
            [f[f"transforms/{n}"][:, :3, 3] for n in tip_names], axis=1
        ).astype(np.float64)  # (T, 5, 3)
        tips_conf = np.stack(
            [f[f"confidences/{n}"][:] for n in tip_names], axis=1
        ).astype(np.float64)  # (T, 5)

        task = _attr(f, "task", "unknown")
        env = _attr(f, "environment", "")
        desc = _attr(f, "description", "")

    return {
        "hdf5_path": str(hdf5_path),
        "active_hand": active,
        "wrist_world": wrist_world,          # (T, 3)
        "wrist_conf": wrist_conf,            # (T,)
        "tips_world": tips_world,            # (T, 5, 3)
        "tips_conf": tips_conf,              # (T, 5)
        "task": task,
        "environment": env,
        "description": desc,
    }


def _attr(f: h5py.File, key: str, default):
    if key not in f.attrs:
        return default
    v = f.attrs[key]
    return v.decode() if isinstance(v, bytes) else str(v)


def wrist_relative_tips(wrist_world: np.ndarray, tips_world: np.ndarray) -> np.ndarray:
    """Subtract the wrist position from each fingertip (world frame).
    The dummy free joint in the Inspire config absorbs any residual base
    transform; no rotation re-frame is needed.
    """
    return (tips_world - wrist_world[:, None, :]).astype(np.float32)


# ---------------------------------------------------------------------------
# Retargeting loop
# ---------------------------------------------------------------------------
def retarget_sequence(
    tips_rel: np.ndarray,
    side: Literal["left", "right"],
) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    """Retarget a (T, 5, 3) wrist-relative fingertip trajectory.

    Returns
    -------
    q_full : (T, 18) full robot qpos (6 dummy free-joint + 12 hand DOFs)
    q_finger : (T, 6) the 6 Inspire target proximal joints in
               `INSPIRE_TARGET_FINGER_JOINTS` order — the slice that feeds
               the Stage 4 action vector.
    joint_names_full : list[str] of length 18 (robot DOF names in URDF order)
    elapsed_s : wall time for the retargeting loop
    """
    retargeter = get_inspire_retargeter(side)
    reset_retargeter(side)  # start from mid-range, not previous episode's q

    T = len(tips_rel)
    n_dof = retargeter.optimizer.robot.dof
    joint_names_full = list(retargeter.optimizer.robot.dof_joint_names)

    # Indices of the 6 target DOFs inside the full robot qpos.
    tgt_idx = np.array(
        [joint_names_full.index(n) for n in INSPIRE_TARGET_FINGER_JOINTS],
        dtype=np.int32,
    )

    q_full = np.zeros((T, n_dof), dtype=np.float32)
    t0 = time.perf_counter()
    for t in range(T):
        q_full[t] = retargeter.retarget(tips_rel[t].astype(np.float32))
    elapsed = time.perf_counter() - t0

    q_finger = q_full[:, tgt_idx].copy()
    return q_full, q_finger, joint_names_full, elapsed


# ---------------------------------------------------------------------------
# Variance / sanity reporting
# ---------------------------------------------------------------------------
def variance_report(
    q_finger: np.ndarray,
    q_full: np.ndarray,
    joint_names_full: list[str],
) -> dict:
    """FIVER-collapse guard + Inspire-specific sanity checks.

    - Per-joint range/std over the episode for the 6 target DOFs and for
      all 12 non-dummy robot DOFs.
    - Count of target DOFs with range > 0.1 rad (Inspire joints move
      within narrower limits than UR5e; 0.1 is the right threshold here,
      not Stage 2's 0.3).
    """
    tgt_names = list(INSPIRE_TARGET_FINGER_JOINTS)
    tgt_ranges = (q_finger.max(axis=0) - q_finger.min(axis=0)).astype(float)
    tgt_stds = q_finger.std(axis=0).astype(float)
    tgt_means = q_finger.mean(axis=0).astype(float)

    # Full (12 non-dummy) joint stats for reporting
    full_start = 6  # first 6 are dummy free-joint
    full_names = joint_names_full[full_start:]
    full_ranges = (q_full[:, full_start:].max(0) - q_full[:, full_start:].min(0)).astype(float)

    return {
        "target_joint_names": tgt_names,
        "target_joint_range_rad": tgt_ranges.tolist(),
        "target_joint_std_rad": tgt_stds.tolist(),
        "target_joint_mean_rad": tgt_means.tolist(),
        "n_target_joints_range_gt_0_1rad": int((tgt_ranges > 0.1).sum()),
        "n_target_joints_total": len(tgt_names),
        "full_joint_names": full_names,
        "full_joint_range_rad": full_ranges.tolist(),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
@dataclass
class RetargetResult:
    episode: str
    task: str
    description: str
    environment: str
    active_hand: str
    n_frames: int
    mean_tip_to_wrist_m: float
    retarget_wall_s: float
    retarget_ms_per_frame: float
    variance: dict = field(default_factory=dict)
    out_npz: str = ""
    notes: str = ""


def process_episode(
    hdf5_path: Path,
    out_dir: Path,
) -> RetargetResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    ep = load_episode(hdf5_path)
    wrist_world = ep["wrist_world"]
    tips_world = ep["tips_world"]
    active = ep["active_hand"]

    # Flag low-confidence fingertip frames but do not reject — the dummy free
    # joint tolerates jitter, and hard-rejecting fingertips would also force
    # gap-filling on the wrist (already handled upstream in Stage 2 if needed).
    # Just report how much of the tip stream was above the confidence floor.
    tips_conf = ep["tips_conf"]          # (T, 5)
    worst_tip_conf = tips_conf.min(axis=1)  # (T,)
    n_all_tips_tracked = int((worst_tip_conf > CONF_TRACKED).sum())
    if n_all_tips_tracked < len(wrist_world):
        notes.append(
            f"{len(wrist_world) - n_all_tips_tracked}/{len(wrist_world)} "
            f"frames had at least one tip below CONF_TRACKED={CONF_TRACKED}"
        )

    tips_rel = wrist_relative_tips(wrist_world, tips_world)  # (T, 5, 3)

    q_full, q_finger, joint_names_full, elapsed = retarget_sequence(tips_rel, active)

    var = variance_report(q_finger, q_full, joint_names_full)

    mean_spread = float(np.linalg.norm(tips_rel, axis=-1).mean())

    out_npz = out_dir / f"{hdf5_path.stem}_fingers.npz"
    np.savez(
        out_npz,
        q_finger=q_finger.astype(np.float32),            # (T, 6) -- action-vector input
        q_full=q_full.astype(np.float32),                # (T, 18) -- full robot DOF
        joint_names_target=np.array(list(INSPIRE_TARGET_FINGER_JOINTS)),
        joint_names_full=np.array(joint_names_full),
        tips_rel=tips_rel.astype(np.float32),            # (T, 5, 3) -- input used
        wrist_world=wrist_world.astype(np.float32),      # (T, 3) -- debugging
        tips_world=tips_world.astype(np.float32),        # (T, 5, 3) -- debugging
        tips_conf=tips_conf.astype(np.float32),          # (T, 5)
        active_hand=np.array(active),
        hand_model=np.array("inspire"),
    )

    res = RetargetResult(
        episode=str(hdf5_path),
        task=ep["task"],
        description=ep["description"],
        environment=ep["environment"],
        active_hand=active,
        n_frames=int(len(wrist_world)),
        mean_tip_to_wrist_m=mean_spread,
        retarget_wall_s=float(elapsed),
        retarget_ms_per_frame=float(elapsed / max(1, len(wrist_world)) * 1000.0),
        variance=var,
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
    ap = argparse.ArgumentParser(description="Finger retargeting (Stage 3)")
    ap.add_argument("hdf5", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/stage3"))
    args = ap.parse_args(argv)

    res = process_episode(args.hdf5, args.out_dir)
    print("=== FingerRetargeting result ===")
    d = asdict(res)
    for k, v in d.items():
        if k == "variance":
            print(f"  variance.n_target_joints_range_gt_0_1rad: "
                  f"{v['n_target_joints_range_gt_0_1rad']}/{v['n_target_joints_total']}")
            for name, rng in zip(v["target_joint_names"], v["target_joint_range_rad"]):
                print(f"    {name:35s}  range={rng:5.3f} rad")
            continue
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
