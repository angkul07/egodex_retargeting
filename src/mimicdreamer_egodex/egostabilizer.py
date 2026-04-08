"""
Stage 1 — EgoStabilizer.

Plane-induced (homography) stabilization for EgoDex egocentric clips.

Approach
--------
EgoDex provides per-frame ARKit camera-to-world poses (`transforms/camera`)
plus a single intrinsic matrix (`camera/intrinsic`). For a planar workspace
(the table), pixel correspondences between two camera views are governed by
the plane-induced homography:

    H_{src->dst} = K (R_{dst<-src} - t_{dst<-src} n_src^T / d_src) K^{-1}

We warp every frame to look as if it were taken from a fixed *reference*
camera (frame 0 by default). After warping, the wearer's small head micro-
movements are removed and the workspace is stationary in the image.

Calibrated conventions
----------------------
- ARKit world frame is **y-up** (doc.md §3, R-001).
- `transforms/camera` is camera-to-world (verified pre-Stage-1).
- Table plane: y = table_y, where table_y = 5th-percentile world-y of the
  active hand's wrist over confidence-gated frames.
- table distance is computed per frame (not the obsolete §1.1 constant 0.5).
- Confidence floor: 0.10 hard reject (doc.md §5, R-002).

CLI
---
    uv run python -m mimicdreamer_egodex.egostabilizer \\
        data/test/basic_pick_place/0.hdf5 \\
        --out-dir outputs/stage1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np

# --- Calibrated constants from doc.md §5 (R-002) ---------------------------
CONF_TRACKED = 0.10  # hard reject below this
CONF_HIGH = 0.50     # "high quality" — used as a soft weight, not a switch

# World-up axis under ARKit y-up convention.
N_WORLD_UP = np.array([0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def pick_active_hand(rwrist_xyz: np.ndarray, lwrist_xyz: np.ndarray) -> str:
    """Active hand = hand whose total wrist excursion (sum of per-axis ranges)
    is larger. Matches the rule used in `notebooks/01_variance_report.py`."""
    r_total = float((rwrist_xyz.max(0) - rwrist_xyz.min(0)).sum())
    l_total = float((lwrist_xyz.max(0) - lwrist_xyz.min(0)).sum())
    return "right" if r_total >= l_total else "left"


def estimate_table_y(
    wrist_world_xyz: np.ndarray,
    wrist_conf: np.ndarray | None,
    conf_floor: float = CONF_TRACKED,
    percentile: float = 5.0,
    bias: float = 0.0,
) -> float:
    """5th-percentile world-y of confidence-gated wrist samples.

    `bias` (m) optionally subtracts a small offset for the wrist-above-
    fingertips gap. For pure stabilization the bias mostly washes out, so the
    default is 0; revisit if homography reprojection error is biased.
    """
    ys = wrist_world_xyz[:, 1]
    if wrist_conf is not None:
        mask = wrist_conf > conf_floor
        if int(mask.sum()) >= 5:
            ys = ys[mask]
    return float(np.percentile(ys, percentile) - bias)


def plane_homography(
    K: np.ndarray,
    T_src: np.ndarray,
    T_dst: np.ndarray,
    table_y: float,
) -> np.ndarray:
    """
    Plane-induced homography mapping pixels in `src` to pixels in `dst`,
    induced by the world plane y = table_y (y-up).

    T_src, T_dst are 4x4 camera-to-world matrices: X_world = T @ X_cam.
    Returns the 3x3 H such that x_dst ~ H @ x_src in homogeneous pixel coords.
    """
    R_ws = T_src[:3, :3]
    t_ws = T_src[:3, 3]
    R_wd = T_dst[:3, :3]
    t_wd = T_dst[:3, 3]

    # Relative pose dst<-src: X_dst = R_ds X_src + t_ds.
    R_ds = R_wd.T @ R_ws
    t_ds = R_wd.T @ (t_ws - t_wd)

    # Plane in src frame as n_s^T X_s = d_s, derived from the world plane:
    #   n_world^T (R_ws X_s + t_ws) = table_y
    #   (R_ws^T n_world)^T X_s = table_y - n_world^T t_ws
    n_s = R_ws.T @ N_WORLD_UP
    d_s = float(table_y - N_WORLD_UP @ t_ws)
    if abs(d_s) < 1e-6:
        raise ValueError(
            f"Camera sits on the table plane (d_s={d_s}); homography undefined."
        )

    # Derivation: for X_s on the plane, n_s.T X_s = d_s, so X_s n_s.T / d_s
    # times any vector "selects" plane points. Then
    #     X_d = R_ds X_s + t_ds = R_ds X_s + t_ds (n_s.T X_s / d_s)
    #         = (R_ds + (t_ds n_s.T) / d_s) X_s
    # Hence the sign is "+", not the "-" that appears in some references
    # (which use the opposite (src,dst) ordering or an inward normal).
    K_inv = np.linalg.inv(K)
    A = R_ds + np.outer(t_ds, n_s) / d_s
    return K @ A @ K_inv


def rotation_angle_rad(R: np.ndarray) -> float:
    cos_t = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cos_t))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def mean_interframe_camera_angle_deg(extrinsics: np.ndarray) -> float:
    """Mean magnitude of inter-frame camera rotation, from raw extrinsics."""
    if len(extrinsics) < 2:
        return 0.0
    angles = []
    for t in range(1, len(extrinsics)):
        R_rel = extrinsics[t, :3, :3].T @ extrinsics[t - 1, :3, :3]
        angles.append(rotation_angle_rad(R_rel))
    return float(np.degrees(np.mean(angles)))


def mean_interframe_orb_displacement(
    frames: list[np.ndarray], max_pairs: int = 60
) -> float:
    """Mean ORB feature displacement (px) between consecutive sampled frames.

    On RAW frames this is large (head moves the camera). On STABILIZED frames
    it should be much smaller — that's the headline stabilization metric.
    """
    if len(frames) < 2:
        return 0.0
    orb = cv2.ORB_create(nfeatures=400)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    idxs = np.linspace(0, len(frames) - 2, num=min(max_pairs, len(frames) - 1), dtype=int)
    disps: list[float] = []
    for i in idxs:
        g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        k1, d1 = orb.detectAndCompute(g1, None)
        k2, d2 = orb.detectAndCompute(g2, None)
        if d1 is None or d2 is None or len(d1) == 0 or len(d2) == 0:
            continue
        m = bf.match(d1, d2)
        if not m:
            continue
        m = sorted(m, key=lambda x: x.distance)[:80]
        p1 = np.array([k1[mm.queryIdx].pt for mm in m])
        p2 = np.array([k2[mm.trainIdx].pt for mm in m])
        disps.append(float(np.linalg.norm(p1 - p2, axis=1).mean()))
    return float(np.mean(disps)) if disps else 0.0


def homography_reprojection_rmse(
    K: np.ndarray,
    extrinsics: np.ndarray,
    frames: list[np.ndarray],
    table_y: float,
    ref_idx: int = 0,
    max_pairs: int = 30,
    inlier_frac: float = 0.5,
) -> float:
    """
    Inlier-filtered H-RMSE: for sampled (ref, t) pairs, ORB-match the two
    frames, project ref points through our geometric homography, and measure
    pixel error vs. the matched target points. We then keep only the best
    `inlier_frac` of matches per pair (RANSAC-lite — the planar-background
    matches dominate the lower tail; the moving hand/object matches dominate
    the upper tail and cannot be explained by *any* planar homography).
    Returns the median of per-pair inlier RMSEs.

    Without inlier filtering this metric is dominated by off-plane content
    (most ORB features in EgoDex sit on the hand and the manipulated object),
    which a plane-induced homography fundamentally cannot predict.
    """
    if len(frames) < 2:
        return float("nan")
    orb = cv2.ORB_create(nfeatures=800)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    candidates = [t for t in range(len(frames)) if t != ref_idx]
    idxs = np.linspace(0, len(candidates) - 1, num=min(max_pairs, len(candidates)), dtype=int)
    g_ref = cv2.cvtColor(frames[ref_idx], cv2.COLOR_BGR2GRAY)
    k_ref, d_ref = orb.detectAndCompute(g_ref, None)
    if d_ref is None or len(d_ref) == 0:
        return float("nan")
    rmses: list[float] = []
    for ii in idxs:
        t = candidates[ii]
        try:
            H = plane_homography(K, extrinsics[ref_idx], extrinsics[t], table_y)
        except ValueError:
            continue
        g_t = cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY)
        k_t, d_t = orb.detectAndCompute(g_t, None)
        if d_t is None or len(d_t) == 0:
            continue
        m = bf.match(d_ref, d_t)
        if len(m) < 16:
            continue
        m = sorted(m, key=lambda x: x.distance)[:120]
        p_ref = np.array([(*k_ref[mm.queryIdx].pt, 1.0) for mm in m]).T  # (3, N)
        p_t = np.array([k_t[mm.trainIdx].pt for mm in m])                # (N, 2)
        proj = (H @ p_ref).T                                             # (N, 3)
        proj = proj[:, :2] / proj[:, 2:3]
        per_pt = np.sqrt(np.sum((proj - p_t) ** 2, axis=1))              # (N,)
        k = max(8, int(round(len(per_pt) * inlier_frac)))
        inliers = np.sort(per_pt)[:k]
        rmses.append(float(np.sqrt(np.mean(inliers ** 2))))
    return float(np.median(rmses)) if rmses else float("nan")


# ---------------------------------------------------------------------------
# Warping
# ---------------------------------------------------------------------------
def warp_to_reference(
    K: np.ndarray,
    extrinsics: np.ndarray,
    frames: list[np.ndarray],
    table_y: float,
    ref_idx: int = 0,
    inpaint: bool = False,
) -> list[np.ndarray]:
    h, w = frames[0].shape[:2]
    out: list[np.ndarray] = []
    for t, frame in enumerate(frames):
        if t == ref_idx:
            out.append(frame.copy())
            continue
        # We want pixels in the destination = ref frame, so src=t, dst=ref.
        H = plane_homography(K, extrinsics[t], extrinsics[ref_idx], table_y)
        warped = cv2.warpPerspective(
            frame,
            H,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        if inpaint:
            mask = (warped.sum(axis=2) == 0).astype(np.uint8) * 255
            if mask.any():
                warped = cv2.inpaint(warped, mask, 3, cv2.INPAINT_TELEA)
        out.append(warped)
    return out


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def load_video(path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames: list[np.ndarray] = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames, fps


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
@dataclass
class StabilizeResult:
    episode: str
    method: str  # "exact" or "ransac_fallback"
    n_frames: int
    n_warped: int
    table_y_world: float
    active_hand: str
    fps: float
    raw_mean_interframe_camera_angle_deg: float
    raw_mean_interframe_pixel_disp: float
    stab_mean_interframe_pixel_disp: float
    pixel_disp_reduction_ratio: float  # raw / stab; >1 = stabilization worked
    homography_rmse_px_median: float
    out_video: str
    notes: str


def stabilize_episode(
    hdf5_path: Path,
    out_dir: Path,
    ref_frame: int = 0,
    inpaint: bool = False,
    downsample: int | None = None,
    force_method: str | None = None,
) -> StabilizeResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = hdf5_path.with_suffix(".mp4")
    if not mp4_path.exists():
        raise FileNotFoundError(f"missing paired MP4 {mp4_path}")

    with h5py.File(hdf5_path, "r") as f:
        K = f["camera/intrinsic"][:].astype(np.float64)
        extrinsics = f["transforms/camera"][:].astype(np.float64)
        rwrist = f["transforms/rightHand"][:, :3, 3]
        lwrist = f["transforms/leftHand"][:, :3, 3]
        rconf = f["confidences/rightHand"][:] if "confidences/rightHand" in f else None
        lconf = f["confidences/leftHand"][:] if "confidences/leftHand" in f else None

    active = pick_active_hand(rwrist, lwrist)
    wrist = rwrist if active == "right" else lwrist
    wconf = rconf if active == "right" else lconf
    table_y = estimate_table_y(wrist, wconf)

    frames, fps = load_video(mp4_path)
    n = min(len(frames), len(extrinsics))
    frames = frames[:n]
    extrinsics = extrinsics[:n]

    if downsample and downsample > 1:
        frames = frames[::downsample]
        extrinsics = extrinsics[::downsample]

    n_tracked = int(
        ((wconf if wconf is not None else np.ones(len(wrist))) > CONF_TRACKED).sum()
    )
    method = force_method or ("exact" if n_tracked >= 5 else "ransac_fallback")

    notes: list[str] = []
    if method == "exact":
        warped = warp_to_reference(
            K, extrinsics, frames, table_y, ref_idx=ref_frame, inpaint=inpaint
        )
    else:
        # RANSAC fallback (vidstab) is intentionally a stub for now: this episode
        # set is overwhelmingly well-tracked (R-002), so we'll wire vidstab in
        # only if/when an episode actually needs it.
        notes.append(
            "RANSAC fallback path is a stub: writing raw frames unchanged. "
            "Wire in vidstab once a real low-confidence episode is found."
        )
        warped = [f.copy() for f in frames]

    raw_disp = mean_interframe_orb_displacement(frames)
    stab_disp = mean_interframe_orb_displacement(warped)
    raw_angle = mean_interframe_camera_angle_deg(extrinsics)
    h_rmse = homography_reprojection_rmse(K, extrinsics, frames, table_y, ref_idx=ref_frame)

    out_video = out_dir / f"{hdf5_path.stem}_stabilized.mp4"
    write_video(out_video, warped, fps)

    res = StabilizeResult(
        episode=str(hdf5_path),
        method=method,
        n_frames=len(frames),
        n_warped=len(warped),
        table_y_world=float(table_y),
        active_hand=active,
        fps=float(fps),
        raw_mean_interframe_camera_angle_deg=raw_angle,
        raw_mean_interframe_pixel_disp=raw_disp,
        stab_mean_interframe_pixel_disp=stab_disp,
        pixel_disp_reduction_ratio=(raw_disp / stab_disp) if stab_disp > 1e-9 else float("inf"),
        homography_rmse_px_median=h_rmse,
        out_video=str(out_video),
        notes=" | ".join(notes),
    )
    (out_dir / f"{hdf5_path.stem}_metrics.json").write_text(json.dumps(asdict(res), indent=2))
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="EgoStabilizer (Stage 1)")
    ap.add_argument("hdf5", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/stage1"))
    ap.add_argument("--ref-frame", type=int, default=0)
    ap.add_argument("--inpaint", action="store_true")
    ap.add_argument(
        "--downsample",
        type=int,
        default=None,
        help="Stride for debugging (e.g. 4 = every 4th frame)",
    )
    ap.add_argument(
        "--force-method",
        choices=["exact", "ransac_fallback"],
        default=None,
    )
    args = ap.parse_args(argv)

    res = stabilize_episode(
        args.hdf5,
        args.out_dir,
        ref_frame=args.ref_frame,
        inpaint=args.inpaint,
        downsample=args.downsample,
        force_method=args.force_method,
    )
    print("=== EgoStabilizer result ===")
    for k, v in asdict(res).items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
