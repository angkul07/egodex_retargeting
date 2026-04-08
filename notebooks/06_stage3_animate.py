"""
06_stage3_animate.py — three animated visualizations of Stage 3 retargeting.

Produces three MP4s per episode:

1. `<idx>_inspire_stick.mp4`
   A rotating 3D stick figure of the Inspire hand moving through the
   retargeted `q_full`, via pinocchio FK. No meshes needed. Fast.

2. `<idx>_side_by_side.mp4`
   Two 3D subplots side by side: the EgoDex 25-joint human hand skeleton
   (left) and the Inspire robot hand skeleton (right), both in
   wrist-relative coordinates. This is the "what does retargeting
   actually DO" visualization — you see the two hands move through the
   same task together.

3. `<idx>_mujoco.mp4`
   MuJoCo offscreen mesh-render of the Inspire hand via the full URDF
   (with a compiler-block injection to resolve the meshdir + drop
   unsupported .glb visual meshes, keeping .obj collision meshes). Uses
   a free camera orbiting the hand. This is the "what will the robot
   look like" view.

Run
---
    uv run python notebooks/06_stage3_animate.py --episodes 0 1 50 100 \\
        --out-dir outputs/stage3/viz

All three MP4s are ~30 FPS to match EgoDex. Total wall time ~15–30 s per
episode depending on frame count and animation type.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

# MUST be set before importing mujoco — picks headless OpenGL backend.
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import mujoco  # imported after MUJOCO_GL is set

from mimicdreamer_egodex.finger_retargeting import get_inspire_retargeter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FPS = 30
FRAME_W, FRAME_H = 720, 480
# MuJoCo's URDF loader silently ignores <visual><global offwidth/></visual>
# inside a <mujoco> extension block, so we can't enlarge the offscreen
# framebuffer from the URDF — fall back to the default 640x480 for mujoco.
MJ_FRAME_W, MJ_FRAME_H = 640, 480

FINGER_COLORS = {
    "thumb": "#d62728",
    "index": "#ff7f0e",
    "middle": "#2ca02c",
    "ring": "#1f77b4",
    "pinky": "#9467bd",
    "wrist": "black",
}

# ---------------------------------------------------------------------------
# EgoDex hand topology
# ---------------------------------------------------------------------------
def egodex_hand_joint_names(side: str) -> list[str]:
    """25 hand-joint dataset names per doc.md §2.1. Order is stable:
    wrist, 4 thumb, 5×4 fingers = 25."""
    names: list[str] = [f"{side}Hand"]
    for n in ("ThumbKnuckle", "ThumbIntermediateBase", "ThumbIntermediateTip", "ThumbTip"):
        names.append(f"{side}{n}")
    for finger in ("IndexFinger", "MiddleFinger", "RingFinger", "LittleFinger"):
        for j in ("Metacarpal", "Knuckle", "IntermediateBase", "IntermediateTip", "Tip"):
            names.append(f"{side}{finger}{j}")
    return names


def egodex_bones(side: str) -> list[tuple[str, str, str]]:
    """Parent-child joint pairs + the finger they belong to (for coloring)."""
    bones: list[tuple[str, str, str]] = []
    # Thumb chain
    thumb_chain = ["ThumbKnuckle", "ThumbIntermediateBase", "ThumbIntermediateTip", "ThumbTip"]
    bones.append((f"{side}Hand", f"{side}{thumb_chain[0]}", "thumb"))
    for a, b in zip(thumb_chain, thumb_chain[1:]):
        bones.append((f"{side}{a}", f"{side}{b}", "thumb"))
    # 4 long fingers
    finger_color_map = {
        "IndexFinger": "index",
        "MiddleFinger": "middle",
        "RingFinger": "ring",
        "LittleFinger": "pinky",
    }
    for finger, color in finger_color_map.items():
        chain = [
            f"{finger}Metacarpal",
            f"{finger}Knuckle",
            f"{finger}IntermediateBase",
            f"{finger}IntermediateTip",
            f"{finger}Tip",
        ]
        bones.append((f"{side}Hand", f"{side}{chain[0]}", color))
        for a, b in zip(chain, chain[1:]):
            bones.append((f"{side}{a}", f"{side}{b}", color))
    return bones


def load_egodex_hand_trajectory(hdf5_path: Path, side: str) -> tuple[np.ndarray, list[str]]:
    """Return (T, 25, 3) wrist-relative world-axis hand positions + the
    joint-name list in the same order. Wrist is at index 0 and trivially
    at the origin after subtraction."""
    names = egodex_hand_joint_names(side)
    with h5py.File(hdf5_path, "r") as f:
        stacks = []
        for n in names:
            stacks.append(f[f"transforms/{n}"][:, :3, 3].astype(np.float32))
    world = np.stack(stacks, axis=1)  # (T, 25, 3)
    wrist = world[:, 0:1, :]
    return world - wrist, names  # (T, 25, 3), wrist at (0,0,0)


# ---------------------------------------------------------------------------
# Inspire robot topology (stick figure)
# ---------------------------------------------------------------------------
INSPIRE_LINKS_OF_INTEREST = [
    "hand_base_link",
    "index_proximal", "index_intermediate", "index_tip",
    "middle_proximal", "middle_intermediate", "middle_tip",
    "ring_proximal", "ring_intermediate", "ring_tip",
    "pinky_proximal", "pinky_intermediate", "pinky_tip",
    "thumb_proximal_base", "thumb_proximal", "thumb_intermediate",
    "thumb_distal", "thumb_tip",
]

INSPIRE_BONES: list[tuple[str, str, str]] = [
    ("hand_base_link", "index_proximal", "index"),
    ("index_proximal", "index_intermediate", "index"),
    ("index_intermediate", "index_tip", "index"),
    ("hand_base_link", "middle_proximal", "middle"),
    ("middle_proximal", "middle_intermediate", "middle"),
    ("middle_intermediate", "middle_tip", "middle"),
    ("hand_base_link", "ring_proximal", "ring"),
    ("ring_proximal", "ring_intermediate", "ring"),
    ("ring_intermediate", "ring_tip", "ring"),
    ("hand_base_link", "pinky_proximal", "pinky"),
    ("pinky_proximal", "pinky_intermediate", "pinky"),
    ("pinky_intermediate", "pinky_tip", "pinky"),
    ("hand_base_link", "thumb_proximal_base", "thumb"),
    ("thumb_proximal_base", "thumb_proximal", "thumb"),
    ("thumb_proximal", "thumb_intermediate", "thumb"),
    ("thumb_intermediate", "thumb_distal", "thumb"),
    ("thumb_distal", "thumb_tip", "thumb"),
]


def inspire_link_trajectory(q_full: np.ndarray, side: str) -> tuple[np.ndarray, list[str]]:
    """Run pinocchio FK at every frame with the dummy free joint zeroed out
    (so the hand base stays at origin and we see pure finger motion).
    Returns (T, N, 3) link positions and the ordered list of link names."""
    retargeter = get_inspire_retargeter(side)
    robot = retargeter.optimizer.robot
    link_names = list(robot.link_names)
    link_indices = [link_names.index(name) for name in INSPIRE_LINKS_OF_INTEREST]

    q = q_full.copy().astype(np.float32)
    q[:, :6] = 0.0  # zero out 6-DOF dummy free joint so base is at origin

    T = len(q)
    out = np.zeros((T, len(INSPIRE_LINKS_OF_INTEREST), 3), dtype=np.float32)
    for t in range(T):
        robot.compute_forward_kinematics(q[t])
        for j, idx in enumerate(link_indices):
            out[t, j] = robot.get_link_pose(idx)[:3, 3]
    return out, INSPIRE_LINKS_OF_INTEREST


# ---------------------------------------------------------------------------
# Matplotlib frame → numpy BGR frame (for cv2.VideoWriter)
# ---------------------------------------------------------------------------
def fig_to_bgr(fig) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba()).reshape(h, w, 4)
    bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    return bgr


def open_writer(path: Path, w: int, h: int, fps: float = FPS) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open {path}")
    return vw


# ---------------------------------------------------------------------------
# Shared: draw a skeleton into a 3D ax
# ---------------------------------------------------------------------------
def draw_skeleton(
    ax,
    positions: np.ndarray,          # (N, 3)
    name_to_idx: dict[str, int],
    bones: list[tuple[str, str, str]],
    dot_size: float = 25.0,
):
    """Plot bones as colored line segments and joints as dots."""
    for a, b, color_key in bones:
        if a not in name_to_idx or b not in name_to_idx:
            continue
        pa = positions[name_to_idx[a]]
        pb = positions[name_to_idx[b]]
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                color=FINGER_COLORS[color_key], lw=2.5)
    # Wrist/base dot
    if "hand_base_link" in name_to_idx:
        p = positions[name_to_idx["hand_base_link"]]
        ax.scatter([p[0]], [p[1]], [p[2]], color="black", s=40)
    for label in ("leftHand", "rightHand"):
        if label in name_to_idx:
            p = positions[name_to_idx[label]]
            ax.scatter([p[0]], [p[1]], [p[2]], color="black", s=40)
    # Fingertip dots
    for b_a, b_b, color_key in bones:
        if b_b.endswith("Tip") or b_b.endswith("_tip"):
            if b_b in name_to_idx:
                p = positions[name_to_idx[b_b]]
                ax.scatter([p[0]], [p[1]], [p[2]],
                           color=FINGER_COLORS[color_key], s=dot_size)


def set_equal_3d(ax, positions_all: np.ndarray, pad: float = 0.01):
    """Make the 3D axis limits cubic so the hand isn't visually warped."""
    mn = positions_all.min(axis=0) - pad
    mx = positions_all.max(axis=0) + pad
    mid = (mn + mx) / 2
    half = (mx - mn).max() / 2
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)


# ---------------------------------------------------------------------------
# Animation 1: Inspire stick figure MP4
# ---------------------------------------------------------------------------
def animate_inspire_stick(
    q_full: np.ndarray, side: str, title: str, out_path: Path
) -> None:
    positions, link_names = inspire_link_trajectory(q_full, side)  # (T, N, 3)
    name_to_idx = {n: i for i, n in enumerate(link_names)}
    T = len(positions)

    fig = plt.figure(figsize=(FRAME_W / 100, FRAME_H / 100), dpi=100)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    all_pos = positions.reshape(-1, 3)
    vw = open_writer(out_path, FRAME_W, FRAME_H)
    try:
        for t in range(T):
            ax.clear()
            draw_skeleton(ax, positions[t], name_to_idx, INSPIRE_BONES)
            set_equal_3d(ax, all_pos, pad=0.01)
            ax.view_init(elev=20, azim=30 + 60 * t / max(1, T))  # slow spin
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_zlabel("z (m)")
            ax.set_title(f"{title}\nInspire hand — frame {t}/{T - 1} (t={t / FPS:.2f}s)")
            frame = fig_to_bgr(fig)
            if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            vw.write(frame)
    finally:
        vw.release()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Animation 2: side-by-side EgoDex skeleton + Inspire skeleton
# ---------------------------------------------------------------------------
def animate_side_by_side(
    hdf5_path: Path,
    q_full: np.ndarray,
    side: str,
    title: str,
    out_path: Path,
) -> None:
    human_pos, human_names = load_egodex_hand_trajectory(hdf5_path, side)  # (T, 25, 3)
    robot_pos, robot_names = inspire_link_trajectory(q_full, side)          # (T, N, 3)
    human_idx = {n: i for i, n in enumerate(human_names)}
    robot_idx = {n: i for i, n in enumerate(robot_names)}
    bones_h = egodex_bones(side)
    T = min(len(human_pos), len(robot_pos))

    fig = plt.figure(figsize=(FRAME_W * 1.6 / 100, FRAME_H / 100), dpi=100)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    human_all = human_pos.reshape(-1, 3)
    robot_all = robot_pos.reshape(-1, 3)
    vw = open_writer(out_path, FRAME_W * 2, FRAME_H)
    try:
        for t in range(T):
            ax1.clear()
            ax2.clear()
            draw_skeleton(ax1, human_pos[t], human_idx, bones_h)
            draw_skeleton(ax2, robot_pos[t], robot_idx, INSPIRE_BONES)
            set_equal_3d(ax1, human_all, pad=0.02)
            set_equal_3d(ax2, robot_all, pad=0.01)
            ax1.view_init(elev=20, azim=30 + 40 * t / max(1, T))
            ax2.view_init(elev=20, azim=30 + 40 * t / max(1, T))
            ax1.set_title(f"EgoDex human hand (25 joints)   t={t / FPS:.2f}s")
            ax2.set_title(f"Inspire robot hand (via retargeting)")
            for ax in (ax1, ax2):
                ax.set_xlabel("x (m)")
                ax.set_ylabel("y (m)")
                ax.set_zlabel("z (m)")
            fig.suptitle(title, fontsize=11)
            frame = fig_to_bgr(fig)
            if frame.shape[1] != FRAME_W * 2 or frame.shape[0] != FRAME_H:
                frame = cv2.resize(frame, (FRAME_W * 2, FRAME_H))
            vw.write(frame)
    finally:
        vw.release()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Animation 3: MuJoCo offscreen mesh render
# ---------------------------------------------------------------------------
def _inspire_mjcf_prepare(side: str) -> Path:
    """Load the Inspire URDF, inject a mujoco compiler block so the meshdir
    resolves and the unsupported .glb visual meshes are dropped, and write
    a temp file. Return the temp path."""
    urdf_path = Path(
        f"/workspace/third_party/dex-urdf/robots/hands/inspire_hand/inspire_hand_{side}.urdf"
    )
    txt = urdf_path.read_text()
    mesh_dir = str(urdf_path.parent.absolute()) + "/"
    injection = (
        f'<mujoco>'
        f'<compiler meshdir="{mesh_dir}" strippath="false" discardvisual="true"/>'
        f'</mujoco>'
    )
    txt2 = re.sub(r"(<robot[^>]*>)", r"\1" + injection, txt, count=1)
    tmp = Path(f"/tmp/inspire_{side}_mj.urdf")
    tmp.write_text(txt2)
    return tmp


def _build_pin2mj_qpos_map(pin_joint_names: list[str], mj_joint_names: list[str]) -> np.ndarray:
    """Return an int array `idx` of length `len(mj_joint_names)` such that
    `mj_qpos[:] = pin_q[idx]`. Pinocchio's q_full has the 6 dummy-joint
    DOFs first; MuJoCo's model has only the 12 finger DOFs. We match by
    joint name."""
    idx = np.zeros(len(mj_joint_names), dtype=np.int32)
    for i, name in enumerate(mj_joint_names):
        if name not in pin_joint_names:
            raise ValueError(
                f"MuJoCo joint {name!r} missing from pinocchio joint list"
            )
        idx[i] = pin_joint_names.index(name)
    return idx


def animate_mujoco(
    q_full: np.ndarray, side: str, title: str, out_path: Path
) -> None:
    tmp_urdf = _inspire_mjcf_prepare(side)
    model = mujoco.MjModel.from_xml_path(str(tmp_urdf))
    data = mujoco.MjData(model)

    mj_joint_names = [model.joint(i).name for i in range(model.njnt)]

    # Pinocchio joint order comes from the retargeter's robot DOF list
    retargeter = get_inspire_retargeter(side)
    pin_joint_names = list(retargeter.optimizer.robot.dof_joint_names)
    q2mj = _build_pin2mj_qpos_map(pin_joint_names, mj_joint_names)

    # Free camera looking at the hand
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = [0.0, 0.0, 0.1]
    cam.distance = 0.35
    cam.elevation = -20
    cam.azimuth = 135

    scene_opt = mujoco.MjvOption()

    renderer = mujoco.Renderer(model, height=MJ_FRAME_H, width=MJ_FRAME_W)
    T = len(q_full)
    vw = open_writer(out_path, MJ_FRAME_W, MJ_FRAME_H)
    try:
        for t in range(T):
            # Pinocchio q_full has (6 dummy + 12 hand) = 18 DOFs, MuJoCo has 12.
            data.qpos[:] = q_full[t, q2mj]
            mujoco.mj_forward(model, data)
            # slow camera spin
            cam.azimuth = 135 + 60 * t / max(1, T)
            renderer.update_scene(data, camera=cam, scene_option=scene_opt)
            pix = renderer.render()  # (H, W, 3) RGB uint8
            bgr = cv2.cvtColor(pix, cv2.COLOR_RGB2BGR)
            # Title/frame overlay
            cv2.putText(
                bgr,
                f"{title}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                bgr,
                f"frame {t}/{T - 1}  t={t / FPS:.2f}s",
                (10, MJ_FRAME_H - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            vw.write(bgr)
    finally:
        vw.release()
        renderer.close()


# ---------------------------------------------------------------------------
# Per-episode driver
# ---------------------------------------------------------------------------
def animate_episode(
    idx: int,
    task_dir: Path,
    stage3_dir: Path,
    out_dir: Path,
) -> None:
    hdf5_path = task_dir / f"{idx}.hdf5"
    npz_path = stage3_dir / f"{idx}_fingers.npz"
    if not hdf5_path.exists():
        raise FileNotFoundError(hdf5_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} — run Stage 3 batch first")

    d = np.load(npz_path, allow_pickle=True)
    q_full = d["q_full"]
    side = str(d["active_hand"])

    # Metadata for title (best-effort)
    import json

    meta_path = stage3_dir / f"{idx}_metrics.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    task = meta.get("task", "unknown")
    desc = meta.get("description", "")
    title_short = f"ep{idx}  {task}  ({side})"
    title_long = f"Episode {idx} — {task} ({side}): {desc}"

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n== animating episode {idx} ({side}, {len(q_full)} frames) ==")
    print(f"  (1/3) Inspire stick figure → {out_dir}/{idx}_inspire_stick.mp4")
    animate_inspire_stick(
        q_full, side, title_long, out_dir / f"{idx}_inspire_stick.mp4"
    )
    print(f"  (2/3) Side-by-side human vs robot → {out_dir}/{idx}_side_by_side.mp4")
    animate_side_by_side(
        hdf5_path, q_full, side, title_long, out_dir / f"{idx}_side_by_side.mp4"
    )
    print(f"  (3/3) MuJoCo mesh render → {out_dir}/{idx}_mujoco.mp4")
    animate_mujoco(q_full, side, title_short, out_dir / f"{idx}_mujoco.mp4")
    print(f"  done episode {idx}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 3 animations")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1])
    ap.add_argument(
        "--task-dir", type=Path, default=Path("data/test/basic_pick_place")
    )
    ap.add_argument("--stage3-dir", type=Path, default=Path("outputs/stage3"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/stage3/viz"))
    args = ap.parse_args(argv)

    for idx in args.episodes:
        animate_episode(idx, args.task_dir, args.stage3_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
