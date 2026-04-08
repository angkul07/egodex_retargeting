"""
Stage 4 — MuJoCo evaluation environment for the trained ACT policy.

Builds a UR5e + Inspire-hand + tabletop + small primitive object scene
and exposes a tiny gym-like Env class with reset / step / render /
get_obs / success methods. Uses `mujoco.MjSpec.attach()` to merge the
menagerie UR5e MJCF and the dex-urdf Inspire URDF (with the strippath/
discardvisual injection from D-010). No external menagerie/lerobot env
needed.

Action vector
-------------
13-D, matching the trained policy:
    [arm_6, gripper_1, finger_6]
where:
- arm_6 = UR5e joint angles in the order
    [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
- gripper_1 = scalar in [0, 1] (binary openness; not directly used in
  kinematic control here, since the dex hand fully describes finger pose)
- finger_6 = Inspire target proximals in the order
    [index, middle, ring, pinky, thumb_yaw, thumb_pitch]
  (matches `INSPIRE_TARGET_FINGER_JOINTS` from finger_retargeting.py)

The env applies the action by writing directly to `data.qpos` for the
arm + the 6 Inspire target proximals (kinematic control). The 6 mimic
joints (intermediate/distal of the long fingers and thumb) are tied to
their proximals via `mj_forward` constraint resolution. The free-joint
object is fully physics-driven so contact + friction determine whether
the hand actually grasps and lifts it.

Coordinate convention
---------------------
Matches Stage 2's H2R mapping:
- Table top sits at robot z = 0 (Stage 2 pushes `table_y` to robot z=0)
- Workspace center at (x=0.5, y=0) — 0.5 m in front of the UR5e base,
  on its midline
- See `action_alignment.py::WORKSPACE_CENTER_XY`

The egocentric camera is placed roughly where an AVP wearer's head
would be relative to the workspace, so the eval-time RGB observations
have a hope of matching the EgoDex training distribution.
"""

from __future__ import annotations

import os

# Headless OpenGL must be set BEFORE `import mujoco`. See plan.md D-009.
os.environ.setdefault("MUJOCO_GL", "egl")

import re
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

# ── Asset locations ────────────────────────────────────────────────────────
UR5E_SCENE = Path(
    "/root/.cache/robot_descriptions/mujoco_menagerie/universal_robots_ur5e/scene.xml"
)
INSPIRE_URDF_RIGHT = Path(
    "/workspace/third_party/dex-urdf/robots/hands/inspire_hand/inspire_hand_right.urdf"
)

# ── Joint name lookup ──────────────────────────────────────────────────────
UR5E_ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
# After attach with prefix='ins_'. The 6 TARGET DOFs (the ones the retargeter
# optimizes); the other 6 mimic joints are driven by URDF constraints.
INSPIRE_TARGET_JOINTS_PREFIXED = (
    "ins_index_proximal_joint",
    "ins_middle_proximal_joint",
    "ins_ring_proximal_joint",
    "ins_pinky_proximal_joint",
    "ins_thumb_proximal_yaw_joint",
    "ins_thumb_proximal_pitch_joint",
)

# ── Constants matching Stage 2 H2R ─────────────────────────────────────────
TABLE_X = 0.5            # Stage 2 WORKSPACE_CENTER_XY[0]
TABLE_Y = 0.0            # Stage 2 WORKSPACE_CENTER_XY[1]
TABLE_HALF_X = 0.30
TABLE_HALF_Y = 0.30
TABLE_HALF_Z = 0.025     # 5 cm thick; top surface at z=0
OBJECT_HALF = 0.02       # 4 cm cube
OBJECT_MASS = 0.05       # 50 g
DEFAULT_OBJECT_POS = (0.5, 0.0, OBJECT_HALF)

# UR5e home keyframe (the menagerie default)
UR5E_HOME = (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)

# Inspire "open hand" pose (all proximals + intermediates near 0)
INSPIRE_OPEN = np.zeros(12, dtype=np.float64)


def _patch_inspire_urdf(urdf_path: Path = INSPIRE_URDF_RIGHT) -> Path:
    """Apply the D-010 compiler injection so MuJoCo's URDF loader can find
    the meshes and skips unsupported .glb visuals. Writes the patched URDF
    to /tmp and returns its path."""
    txt = urdf_path.read_text()
    mesh_dir = str(urdf_path.parent.absolute()) + "/"
    inj = (
        f'<mujoco>'
        f'<compiler meshdir="{mesh_dir}" strippath="false" discardvisual="true"/>'
        f'</mujoco>'
    )
    patched = re.sub(r"(<robot[^>]*>)", r"\1" + inj, txt, count=1)
    out = Path("/tmp") / f"{urdf_path.stem}_patched.urdf"
    out.write_text(patched)
    return out


def build_scene_spec(
    inspire_urdf: Path = INSPIRE_URDF_RIGHT,
    object_pos: tuple[float, float, float] = DEFAULT_OBJECT_POS,
    camera_pos: tuple[float, float, float] = (-0.05, 0.05, 1.10),
) -> mujoco.MjSpec:
    """Assemble the merged UR5e + Inspire + table + object + camera scene
    via MjSpec API. Returns the unflushed `MjSpec` (call `.compile()` to
    get an `MjModel`)."""
    parent = mujoco.MjSpec.from_file(str(UR5E_SCENE))
    inspire_patched = _patch_inspire_urdf(inspire_urdf)
    child = mujoco.MjSpec.from_file(str(inspire_patched))

    attach_site = next((s for s in parent.sites if s.name == "attachment_site"), None)
    if attach_site is None:
        raise RuntimeError("UR5e scene is missing attachment_site")
    parent.attach(child, prefix="ins_", site=attach_site)

    wb = parent.worldbody

    # Table — top surface flush with z=0 (matches Stage 2 H2R `table_y→z=0`).
    table_body = wb.add_body(name="table", pos=[TABLE_X, TABLE_Y, -TABLE_HALF_Z])
    table_body.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[TABLE_HALF_X, TABLE_HALF_Y, TABLE_HALF_Z],
        rgba=[0.8, 0.6, 0.4, 1.0],
    )

    # Target object — 4 cm cube with a free joint (physics-driven)
    obj_body = wb.add_body(name="target_object", pos=list(object_pos))
    obj_body.add_freejoint()
    obj_body.add_geom(
        name="target_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[OBJECT_HALF] * 3,
        rgba=[0.1, 0.5, 0.9, 1.0],
        mass=OBJECT_MASS,
    )

    # Egocentric camera — auto-orients at the object via targetbody mode
    wb.add_camera(
        name="egocentric",
        pos=list(camera_pos),
        mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY,
        targetbody="target_object",
    )
    return parent


def build_scene_model(**kwargs) -> mujoco.MjModel:
    spec = build_scene_spec(**kwargs)
    return spec.compile()


# ───────────────────────────────────────────────────────────────────────────
# Env wrapper
# ───────────────────────────────────────────────────────────────────────────
@dataclass
class StepResult:
    obs: dict
    success: bool
    object_z: float
    done: bool


class EvalEnv:
    """Minimal kinematic-control eval environment for the 13-D ACT policy.

    The arm + Inspire target proximals are written directly to `data.qpos`
    each step (no torque actuators). The free-joint object is the only
    physics-driven body, and the hand → object interaction goes through
    mujoco's contact solver.

    The 7th action dimension (`gripper_1`) is currently a no-op — the
    Inspire DOFs already encode the full hand pose, and the dataset
    defines `gripper` as `mean(fingertip-to-wrist) > median` which is a
    redundant scalar. Pass-through for compatibility with the trained
    policy's output shape.
    """

    def __init__(
        self,
        episode_length: int = 200,
        success_lift_m: float = 0.05,
        img_h: int = 224,
        img_w: int = 224,
        seed: int = 0,
    ):
        self.episode_length = episode_length
        self.success_lift_m = success_lift_m
        self.img_h = img_h
        self.img_w = img_w
        self.rng = np.random.default_rng(seed)

        self.model = build_scene_model()
        self.data = mujoco.MjData(self.model)

        # Joint qpos addresses
        self.arm_qadr = np.array(
            [self._jnt_qadr(n) for n in UR5E_ARM_JOINT_NAMES], dtype=np.int64
        )
        self.inspire_target_qadr = np.array(
            [self._jnt_qadr(n) for n in INSPIRE_TARGET_JOINTS_PREFIXED], dtype=np.int64
        )

        # Body id of the target object (for success check + initial pose)
        self.obj_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_object"
        )
        self.obj_qadr = self.model.body_jntadr[self.obj_body_id]
        self.obj_qadr = self.model.jnt_qposadr[self.obj_qadr]

        # Single shared offscreen renderer (240x320 default mujoco fb is fine
        # for 224x224 since we crop/resize after).
        self._renderer = mujoco.Renderer(self.model, height=self.img_h, width=self.img_w)

        self._step = 0
        self._obj_init_z = 0.0

    def _jnt_qadr(self, name: str) -> int:
        jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jnt_id < 0:
            raise KeyError(f"joint not found: {name}")
        return int(self.model.jnt_qposadr[jnt_id])

    # ── Episode lifecycle ─────────────────────────────────────────────────
    def reset(self, object_pos: tuple[float, float, float] | None = None) -> dict:
        """Reset the env. If `object_pos` is None, sample uniformly inside
        a small workspace patch on the table top."""
        mujoco.mj_resetData(self.model, self.data)

        # UR5e at home
        self.data.qpos[self.arm_qadr] = np.array(UR5E_HOME)
        # Inspire fully open (12 non-dummy DOFs)
        # The 6 target + 6 mimic joints in URDF order are at qpos addresses
        # we don't enumerate explicitly; just set the 6 target DOFs to 0.
        self.data.qpos[self.inspire_target_qadr] = 0.0

        # Object pose: free joint = (x, y, z, qw, qx, qy, qz)
        if object_pos is None:
            x = float(self.rng.uniform(TABLE_X - 0.10, TABLE_X + 0.10))
            y = float(self.rng.uniform(TABLE_Y - 0.10, TABLE_Y + 0.10))
            z = OBJECT_HALF
        else:
            x, y, z = object_pos
        self.data.qpos[self.obj_qadr : self.obj_qadr + 7] = [x, y, z, 1, 0, 0, 0]
        self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        self._step = 0
        self._obj_init_z = float(self.data.xpos[self.obj_body_id][2])
        return self.get_obs()

    def step(self, action: np.ndarray) -> StepResult:
        """Apply 13-D action via direct qpos writes + step physics one tick."""
        if action.shape != (13,):
            raise ValueError(f"expected action shape (13,), got {action.shape}")

        # Arm: action[0:6] → UR5e joint angles
        self.data.qpos[self.arm_qadr] = action[:6]
        # Inspire target proximals: action[7:13]
        self.data.qpos[self.inspire_target_qadr] = action[7:13]
        # action[6] = gripper scalar — ignored, fingers fully describe pose

        # Step physics so the object can settle / be grasped / fall
        mujoco.mj_step(self.model, self.data)
        self._step += 1

        obj_z = float(self.data.xpos[self.obj_body_id][2])
        success = (obj_z - self._obj_init_z) > self.success_lift_m
        done = success or (self._step >= self.episode_length)

        return StepResult(
            obs=self.get_obs(),
            success=success,
            object_z=obj_z,
            done=done,
        )

    # ── Observation ───────────────────────────────────────────────────────
    def get_obs(self) -> dict:
        """Return the observation dict in the same shape the trained policy
        expects: float32 image (3, H, W) in [0, 1] + 13-D state."""
        self._renderer.update_scene(self.data, camera="egocentric")
        rgb = self._renderer.render()                              # (H, W, 3) uint8
        img = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0    # (3, H, W) float32

        state = np.zeros(13, dtype=np.float32)
        state[:6] = self.data.qpos[self.arm_qadr]
        state[6] = 0.0  # gripper scalar — not tracked here
        state[7:13] = self.data.qpos[self.inspire_target_qadr]

        return {
            "observation.image": img,
            "observation.state": state,
        }

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
