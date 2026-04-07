# MimicDreamer Replication Plan
### Dataset: EgoDex (Apple, 829h) | Focus: Dexterous Manipulation

---

## Context & Goals


The plan has **4 stages**, ordered by dependency. Each stage produces a concrete artifact.

**Why EgoDex over FIVER for this project:**
- Camera intrinsics + extrinsics provided at every frame (30Hz) — no approximation needed
  for human-to-robot frame alignment
- 25-joint finger tracking from Apple ARKit SLAM — production-grade, not estimated post-hoc
- 194 tasks with large, varied joint excursions — the trajectory variance problem from
  FIVER v1 is solved by dataset design, not by hacks
- MimicDreamer itself used EgoDex for all experiments — results are directly comparable
- 829 hours / 338,000 episodes — enough to run meaningful scaling experiments

---

## Stage 0: Foundation + Data Exploration (1 week)
**Goal**: Fill conceptual gaps AND get hands on the data before writing pipeline code.

You already know: transformers, PPO, GRPO, SFT, flow matching, pi0 architecture,
behavior cloning. You still need to internalize three things:

### 0.1 — Homography & Camera Projection
EgoStabilizer is built entirely on homography. A homography is a 3x3 matrix that maps
one plane to another under projective transformation. When a camera rotates (but doesn't
translate) between frames, all point correspondences satisfy a homography — which is why
it works for stabilizing a mostly-static workspace viewed from a head-mounted camera.

**Resources:**
- [OpenCV Homography Tutorial](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html)
  — read "Basic Concepts", skip the AR sections
- [LearnOpenCV Video Stabilization](https://learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/)
  — this is the exact algorithm EgoStabilizer implements

### 0.2 — DLS Inverse Kinematics with Smoothness
MimicDreamer's IK uses Damped Least Squares (DLS):
  Dq = Jt(JJt + u^2 I)^-1 e  +  lambda * ||q - q_{t-1}||^2

The first term moves the end-effector toward the target. Damping u prevents joint
velocity blowup near singularities. The smoothness term penalizes large jumps between
timesteps — this is what was missing in FIVER v1.

**Resources:**
- [mink GitHub](https://github.com/kevinzakka/mink) — your IK library; read README + arm_ur5e.py example
- [Pink (mink's parent)](https://github.com/stephane-caron/pink) — better documented, same API concept

### 0.3 — EgoDex Data Format (do this on Day 1)
The full dataset is 2TB. Start with just the test set (~7 hours, smallest download):

```bash
curl "https://ml-site.cdn-apple.com/datasets/egodex/test.zip" -o test.zip
unzip test.zip

# Also grab basic_pick_place from Part 2 for initial policy training
curl "https://ml-site.cdn-apple.com/datasets/egodex/part2.zip" -o part2.zip
```

The HDF5 structure per episode:

```
episode_N.hdf5
├── camera/
│   ├── intrinsic          # (3, 3) — same for every file in the dataset
│   └── extrinsic          # (T, 4, 4) — camera pose in world frame at each frame
├── hand/
│   ├── left/
│   │   ├── joints         # (T, 25, 3) — 3D position of all 25 joints, left hand
│   │   ├── wrist_pose     # (T, 4, 4) — wrist SE(3) transform
│   │   └── confidence     # (T,) — ARKit tracking confidence per frame
│   └── right/
│       └── ...            # same structure
├── language/
│   └── annotation         # string — natural language task description
└── (paired MP4 at same path with same index)
```

Write a quick exploration script on Day 1:

```python
import h5py
import numpy as np

with h5py.File('test/basic_pick_place/0.hdf5', 'r') as f:
    joints_right = f['hand/right/joints'][:]      # (T, 25, 3)
    wrist_right  = f['hand/right/wrist_pose'][:]  # (T, 4, 4)
    conf         = f['hand/right/confidence'][:]  # (T,)
    K            = f['camera/intrinsic'][:]        # (3, 3)
    extrinsics   = f['camera/extrinsic'][:]        # (T, 4, 4)

    wrist_pos = wrist_right[:, :3, 3]  # translation column
    print(f"Episode length: {joints_right.shape[0]} frames")
    print(f"Wrist x range: {wrist_pos[:,0].min():.3f} to {wrist_pos[:,0].max():.3f} m")
    print(f"Wrist y range: {wrist_pos[:,1].min():.3f} to {wrist_pos[:,1].max():.3f} m")
    print(f"Wrist z range: {wrist_pos[:,2].min():.3f} to {wrist_pos[:,2].max():.3f} m")
    print(f"Mean tracking confidence: {conf.mean():.3f}")
```

You should see wrist ranges of 0.2–0.5m. This confirms EgoDex doesn't have the
83->83.2 low-variance problem from FIVER.

**Resources:**
- [EgoDex GitHub (Apple)](https://github.com/apple/ml-egodex) — official toolkit + download links
- [EgoDex paper](https://arxiv.org/abs/2505.11709) — read Section 3 (Data Collection) and
  Section 4 (Benchmarks) to understand what the dataset was designed for

**Deliverable**: exploration notebook confirming data structure + variance report.

---

## Stage 1: EgoStabilizer (1.5 weeks)
**Goal**: Implement a working video stabilizer for EgoDex egocentric footage.

EgoDex is collected with Apple Vision Pro — much more stable than factory egocentric
footage. But stabilization still matters: head micro-movements create background drift,
and the stabilized canonical view is what you feed as policy observations.

**Key advantage over FIVER**: EgoDex provides camera extrinsics at every frame
(camera/extrinsic in HDF5). This means you can do exact homography estimation from
known camera poses rather than estimating from feature matching. This is strictly better.

### 1.1 — Exact Homography from Known Extrinsics

```python
import numpy as np
import cv2

def compute_homography_from_extrinsics(K, T1, T2, table_dist=0.5):
    """
    K:          (3,3) camera intrinsics
    T1, T2:     (4,4) camera extrinsics (world-to-cam) at frames t and t+1
    table_dist: approximate distance to table plane in meters

    For a planar scene: H = K (R - t n^T / d) K^-1
    """
    R_rel = T2[:3, :3] @ T1[:3, :3].T
    t_rel = T2[:3, 3] - R_rel @ T1[:3, 3]
    n = np.array([0., 0., 1.])  # table normal (z-up)
    H = K @ (R_rel - np.outer(t_rel, n) / table_dist) @ np.linalg.inv(K)
    return H

def stabilize_egodex_clip(frames, K, extrinsics):
    """
    frames:     list of (H, W, 3) uint8 images
    K:          (3,3) intrinsics
    extrinsics: (T, 4, 4) camera poses
    Returns:    list of stabilized frames
    """
    stabilized = []
    for t in range(1, len(frames)):
        H = compute_homography_from_extrinsics(K, extrinsics[t-1], extrinsics[t])
        warped = cv2.warpPerspective(
            frames[t], H, (frames[t].shape[1], frames[t].shape[0])
        )
        stabilized.append(warped)
    return stabilized
```

### 1.2 — Fallback: Feature-Based Stabilization
For clips with low tracking confidence, fall back to RANSAC-based estimation:

```bash
pip install vidstab opencv-python
```

```python
from vidstab import VidStab

def choose_method(confidence, threshold=0.8):
    return 'exact' if confidence.mean() > threshold else 'ransac'

# RANSAC fallback
stabilizer = VidStab(kp_method='GFTT')
stabilizer.stabilize(
    input_path='egodex_clip.mp4',
    output_path='stabilized_clip.mp4',
    border_type='black',
    smoothing_window=30
)
```

**Resources:**
- [python_video_stab GitHub](https://github.com/AdamSpannbauer/python_video_stab)
- [Nghia Ho's original article](https://nghiaho.com/?p=2093)

### 1.3 — Inpainting Black Borders
After warping, borders appear. Start with OpenCV; upgrade to ProPainter if needed:

```python
mask = (frame == 0).all(axis=2).astype(np.uint8) * 255
inpainted = cv2.inpaint(frame, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
```

- [ProPainter GitHub](https://github.com/sczhou/ProPainter)

### 1.4 — Quantitative Evaluation
Reproduce MimicDreamer Table 2 metrics (defined in Appendix B.4):
- **Stability**: mean per-frame viewpoint angle change (lower is better)
- **Jitter RMS**: high-frequency residual after low-pass filtering the path
- **H-RMSE**: homography reprojection error (geometric consistency)

**Deliverable**: `egostabilizer.py` — takes EgoDex HDF5 + MP4, outputs stabilized
video + three stability metrics. Handles both exact and RANSAC paths.

---

## Stage 2: Action Alignment — Human Hand to Robot Joint Angles (2 weeks)
**Goal**: Convert EgoDex 3D wrist + finger poses into smoothed robot joint angle sequences.

This is MimicDreamer's "Actions" branch. With EgoDex the frame alignment problem is
almost entirely solved — you have calibrated camera poses and 3D wrist poses already
in world coordinates. No approximation needed.

### 2.1 — Extract Wrist Trajectories

```python
import h5py
import numpy as np

def extract_wrist_trajectory(hdf5_path, hand='right', min_confidence=0.5):
    """
    Returns wrist poses in world frame, filtered by confidence.
    Output: (T, 4, 4) SE(3) transforms
    """
    with h5py.File(hdf5_path, 'r') as f:
        wrist_poses = f[f'hand/{hand}/wrist_pose'][:]  # (T, 4, 4) in camera frame
        confidence  = f[f'hand/{hand}/confidence'][:]  # (T,)
        extrinsics  = f['camera/extrinsic'][:]          # (T, 4, 4) world-to-cam

    wrist_world = []
    for t in range(len(wrist_poses)):
        if confidence[t] > min_confidence:
            T_world_cam = np.linalg.inv(extrinsics[t])
            T_world_wrist = T_world_cam @ wrist_poses[t]
        else:
            T_world_wrist = wrist_world[-1] if wrist_world else np.eye(4)
        wrist_world.append(T_world_wrist)

    return np.array(wrist_world)
```

### 2.2 — Human-to-Robot Frame Alignment

```python
def estimate_h2r_transform(wrist_trajectory_world):
    """
    Estimate rigid transform from EgoDex world frame to robot base frame.
    Uses table plane (lowest wrist positions) as reference.
    """
    wrist_z = wrist_trajectory_world[:, 2, 3]
    table_z = np.percentile(wrist_z, 5)  # 5th percentile ~ table height

    # Robot arm reach ~0.6m, typical human reach ~0.5m
    scale = 0.6 / 0.5
    t_HR = np.array([0, 0, -table_z])  # shift table to z=0
    R_HR = np.eye(3)                   # EgoDex z-up matches robot convention

    return R_HR, t_HR, scale

def apply_h2r_transform(wrist_poses_world, R_HR, t_HR, scale):
    robot_targets = []
    for T in wrist_poses_world:
        p = T[:3, 3]
        R = T[:3, :3]
        p_robot = scale * (R_HR @ p + t_HR)
        R_robot = R_HR @ R
        robot_targets.append((p_robot, R_robot))
    return robot_targets
```

### 2.3 — IK with Temporal Smoothness

```python
import mink

def solve_ik_trajectory(model, robot_targets, lambda_smooth=0.1, dt=1/30.0):
    """
    robot_targets: list of (p, R) tuples — EE targets in robot base frame
    Returns: (T, num_joints) joint angle trajectory
    """
    configuration = mink.Configuration(model)
    joint_angles = []

    for p_target, R_target in robot_targets:
        ee_task = mink.FrameTask(
            frame_name="end_effector",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.3,  # lower weight — wrist orientation is noisier
        )
        target_pose = mink.SE3.from_rotation_and_translation(R_target, p_target)
        ee_task.set_target(target_pose)

        # Smoothness: penalize deviation from previous config
        posture_task = mink.PostureTask(cost=lambda_smooth)
        posture_task.set_target_from_configuration(configuration)

        vel = mink.solve_ik(
            configuration,
            tasks=[ee_task, posture_task],
            dt=dt,
            solver="quadprog",
        )
        configuration.integrate_inplace(vel, dt)
        joint_angles.append(configuration.q.copy())

    return np.array(joint_angles)  # (T, num_joints)
```

### 2.4 — Gripper State from 25-Joint Finger Data
EgoDex gives raw finger joint positions — compute hand openness from fingertip spread:

```python
from scipy.ndimage import median_filter

FINGERTIP_IDS = [4, 8, 12, 16, 20]  # EgoDex 25-joint layout
PALM_ID = 0

def compute_hand_openness(finger_joints):
    """finger_joints: (25, 3). Returns scalar in [0,1]."""
    tips = finger_joints[FINGERTIP_IDS]       # (5, 3)
    palm = finger_joints[PALM_ID]              # (3,)
    distances = np.linalg.norm(tips - palm, axis=1)
    return float(np.clip(distances.mean() / 0.12, 0, 1))

def infer_gripper_state(finger_joints_seq, threshold=0.4, window=5):
    """finger_joints_seq: (T, 25, 3). Returns (T,) binary states."""
    openness = [compute_hand_openness(fj) for fj in finger_joints_seq]
    binary = (np.array(openness) > threshold).astype(float)
    return median_filter(binary, size=window)
```

### 2.5 — Variance Sanity Check

```python
def variance_report(joint_angles, label=""):
    ranges = joint_angles.max(axis=0) - joint_angles.min(axis=0)
    stds   = joint_angles.std(axis=0)
    print(f"\n{label}")
    print(f"  Per-joint ranges (rad): {np.round(ranges, 3)}")
    print(f"  Joints with range > 0.3 rad: {(ranges > 0.3).sum()}/{len(ranges)}")
    # Target: most primary joints should have range > 0.3 rad
```

**Deliverable**: `action_alignment.py` — EgoDex HDF5 -> joint angles + gripper states
in LeRobot-compatible format.

---

## Stage 3: Dexterous Finger Retargeting (1.5 weeks)
**Goal**: Retarget EgoDex 25-joint finger poses to a robot dexterous hand.

### 3.1 — Why Finger Retargeting Is Non-Trivial
Human finger kinematics don't map cleanly to robot hands because joint ranges, DOF
counts, and thumb kinematics all differ by embodiment. You need to retarget *relative
joint angles*, not absolute positions.

### 3.2 — dex-retargeting Library

```bash
pip install dex-retargeting
```

```python
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting

# Configure for your target robot hand (check if Inspire is in their assets)
config = RetargetingConfig.from_file('configs/inspire_hand.yml')
retargeter = SeqRetargeting.from_config(config)

def retarget_finger_sequence(finger_joints_seq):
    """
    finger_joints_seq: (T, 25, 3) EgoDex finger positions
    Returns: (T, N_robot_dof) robot finger joint angles
    """
    return np.array([retargeter.retarget(fj) for fj in finger_joints_seq])
```

**Resources:**
- [dex-retargeting GitHub](https://github.com/dexsuite/dex-retargeting) — PKU's library,
  used by multiple dexterous manipulation papers
- Check their assets folder for Inspire hand URDF support

### 3.3 — Combined Action Vector

```python
# Full action: [arm_joints (6), gripper (1), finger_joints (N)]
# For Inspire hand: N = 6 dexterous DOF
action = np.concatenate([arm_joints, gripper[:, None], finger_joints], axis=1)
# Shape: (T, 13)
```

**Deliverable**: `finger_retargeting.py` — EgoDex 25-joint sequences to robot finger
angles. Visualize in MuJoCo to verify the retargeted motion looks correct.

---

## Stage 4: Policy Training & Evaluation (1 week)
**Goal**: Train a policy and produce an honest ablation table.

### 4.1 — Data Format for LeRobot
EgoDex is 1080p — resize to 224x224 before training:

```python
# Data schema:
# - observation.images.ego: (T, 3, 224, 224) stabilized egocentric frames
# - action: (T, 7) arm-only OR (T, 13) arm + dexterous hand
# - task_description: str from EgoDex language annotation
# - task_index: int
```

**Resources:**
- [LeRobot dataset format](https://github.com/huggingface/lerobot/blob/main/examples/1_load_lerobot_dataset.py)

### 4.2 — Start with `basic_pick_place`
Don't train on all 194 tasks. Start here:
- Most diverse trajectories within a single category
- Best language annotation quality
- Large wrist range of motion (full pick + place arm extension)
- Easy to build a matching MuJoCo evaluation env

### 4.3 — Train with Action Chunking

```bash
python lerobot/scripts/train.py \
  policy=act \
  dataset_repo_id=your_egodex_processed \
  policy.chunk_size=10 \
  policy.n_action_steps=8 \
  training.batch_size=32 \
  training.num_workers=4
```

### 4.4 — Honest Evaluation Protocol
Roll out in MuJoCo simulation, 100 episodes, pick-and-place matching EgoDex setup.
Report success rate (object within 5cm of goal) + mean trajectory smoothness.

**Ablation table — four conditions:**

| Condition | Stabilization | Smooth IK | Finger Retargeting |
|-----------|:---:|:---:|:---:|
| FIVER v1 baseline | - | - | - |
| EgoDex data only  | + | - | - |
| + Smooth IK       | + | + | - |
| Full pipeline     | + | + | + |

This cleanly shows: EgoDex data fixes variance; smooth IK fixes jitter; finger
retargeting improves contact-critical subtasks.

**Deliverable**: Checkpoint + ablation table + 2-page write-up as partial MimicDreamer
replication with dexterous extension.

---


---

## Timeline

| Week | Stage | Deliverable |
|------|-------|-------------|
| 1 | Stage 0 | Data exploration notebook + variance report + concepts read |
| 2–3 | Stage 1 | `egostabilizer.py` + stability metrics on EgoDex clips |
| 4–5 | Stage 2 | `action_alignment.py` + joint angles + variance check |
| 6 | Stage 3 | `finger_retargeting.py` + MuJoCo visualization |
| 7–8 | Stage 4 | Trained policy + ablation table + write-up |

---

## Key Papers + Resources

| Resource | Why |
|----------|-----|
| [MimicDreamer](https://arxiv.org/abs/2509.22199) | Your replication target |
| [EgoDex paper](https://arxiv.org/abs/2505.11709) | Read Sections 3 + 4 |
| [EgoDex GitHub](https://github.com/apple/ml-egodex) | Download + HDF5 format docs |
| [EgoMimic](https://egomimic.github.io/) | Co-training strategy reference |
| [EgoVLA](https://arxiv.org/abs/2507.12440) | IK retargeting + VLA fine-tuning |
| [dex-retargeting](https://github.com/dexsuite/dex-retargeting) | Finger retargeting library |
| [mink](https://github.com/kevinzakka/mink) | IK solver |
| [OpenEgo](https://arxiv.org/abs/2509.05513) | Consolidates EgoDex + 5 others; useful for scale |
| [ProPainter](https://github.com/sczhou/ProPainter) | Video inpainting for stabilizer borders |
| [LeRobot](https://github.com/huggingface/lerobot) | Policy training framework |

---

## The One Thing to Always Remember

> EgoDex gives you intrinsics, extrinsics, 25-joint finger tracking, and ARKit
> confidence scores for free — things you had to approximate or skip with FIVER.
> Use all of them. The quality of your input data determines the ceiling of your policy.
