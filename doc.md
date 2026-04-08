# doc.md — EgoDex reference for this project

A consolidated, cite-able reference for everything we have learned about the
EgoDex test split that the rest of the pipeline (Stages 1–4) needs to know.
This file is **derived from data**, not from `initial_plan.md`. Where the
two disagree, this file is correct (the disagreements are documented inline).

If you change something here, mirror the change into `plan.md` under the
matching D-/R- decision so future sessions don't drift.

---

## 0. Scope decision

**We use only the EgoDex `test` split.** No `part1.zip`, no `part2.zip`, no
streaming downloads. See `plan.md` D-005 for the rationale. Concrete
implications:

- All Stage 1–4 code reads from `data/test/<task>/`.
- Policy training (Stage 4) is on `data/test/basic_pick_place/` — 277
  episodes, ~17.9 minutes of 30 Hz footage. The `initial_plan.md` §4.2
  reference to "Part 2 download for training" is obsolete.
- Anywhere the original plan said "test split for development, Part 2 for
  training", read "test split for both".

---

## 1. Disk layout (after `unzip data/test.zip` into `data/`)

```
data/
├── test.zip                       # 16,114,529,397 bytes (16.1 GB)
└── test/                          # extracted, 6,598 file entries
    ├── add_remove_lid/
    │   ├── 0.hdf5
    │   ├── 0.mp4
    │   ├── 1.hdf5
    │   └── ...
    ├── basic_pick_place/          # 277 episodes (= 555 files: hdf5 + mp4)
    └── ...                        # 111 task subdirectories total
```

- 111 task categories. Names range from `basic_pick_place` to
  `assemble_disassemble_furniture_bench_lamp` to `tie_and_untie_shoelace`.
  Full list captured in
  `logs/runs/2026-04-07_105200_unzip_test.log` (or rerun
  `notebooks/01_variance_report.py` after pointing it at `data/test`).
- Each episode is a paired `.hdf5` + `.mp4`. The MP4 is the egocentric
  video the AVP recorded; the HDF5 is the parallel ARKit metadata.
- Sample episode size: `data/test/basic_pick_place/0.hdf5` is 0.65 MB.
  HDF5 files are small because they contain only pose tracks, not video.

---

## 2. The HDF5 schema (real, not the one in `initial_plan.md`)

`initial_plan.md` §0.3 documents a schema with `/hand/{left,right}/joints` and
`/language/annotation`. **That schema does not exist in the actual data.** The
real layout, verified on `data/test/basic_pick_place/0.hdf5` and confirmed
across all 277 `basic_pick_place` episodes, is:

```
episode_N.hdf5
├── camera/
│   └── intrinsic                                (3, 3)   float32
├── transforms/                                  # 69 datasets, all (T, 4, 4) float32
│   ├── camera                                   # camera-to-world per frame (see §3)
│   ├── hip                                      # body kinematic chain
│   ├── spine1 .. spine7
│   ├── neck1  .. neck4
│   ├── leftShoulder, leftArm, leftForearm, leftHand
│   ├── leftThumbKnuckle, leftThumbIntermediateBase,
│   │   leftThumbIntermediateTip, leftThumbTip                  # 4 thumb joints
│   ├── leftIndexFingerMetacarpal, leftIndexFingerKnuckle,
│   │   leftIndexFingerIntermediateBase,
│   │   leftIndexFingerIntermediateTip, leftIndexFingerTip      # 5 per finger
│   ├── leftMiddleFinger…  (5)
│   ├── leftRingFinger…    (5)
│   ├── leftLittleFinger…  (5)
│   └── (mirrored right* for the right side)
└── confidences/                                 # 68 datasets, all (T,) float32
    └── <jointName>                              # one per transform except `camera`
```

**File-level HDF5 attributes** carry the language metadata that
`initial_plan.md` §0.3 expected at `/language/annotation`. From
`basic_pick_place/0.hdf5`:

```
annotated         True
annotator_version 0.3
description       'pick up a black stapler from the table and place it in the box lid.'
environment       'table:metal, position:sitting, background:white,
                   from:table, to:box lid, hand:right'
extra             'C6'
llm_description   'Pick up a black stapler from the metal table and place it in the box lid.'
llm_description2  'None'
llm_objects       array(['stapler'], dtype=object)
llm_type          'reset'
llm_verbs         array(['pick', 'place'], dtype=object)
object            'object:stapler, color:black'
session_name      '2025-03-27_16-14-11.mov'
task              'basic_pick_place'
type              'reset'
```

The `environment` attribute often encodes which hand is active
(`hand:right` / `hand:left` / `hand:bimanual`) — useful as a label, but
inferring from wrist excursion (see §6) is more reliable when present.

### 2.1 — 25 joints per hand

EgoDex matches the "25-joint hand" claim from the paper, just unpacked into
top-level datasets. Per side:

- 1 wrist: `<side>Hand`
- 4 thumb joints (no Metacarpal — ARKit's thumb model doesn't have one):
  `ThumbKnuckle`, `ThumbIntermediateBase`, `ThumbIntermediateTip`, `ThumbTip`
- 4 fingers (Index, Middle, Ring, Little) × 5 joints each:
  `Metacarpal`, `Knuckle`, `IntermediateBase`, `IntermediateTip`, `Tip`

That is `1 + 4 + 4*5 = 25`. Helper:
`notebooks/00_explore_egodex.py::hand_joint_names("left" | "right")`.

### 2.2 — Frame counts

From the 277-episode `basic_pick_place` aggregate:

| stat | frames | seconds @ 30 Hz |
|------|--------|-----------------|
| min  | 35     | 1.17 s |
| median | 98   | 3.27 s |
| max  | 483    | 16.10 s |
| total | 32,271 | 17.93 min |

So the `basic_pick_place` test split is ~18 minutes of footage. Small. Plan
the policy ablation accordingly (see §8).

---

## 3. Coordinate convention (the gotcha)

**ARKit / EgoDex world frame is y-up.** `initial_plan.md` §1.1 hard-coded
`n = [0, 0, 1]` (z-up); that is wrong on this dataset.

Empirical verification on `basic_pick_place/0.hdf5`:

| quantity | x range | y range | z range |
|----------|---------|---------|---------|
| camera position | [-0.121, -0.052] | **[1.052, 1.091]** | [-0.421, -0.387] |
| right wrist position | [-0.062, 0.399] | **[0.803, 0.984]** | [-0.794, -0.619] |

The camera y is essentially constant near 1.07 m (head height while seated),
and the wrist y varies between 0.80 and 0.98 m (below head height, swinging
through pick-and-place). The only axis on which "camera is consistently
above wrist" is y → y is up.

### 3.1 — `transforms/camera` is camera-to-world  *(confirmed 2026-04-08)*

`transforms/camera[t]` = `T_world_cam_t`, i.e. `X_world = T @ X_cam`.

Empirical confirmation on `basic_pick_place/0.hdf5` (verification log:
`logs/runs/2026-04-08_*_verify_camera_convention.log`):

- translation column mean = `(-0.077, 1.069, -0.395)`, std `(0.022, 0.015, 0.010)`
  over 126 frames. The y component sits at seated AVP head height (~1.07 m)
  and is essentially constant. World-to-camera would not put a head-height
  point in the translation column.
- cam y-mean (1.069) > active right-wrist y-mean (0.877) — head above hand,
  consistent with cam-to-world.
- `det(R) = +1`, `R R^T = I` to machine precision, so the rotation block is
  a proper rotation (not a flip).

**To go to camera-frame**: invert. For an SE(3) `T = [[R, t], [0, 1]]`, the
inverse is `T_inv = [[R^T, -R^T t], [0, 1]]` — cheap, no `np.linalg.inv`
needed.

The Stage 1 EgoStabilizer relies on this convention; cross-referenced as
`plan.md` R-003.

### 3.2 — Plane normals in the y-up frame

- Table normal in **world** frame: `n_world = (0, 1, 0)`.
- Table normal in **camera-1** frame for the homography formula: take the
  rotation part of the world-to-camera-1 transform and apply it to
  `n_world`. With camera-to-world `T_w2c_inv = T_c2w`, that means
  `n_cam = T_c2w[:3, :3].T @ n_world` which is just the second *row* of
  `T_c2w[:3, :3]` (i.e. `T_c2w[1, :3]`).

---

## 4. Variance report — Stage 0 deliverable

`notebooks/01_variance_report.py`, full output at
`logs/runs/2026-04-07_110200_variance_report_basic_pick_place.log`,
CSV at `outputs/variance_report_basic_pick_place.csv`.

**Across 277 `basic_pick_place` test episodes**, taking the active hand
(whichever has the larger summed wrist excursion):

| metric | value |
|--------|-------|
| max-axis range, mean | **0.260 m** |
| max-axis range, median | 0.251 m |
| max-axis range, min | 0.074 m |
| max-axis range, max | 0.523 m |
| sum-of-axes range, mean | 0.566 m |
| sum-of-axes range, median | 0.592 m |
| episodes with max-axis < 0.05 m (FIVER v1 collapse signature) | **0 / 277 (0.0%)** |
| episodes with max-axis ∈ [0.20, 0.50] m (plan target window) | 195 / 277 (70.4%) |
| active hand split | 221 right, 56 left |

**Conclusion**: the FIVER v1 low-variance failure mode does not exist on
EgoDex. The dataset is fit for purpose. Two-handed episodes are not
detected by this metric (would need a different test) and are believed to
be rare in `basic_pick_place`.

---

## 5. ARKit confidence calibration (resolves `plan.md` R-002)

`notebooks/02_calibrate_open_questions.py`, full output at
`logs/runs/2026-04-07_111200_calibrate_open_questions_v2.log`,
CSV at `outputs/calibration_basic_pick_place.csv`.

Sample size: 806,775 confidence values (25 active-hand joints × all frames
× 277 episodes). The distribution is **bimodal** in [0, 1]:

- **Floor mode** (untracked): a sharp spike at 0.000–0.003. 17.88% of all
  samples sit here, 15.58% at the single value `0.0032`. This is ARKit's
  "no signal" default for joints it failed to track that frame.
- **Tracked mode**: broad, ~0.2 to 1.0, p25=0.257, median=0.614, p75=0.816,
  p95=0.940.

| stat | value |
|------|-------|
| min  | 0.000 |
| p01  | 0.000 |
| p05  | 0.003 |
| p10  | 0.003 |
| p25  | 0.257 |
| p50  | 0.614 |
| p75  | 0.816 |
| p90  | 0.895 |
| p95  | 0.940 |
| p99  | 0.996 |
| max  | 1.000 |
| mean | 0.529 |
| std  | 0.327 |

**Threshold sweep** (fraction of samples kept above threshold):

| threshold | % kept | use case |
|-----------|--------|----------|
| `> 0.1`   | 82.0%  | **"is this joint tracked at all" — recommended hard reject filter** |
| `> 0.2`   | 78.8%  | slightly stricter floor cut |
| `> 0.3`   | 72.0%  |  |
| `> 0.5`   | 58.0%  | original `initial_plan.md` §2.1 — too aggressive |
| `> 0.7`   | 42.0%  |  |
| `> 0.8`   | 27.9%  | "high quality" — too strict for production filter |
| `> 0.9`   |  9.3%  | research-grade only |

### 5.1 — Recalibrated thresholds we will use

| name | value | semantics | where used |
|------|-------|-----------|------------|
| `CONF_TRACKED` | **0.10** | hard floor — reject anything below | every per-joint extractor in Stage 1/2 |
| `CONF_HIGH` | **0.50** | "high quality" — soft, used as a *weight* | confidence-aware IK weighting (contribution #1) |

Two extra rules:

1. **Compute thresholds per joint, not per hand.** The wrist tracks better
   than distal fingertips; a hand-mean would over-penalize "good wrist +
   flaky pinky" frames.
2. **Treat confidence as a weight, not a switch, where you can.** For IK,
   pass `w_j = max(0, conf_j - CONF_TRACKED) / (1 - CONF_TRACKED)` as a
   per-joint task cost multiplier. That gives the contribution-#1 ablation
   for free without writing a separate path.

---

## 6. Camera-to-table distance (resolves `plan.md` R-001)

Same script + same log as §5. With y-up applied:

| stat (per-episode mean) | value |
|-------------------------|-------|
| median | **0.243 m** |
| p10 | 0.171 m |
| p90 | 0.312 m |
| min over episodes | 0.112 m |
| max over episodes | 0.718 m |
| std over episodes | 0.068 m |
| % within ±20% of 0.5 m | **0.4%** |

The `initial_plan.md` §1.1 hard-coded `table_dist = 0.5` is wrong by
roughly 2× on `basic_pick_place`.

### 6.1 — What to do instead

Don't use a constant. The data has per-frame camera poses, so compute
the distance exactly per frame:

```python
# Once per episode
active = "right" if right_wrist_total > left_wrist_total else "left"
wrist_y = f[f"transforms/{active}Hand"][:, 1, 3]
table_y = float(np.percentile(wrist_y, 5))    # estimate of table surface
# Optional: subtract a small offset (~0.05 m) for the wrist-above-fingertips
# bias. For stabilization the bias mostly washes out, so leave it 0 by default.

# Per frame (vectorized)
cam_y = f["transforms/camera"][:, 1, 3]       # (T,)
d_t = np.abs(cam_y - table_y)                  # (T,)
```

`d_t` then plugs into the §1.1 homography formula `H = K (R - t n^T / d) K^-1`
with `n` = the camera-frame table normal from §3.2.

### 6.2 — Active-hand selection

In `basic_pick_place`, 221/277 episodes are right-handed and 56/277 are
left-handed (per the variance report's "larger total wrist excursion"
rule). The `environment` HDF5 attribute often spells this out
(`hand:right` etc.) but not always — fall back to the excursion rule.

---

## 7. Fingertip spread (Stage 2 hand-openness proxy preview)

Just a snapshot from `basic_pick_place/0.hdf5`:

- right hand fingertip-to-wrist mean spread: 0.123 → 0.151 m, range 28 mm
  over the 4.2-second episode
- left hand (idle): 0.129 → 0.131 m, range 2 mm

This is consistent with §2.4 of `initial_plan.md`'s plan to use mean
fingertip spread as the binary gripper signal. Threshold around the median
of spread on the active hand, then median-filter. Don't lock the absolute
threshold yet — calibrate from a per-task histogram in Stage 2.

---

## 8. Implications for Stage 4 (policy training)

With test-split-only data:

- 277 `basic_pick_place` episodes is **small** for behavior cloning. Don't
  expect strong absolute success rates from `lerobot policy=act` out of
  the box. Plan to over-fit gracefully and report the **deltas** in the
  ablation table, not absolute numbers.
- 17.9 minutes of footage is also small for the §4 scaling ablation
  (10/50/100/500/1000 episodes). The 500/1000 buckets aren't reachable on
  `basic_pick_place` test alone — drop them or use other test-split tasks
  that share verbs (`pick_place_food`, `vertical_pick_place`, etc.) as
  augmentation.
- Train/eval split: hold out 20% of episodes (~55) for evaluation. No
  separate val set is necessary at this scale.

---

## 8.5 Stage 1 EgoStabilizer — geometry & first results

`src/mimicdreamer_egodex/egostabilizer.py`. Plane-induced homography from
`transforms/camera`, warping every frame to a fixed reference (default frame 0).

### 8.5.1 The formula (sign matters — `plan.md` R-004)

For the world plane y = `table_y` (y-up), with `T_src`, `T_dst` 4x4
camera-to-world transforms and `K` the shared intrinsic:

```
R_ds   = R_wd^T R_ws                       # dst-from-src rotation
t_ds   = R_wd^T (t_ws - t_wd)              # dst-from-src translation
n_s    = R_ws^T (0, 1, 0)                  # plane normal in src cam frame
d_s    = table_y - cam_y_world[src]        # signed; negative when cam above
H_src→dst = K (R_ds + t_ds n_s^T / d_s) K^-1
```

The **plus sign** is correct for our (src→dst, y-up, cam-to-world) convention.
The minus form that appears in some references uses the opposite ordering
or an inward-pointing normal — see R-004 for the derivation. Apply
`cv2.warpPerspective(frame_src, H, (W, H))` to get a frame that aligns with
the dst camera's view of the table plane.

### 8.5.2 Metric definitions

- **`raw_mean_interframe_camera_angle_deg`**: mean magnitude (deg) of camera
  rotation between consecutive `transforms/camera` frames. Tells you how
  jittery the head was. EgoDex AVP runs are typically 0.3–0.5°/frame.
- **`raw_mean_interframe_pixel_disp`** / **`stab_mean_interframe_pixel_disp`**:
  mean ORB-feature displacement between consecutive frames, computed on raw
  vs. warped frames. The `pixel_disp_reduction_ratio = raw / stab` is the
  headline stabilization-quality number.
- **`homography_rmse_px_median`**: inlier-filtered (best 50% of per-pair
  matches) reprojection error of ORB matches between (ref, t) frames under
  our geometric homography. **Without inlier filtering this metric is
  meaningless on EgoDex** — most ORB features land on the moving hand and
  the manipulated object, which no plane-induced homography can predict.
  Single-digit pixels = the planar background is correctly tracked.

### 8.5.3 First-cut results

| episode | frames | raw disp (px) | stab disp (px) | reduction | inlier H-RMSE (px) |
|---------|------:|---------:|---------:|---------:|---------:|
| `basic_pick_place/0` | 126 | 6.05 | 0.92 | **6.6×** | 1.49 |
| `basic_pick_place/1` | 160 | 8.98 | 2.11 | **4.25×** | 5.45 |

Episode 0 is mostly stationary (camera std 1–2 cm); episode 1 has the most
camera motion of the first 80 episodes (~24 cm range). Both stabilize cleanly.

### 8.5.4 RANSAC fallback  *(updated 2026-04-08 — see `plan.md` R-006)*

`vidstab` is installed (Stage 1 dep) but not yet wired in — the fallback
path currently writes raw frames unchanged. **Original claim was that no
episode in the test split would trigger it; that was wrong.** The full
277-episode batch identified **6 episodes** where the active-hand wrist
had fewer than 5 frames above `CONF_TRACKED = 0.10`, so they dropped to
the stub:

    [15, 26, 102, 131, 192, 251]   (5 of 6 are left-hand episodes)

All 6 report a `pixel_disp_reduction_ratio` of exactly `1.00×` because the
stub is the identity warp. None of them overlap with the Stage 2 IK tail
(§8.6.8), so their impact is localized. We will wire in real vidstab only
if Stage 4 ablation metrics show a regression on these specific episodes
vs. the 271 `exact` ones; see `plan.md` R-006.

### 8.5.5 Full-task batch results (277 episodes)

`notebooks/03_stage2_batch.py` aggregates every `outputs/stage1/<idx>_metrics.json`
into `outputs/stage1_summary_basic_pick_place.csv` and writes an aggregate
JSON at `outputs/stage1_aggregate.json`. Numbers below are over all 277
`basic_pick_place` episodes, active-hand split 221 R / 56 L.

**Method mix**: `exact` = 271, `ransac_fallback` (stub) = 6 (see §8.5.4).

| metric | p05 | p50 | mean | p95 | max |
|---|---:|---:|---:|---:|---:|
| raw inter-frame camera angle (°) | 0.076 | **0.170** | 0.228 | 0.532 | 0.832 |
| raw inter-frame ORB disp (px) | 1.42 | **4.10** | 5.01 | 11.19 | 17.76 |
| stabilized inter-frame ORB disp (px) | 0.26 | **1.72** | 2.49 | 7.39 | 11.25 |
| reduction ratio (raw/stab) | 1.04× | **2.50×** | 3.26× | 8.28× | 31.14× |
| inlier H-RMSE (px, per episode median) | 1.06 | **2.66** | 4.20 | 14.26 | 38.91 |

Headline pass/fail fractions:

- **61.7%** of episodes achieve > 2× stabilization
- **24.5%** of episodes achieve > 4×
- **92.4%** of episodes have inlier H-RMSE < 10 px

The raw inter-frame camera angle is small (median 0.17°/frame, max 0.83°) —
AVP wearers hold their head pretty still for seated tabletop PnP. That
matches the qualitative "mostly static" impression from the smoke test on
episode 0 and explains why the reduction ratio p05 is barely above 1× —
there isn't much motion to take out in the quietest episodes. The
reduction is load-bearing on the noisier 50–75th percentile of episodes
where raw inter-frame disp is 4–7 px and the stabilizer brings it to
1.5–3 px.

Per-episode rows are in `outputs/stage1_summary_basic_pick_place.csv`;
aggregate stats cached at `outputs/stage1_aggregate.json`.

---

## 8.6 Stage 2 action alignment — H2R frame + IK

`src/mimicdreamer_egodex/action_alignment.py`. Converts an EgoDex episode's
active-hand wrist trajectory into a UR5e 6-DOF joint-angle trajectory plus a
binary gripper signal.

### 8.6.1 Target robot — UR5e (`plan.md` R-005)

- Loaded via `robot_descriptions.loaders.mujoco.load_robot_description("ur5e_mj_description")`.
  First call clones `mujoco_menagerie` (~30 s, one-time).
- `njnt = nq = nv = 6`. Joint order:
  `[shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]`. All
  revolute hinges. Limits `±2π` except elbow `±π`.
- IK target frame: **`attachment_site`** (tool flange, `frame_type="site"`) —
  also the mount point where a Stage 3 dexterous hand goes, so arm IK and
  finger retargeting stay decoupled.
- One keyframe: `home = [-π/2, -π/2, π/2, -π/2, -π/2, 0]`, which puts
  `attachment_site` at robot-frame `(-0.134, 0.492, 0.488)` with the tool
  pointing up. Used as the IK seed.

### 8.6.2 Schema contract (correcting §2.1 of `initial_plan.md`)

The `initial_plan.md` §2.1 extractor does
`np.linalg.inv(extrinsics[t]) @ wrist_poses[t]`. **Do not do this** — in the
real schema (`plan.md` D-004), `transforms/leftHand`, `transforms/rightHand`,
and every finger joint are **already in world frame** (see §2 above). Reading
them directly is both correct and cheaper. `action_alignment.load_episode`
does exactly this.

Fingertip joint names per side (see §2.1):

- `<side>ThumbTip`
- `<side>IndexFingerTip`
- `<side>MiddleFingerTip`
- `<side>RingFingerTip`
- `<side>LittleFingerTip`

(5 per hand, in that order, matching `FINGERTIPS` in `action_alignment.py`.)

### 8.6.3 H2R — human world to robot base frame

ARKit world is y-up (§3); the UR5e MJCF is z-up. We fix the y→z mapping with
a constant rotation:

```
R_W2R = [[1, 0,  0],
         [0, 0, -1],
         [0, 1,  0]]
```

`det(R_W2R) = +1`, `R_W2R R_W2R^T = I`. Under this: `world x → robot x`,
`world y → robot z`, `world z → robot -y`. **Yaw around the robot z-axis is
arbitrary** — ARKit world yaw depends on where the wearer was facing when
the AVP initialized, so there is no "correct" alignment of the world
horizontal plane to the robot horizontal plane. We absorb this freedom into
the translation step below.

The full H2R transform is `p_robot = scale * (R_W2R @ p_world) + t`. The
translation `t` is computed per episode so that:

1. The table surface (`table_y = 5th-pctl active-wrist world-y`, per R-001)
   lands at **robot z = 0**.
2. The **mean wrist xy** in the rotated frame lands at
   `WORKSPACE_CENTER_XY = (0.5, 0.0)` — 0.5 m in front of the UR5e base,
   centered on the midline. UR5e reach is ~0.85 m, so this leaves headroom
   on both sides.

Rotations are mapped via `R_robot = R_W2R @ R_world`. Scale defaults to
**1.0**. The `initial_plan.md` §2.2 `scale = 0.6 / 0.5` hack is rejected:
(a) the "0.5 m human reach" number is eyeballed, and (b) a non-1 scale
breaks the simultaneous "table at z=0" + "centered workspace" condition
unless the two translations are re-derived consistently. `--scale` remains
a CLI flag for experiments.

### 8.6.4 IK — mink FrameTask + PostureTask smoothness

Per-frame, one `FrameTask(attachment_site, position_cost=1.0,
orientation_cost=0.3)` is targeted at the H2R-aligned wrist pose, and one
`PostureTask(cost=lambda_smooth)` is set to the **previous frame's solution**
and held fixed through inner iterations. The inner loop runs up to
`step_iters=10` Newton steps (`seed_iters=80` on frame 0) and breaks when
`||pos_err|| < 2 mm`. Solver: `quadprog`, damping `1e-3`.

The "posture anchored at previous-frame q" choice is the smoothness term
`initial_plan.md` §2.3 wanted (but wrote slightly wrong — it sets the
anchor to the *current* configuration instead, which reduces to a generic
velocity damping). With the anchor fixed to `q_{t-1}` for the duration of
frame t's iterations, the posture residual grows as q drifts toward the
frame-t target, which pulls q back and trades off "reach the target" vs.
"stay close to yesterday's solution". That is the FIVER-v1-killing knob.

`mink.FrameTask.compute_error` returns a body twist in se(3) as
**`[tx, ty, tz, rx, ry, rz]`** (verified empirically on 2026-04-08 by
shifting the target +10 cm along robot x and reading the result). Position
error = `||err[:3]||`, orientation error = `||err[3:]||` (radians).

### 8.6.5 Gripper signal

Mean of `||tip_i - wrist||` over the 5 fingertips (world frame). Threshold
at the **per-episode median** of openness, then median-filter with
`window=5`. Calibrate per task later if needed (`--gripper-threshold` CLI
flag). Snapshot from `basic_pick_place/0`: threshold 0.142 m, 49% open.

### 8.6.6 Metrics & variance guard

Written to `outputs/stage2/<idx>_metrics.json` per episode:

- `pos_err_m_median / _p95` — final per-frame Cartesian error after the
  inner IK loop exits (m).
- `ori_err_deg_median / _p95` — orientation twist magnitude (deg).
- `iters_mean / _max` — inner-loop iterations per frame.
- `variance.per_joint_range_rad` — per-joint range over the episode.
- `variance.n_joints_range_gt_0_3rad` — mandatory FIVER-collapse guard
  (CLAUDE.md); for UR5e on tabletop PnP this should be 5 or 6 out of 6.

First-cut results (two smoke-test episodes, `basic_pick_place`):

| episode | frames | pos_err med (mm) | pos_err p95 (mm) | ori_err med (°) | range > 0.3 rad |
|---------|------:|-----:|-----:|-----:|---:|
| `basic_pick_place/0` | 126 | **2.14** |  15.3 | 0.23 | **6/6** |
| `basic_pick_place/1` | 160 | **3.87** |  59.9 | 0.68 | **6/6** |

Inter-frame `|Δq|` on ep 0: mean 0.024 rad (~1.4° per frame), max 0.28 rad.
The ep-1 p95 tail is ~8 frames at peak arm excursion where the posture
anchor fights the frame task near the edge of the UR5e workspace;
non-systematic on 2 episodes but worth checking in the 277-episode batch.

### 8.6.7 Full-task batch results (277 episodes)

`notebooks/03_stage2_batch.py` runs `process_episode` over all 277
`basic_pick_place` episodes in a single process (so the UR5e MJCF is
loaded once and cached — see `_CACHED_UR5E_MODEL`). **Wall time: 62.7 s
total, ~0.23 s per episode.** Zero failures. Active-hand split 221 R / 56 L
(identical to Stage 0 variance report, as expected).

Per-episode pos/orientation/variance summary:

| metric | p05 | p50 | mean | p95 | max |
|---|---:|---:|---:|---:|---:|
| pos_err median (mm) | 0.29 | **1.38** | 2.48 | 4.53 | 99.6 |
| pos_err p95 (mm) | 2.36 | **5.74** | 10.97 | 24.43 | 444.6 |
| ori_err median (°) | 0.06 | **0.29** | 0.59 | 0.92 | 20.0 |
| ori_err p95 (°) | 0.54 | **1.49** | 2.60 | 7.28 | 34.9 |
| IK iters (mean per frame) | 2.25 | 4.89 | 5.20 | 9.15 | 10.36 |
| n_joints with range > 0.3 rad | 4 | **5** | 5.39 | 6 | 6 |
| min joint range (rad) | 0.112 | **0.297** | 0.311 | 0.576 | 1.206 |

Headline fractions:

- **96.8%** of episodes have pos_err median < 5 mm
- **97.5%** have pos_err median < 10 mm
- **92.4%** have pos_err p95 < 20 mm
- **97.1%** have pos_err p95 < 50 mm
- **90.6%** of episodes clear ≥ 5/6 joints > 0.3 rad (the FIVER-collapse
  guard; see §8.6.8 below on why 5/6 is the right cut, not 6/6)
- 49.5% clear all 6/6 joints
- 96.8% of episodes have the entire wrist trajectory above `CONF_TRACKED`
  with no gap-filling needed; the remaining 3.2% gap-fill a handful of
  frames each

Per-episode CSV: `outputs/stage2_summary_basic_pick_place.csv`. Aggregate
JSON: `outputs/stage2_aggregate.json`.

### 8.6.8 Variance-guard interpretation + known IK tail

**Variance guard, re-framed.** The FIVER v1 failure mode was collapsed
*global* joint ranges — the IK gave basically constant joint values even
though the wrist moved. The original `initial_plan.md` §2.5 check asks
"are most joints > 0.3 rad", which on UR5e tabletop PnP is too strict:
wrist_2 / wrist_3 sometimes stay near zero because the human wearer
doesn't twist the hand much. That is not collapse — it is a legitimate
quiet DOF. The meaningful cut is **≥ 5/6 joints > 0.3 rad**, which 90.6%
of episodes clear. The reaching joints (shoulder_lift, elbow, wrist_1)
clear the cut on effectively 100% of episodes. No FIVER-style collapse is
present.

**Known IK tail: 8 episodes with pos_err p95 > 50 mm** (2.9% of the split).
Sorted by p95:

| idx | frames | tracked | pos_med (mm) | pos_p95 (mm) | var |
|----:|------:|------:|------:|------:|---:|
| 187 |  62 |  62 |   3.04 | **444.65** | 6/6 |
| 190 | 171 | 171 |  99.59 |    137.34  | 5/6 |
| 183 |  84 |  84 |  23.34 |     94.49  | 6/6 |
|  12 |  96 |  96 |  32.69 |     90.40  | 5/6 |
|  80 | 111 | 111 |  23.01 |     87.51  | 5/6 |
|  61 |  85 |  85 |  33.16 |     82.64  | 6/6 |
| 103 |  79 |  79 |  37.84 |     68.09  | 5/6 |
|   1 | 160 | 160 |   3.87 |     59.90  | 6/6 |

Notes:

- **All 8 have fully-tracked wrists** — the tail is not caused by bad
  ARKit confidence. The overlap with the Stage 1 R-006 fallback episodes
  (15, 26, 102, 131, 192, 251) is **empty**.
- Ep **187** is the worst p95 but its median is only 3 mm — one or two
  frames blow up catastrophically (>40 cm). That shape is consistent with
  the IK jumping basins at a wrist-singularity or at a target just outside
  the UR5e reach envelope.
- Ep **190** is the worst *sustained* tail: median 100 mm across a 171-frame
  episode. Likely the H2R-mapped wrist trajectory is partly outside the
  UR5e workspace for a chunk of this episode (the human wrist went further
  than `WORKSPACE_CENTER_XY + reach`), so the IK is pinned to the nearest
  reachable point. Worth revisiting if Stage 4 metrics on this episode
  matter.
- Ep **1** (the smoke-test episode from §8.5.3) confirms the earlier
  observation that its ~60 mm p95 was real but contained — only 8 frames
  above 50 mm and it still clears 6/6 joints > 0.3 rad.

**Tuning knobs to try if/when this tail starts mattering**:

1. Bump `step_iters` from 10 → 25 (mink is cheap; converge harder).
2. Lower `orientation_cost` from 0.3 → 0.1 on the worst frames (trade
   wrist twist for position reach).
3. Lower `lambda_smooth` from 0.1 → 0.03 on the worst frames (let the IK
   break from the previous-frame anchor when the target is far).
4. Per-episode workspace centering on a *moving* window instead of the
   global mean (handles the ep-190 sliding-reach case).

None of these is done yet — deferred until the Stage 4 ablation proves
the tail matters.

### 8.6.9 Artifact format

`outputs/stage2/<idx>_actions.npz`:

| key | shape | dtype | notes |
|-----|-------|-------|-------|
| `q` | `(T, 6)` | float32 | UR5e joint trajectory (rad) |
| `gripper` | `(T,)` | float32 | binary (0/1), median-filtered |
| `gripper_openness` | `(T,)` | float32 | raw openness (m) |
| `pos_err_m` | `(T,)` | float32 | per-frame IK pos error |
| `ori_err_deg` | `(T,)` | float32 | per-frame IK orientation error |
| `iters` | `(T,)` | int32 | inner-loop iterations |
| `wrist_targets_robot` | `(T, 4, 4)` | float32 | robot-frame H2R targets |
| `h2r_R`, `h2r_t`, `h2r_scale` |  |  | H2R transform used |
| `table_y_world` |  | float32 | table plane y in world (m) |
| `joint_names` | `(6,)` | str | UR5e joint names |
| `active_hand` | scalar | str | `"left"` or `"right"` |

Ready to be concatenated with Stage 3 finger angles into the final
`[q_arm_6, gripper_1, q_finger_N]` action vector.

---

## 8.7 Stage 3 dexterous finger retargeting — Inspire hand

`src/mimicdreamer_egodex/finger_retargeting.py`. Converts each episode's
active-hand fingertip trajectory (5 tips, world frame) into a per-frame
Inspire-hand joint-angle trajectory via `dex-retargeting`'s
`PositionOptimizer`. Uses the **bundled** `offline/inspire_hand_{left,
right}.yml` configs from the `dex_retargeting` wheel and the URDFs
vendored at `third_party/dex-urdf@7304c7f`.

### 8.7.1 Target hand — Inspire (`plan.md` R-007)

- 6 target (optimized) DOFs per hand:
  `{index, middle, ring, pinky}_proximal_joint`, `thumb_proximal_yaw_joint`,
  `thumb_proximal_pitch_joint`. These are the "canonical" action-vector
  order we use downstream.
- 6 additional URDF DOFs (intermediate + distal) are declared as **mimic
  joints** in the URDF and track the proximals automatically. Total
  non-dummy robot DOF = 12.
- `add_dummy_free_joint: True` prepends a 6-DOF free joint at the hand
  root so the optimizer can absorb any input-frame offset. Full robot
  qpos shape = **18** (6 dummy + 12 hand).
- URDF assets live at
  `third_party/dex-urdf/robots/hands/inspire_hand/inspire_hand_{left,
  right}.urdf`. Override the directory with `DEX_URDF_DIR` env var if
  cloned elsewhere.

### 8.7.2 Input contract (`plan.md` D-007)

`SeqRetargeting.retarget(ref_value)` expects a **pre-sliced `(5, 3)`**
array, in this exact order:

    [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]

This matches the YAML's `target_link_names`. The caller is responsible
for the slice — the `target_link_human_indices: [4, 8, 12, 16, 20]` field
in the YAML is a MediaPipe-21-point landmark convention that
`PositionOptimizer` does **not** apply internally. Passing a `(21, 3)`
array triggers a silent `SmoothL1Loss` broadcast warning and produces
garbage (near-zero deltas). Verified empirically on 2026-04-08.

We feed the 5 tips in the wrist-relative world frame:
`tip_world - wrist_world`. No re-orientation to a hand-local frame is
needed because the dummy free joint absorbs it. No scaling either — the
Inspire hand and a human adult hand are geometrically similar enough
that 1:1 works (verified: episode-0 fingertip-to-wrist distance range
0.123–0.151 m matches §7 exactly).

### 8.7.3 EgoDex → Inspire fingertip mapping

EgoDex calls pinky "LittleFinger"; otherwise names are straightforward:

| Inspire target link | EgoDex joint (right)         | EgoDex joint (left)         |
|---------------------|------------------------------|-----------------------------|
| `thumb_tip`         | `rightThumbTip`              | `leftThumbTip`              |
| `index_tip`         | `rightIndexFingerTip`        | `leftIndexFingerTip`        |
| `middle_tip`        | `rightMiddleFingerTip`       | `leftMiddleFingerTip`       |
| `ring_tip`          | `rightRingFingerTip`         | `leftRingFingerTip`         |
| `pinky_tip`         | `rightLittleFingerTip`       | `leftLittleFingerTip`       |

### 8.7.4 Retargeter cache + reset

Building a `RetargetingConfig` includes constructing a pinocchio model +
an nlopt optimizer, which isn't free (~100 ms). We cache one
`SeqRetargeting` per side module-level and `reset()` it between
episodes so the previous episode's terminal qpos doesn't warm-start the
next episode's first frame. Within an episode, `last_qpos` is retained
across frames so frame N warm-starts from frame N-1 — that is what drops
the mean per-frame cost from 4.1 ms (cold start at frame 0 of ep 0) to
2.4 ms at steady state in the 277-episode batch.

### 8.7.5 Full-task batch results (277 episodes)

`notebooks/04_stage3_batch.py`. Wall time **90.7 s** (~0.33 s/episode
including h5 I/O; **2.4 ms/frame** mean on the pure retargeting loop).
Zero failures. Active-hand split 221 R / 56 L (matches Stages 0–2).

Per-episode per-joint range distribution across the 277 episodes:

| target joint | p05 | p50 | mean | p95 | max |
|---|---:|---:|---:|---:|---:|
| index_proximal        | 0.189 | **0.393** | 0.400 | 0.658 | 1.013 |
| middle_proximal       | 0.216 | **0.454** | 0.461 | 0.745 | 0.874 |
| ring_proximal         | 0.212 | **0.420** | 0.425 | 0.682 | 0.857 |
| pinky_proximal        | 0.192 | **0.426** | 0.436 | 0.700 | 0.895 |
| thumb_proximal_yaw    | 0.185 | **0.433** | 0.450 | 0.772 | 1.310 |
| thumb_proximal_pitch  | 0.144 | **0.279** | 0.288 | 0.467 | 0.576 |

Headline fractions:

- **96.8%** of episodes clear all 6/6 target joints > 0.1 rad
- 71.1% have a minimum target joint range > 0.2 rad
- 24.9% have a minimum target joint range > 0.3 rad
- Mean fingertip-to-wrist distance (per episode): p05=0.127 m, p50=0.142 m,
  p95=0.158 m — hand size is consistent within the dataset

The thumb yaw distribution is the widest across episodes (p05 → p95
goes from 0.19 rad to 0.77 rad) — this is **task-adaptive thumb
opposition**. The stapler grasp in ep 0 produces 0.68 rad yaw; the
iPhone grasp in ep 1 needs only 0.48 rad (flatter grip). The
`mean_proximal`, by contrast, is tighter (p05 0.2 → p95 ~0.7) because
all 4 fingers tend to close together in PnP.

### 8.7.6 Variance-guard tail (9 episodes, decomposes cleanly)

| idx | frames | hand  | var | min_range | tip_spread | note |
|----:|------:|-------|---:|---:|---:|---|
| 255 |  67 | right | 4/6 | 0.022 | 0.152 | short |
| 251 | 113 | left  | 5/6 | 0.034 | 0.130 | **R-006 overlap** |
| 180 |  38 | right | 5/6 | 0.054 | 0.153 | short |
| 159 |  49 | left  | 5/6 | 0.071 | 0.145 | short |
|  28 |  77 | right | 4/6 | 0.073 | 0.153 | short |
| 131 | 142 | left  | 5/6 | 0.075 | 0.119 | **R-006 overlap** |
| 271 |  77 | right | 5/6 | 0.079 | 0.130 | short |
| 197 |  46 | right | 5/6 | 0.084 | 0.123 | short |
| 102 | 100 | left  | 5/6 | 0.090 | 0.150 | **R-006 overlap** |

Two disjoint causes:

1. **R-006 data-quality overlap** (3 episodes: 102, 131, 251). These are
   the same episodes where Stage 1 fell through to the RANSAC fallback
   stub because the active-hand wrist confidence was below the floor.
   Low wrist confidence → low fingertip confidence → noisy retargeting
   input → narrower effective DOF ranges. This is a shared upstream
   issue, not a Stage 3 bug.
2. **Short-episode variance floor** (6 episodes, all under 80 frames).
   Tip spreads are normal (0.12–0.15 m), so the fingers are physically
   reasonable — they simply don't have enough frames for the grasp to
   evolve through its full range. This is a data-duration floor, not a
   retargeting failure.

**Intersection with the Stage 2 IK tail is empty.** Stage 2's convergence
tail (8 episodes with pos_err p95 > 50 mm) and Stage 3's variance-floor
tail (9 episodes with <6/6) are completely disjoint failure modes.
Neither is blocking; both are documented for Stage 4 ablation revisit.

### 8.7.8 Retargeting quality assessment

This section is the **honest evaluation** of how good the Stage 3
retargeting actually is. It is the answer to "is this successful?" and
exists so future sessions don't oversell the results.

#### Pipeline-level quality (already covered in §8.7.5)

Strong on the technical axes:

- 277/277 episodes complete, 0 failures
- 2.4 ms/frame mean, 90.7 s for the full task batch
- 96.8% of episodes clear all 6/6 target joints > 0.1 rad
- Per-finger position error ‖human_tip − robot_tip‖: median 5–10 mm,
  p95 10–20 mm (from `notebooks/05_stage3_visualize.py` on a 4-episode
  sample)

#### What 5–10 mm fingertip error actually means

The Inspire-hand fingerpad is roughly **15 mm wide**, so a 10 mm
median error is **~60% of the pad width**. For:

- **Power grasps** (stapler, iPhone, mug, can, plushie, donut, …):
  fine — the contact patch is much larger than the error envelope.
- **Precision pinches** (screw, bead, cable): marginal to bad — the
  fingertip would miss the object more often than not.

The 277-episode `basic_pick_place` distribution is overwhelmingly
power grasps, so we are firmly inside the safe operating regime *for
this dataset*. The pipeline is not validated for precision tasks.

#### Per-object grasp clustering (`notebooks/07_grasp_clustering.py`)

The single-number question — *does the retargeting actually preserve
object identity?* — was answered by extracting a per-episode grasp
signature and grouping by `llm_objects[0]` from the HDF5 attrs. **25
distinct objects** appear with ≥ 3 episodes each in `basic_pick_place`,
covering 174 of the 277 episodes (62.8%).

Two signature flavors were tested:

| signature | dim | within std (rad) | between std (rad) | ratio | silhouette | % positive |
|---|---|---:|---:|---:|---:|---:|
| **6-D peak grasp** (`q_finger` at most-closed frame) | 6 | 0.143 | 0.115 | **0.81** | **−0.27** | 21% |
| **18-D trajectory** (per-joint min/max/mean over episode) | 18 | 0.103 | 0.098 | **0.95** | **−0.15** | 28% |

Both ratios are **< 1** and both silhouette scores are **negative**.
A naive read of those two numbers says "retargeting does not cluster
by object". But the per-pair distance matrix tells a more nuanced
story:

**Most-distinct object pairs (18-D, rad):**

| pair | distance | physical interpretation |
|---|---:|---|
| dice box vs toy block | 1.34 | large container vs small block |
| iphone vs toy block | 1.31 | flat thin vs small block |
| iphone vs fry | 1.26 | flat thin vs irregular |
| mouse vs toy block | 1.24 | flat thin vs small block |
| cup vs toy block | 1.24 | curved handle vs small block |

**Most-similar object pairs:**

| pair | distance | physical interpretation |
|---|---:|---|
| iphone vs mouse | 0.15 | both flat-and-thin power grips |
| container of slime vs donut | 0.16 | both ~spherical power grips |
| plushie vs tea cup | 0.17 | both wide-grip targets |
| dice vs croissant | 0.19 | both medium-sized softer grips |
| rubber duck vs strawberry | 0.20 | both small soft round things |

**Pairwise distances among the 6 largest object groups (18-D, rad):**

```
                  block    duck    plushie    egg    dice    bread
   block (n=41)     -      0.441    0.511    0.237   0.222   0.384
   duck  (n=20)   0.441    -        0.210    0.439   0.392   0.474
 plushie(n=14)    0.511   0.210     -        0.443   0.516   0.409
   egg   (n=10)   0.237   0.439    0.443      -      0.371   0.261
   dice  (n= 8)   0.222   0.392    0.516    0.371    -       0.531
   bread (n= 7)   0.384   0.474    0.409    0.261   0.531    -
```

Two physically meaningful groupings emerge:

1. **{block, dice, egg}**: small hard compact objects — pairwise
   distances 0.22–0.37 (close).
2. **{duck, plushie}**: soft plush — distance 0.21 (close).
3. Cross-group distances ({block, dice} ↔ {duck, plushie}) are larger
   (0.39–0.52).

#### Verdict — affordance class, not within-class precision

The retargeting captures **grasp affordance class** (small hard,
soft plush, flat thin, irregular bulky, …) but does **not** cleanly
discriminate within an affordance class. The mean silhouette being
negative reflects the fact that, e.g., an iphone and a mouse really do
require nearly identical 6-DOF Inspire grasps — calling that a
"clustering failure" would be overfitting the metric.

This is consistent with three known limits:

1. **6-DOF Inspire** can only express a limited grasp vocabulary on a
   power-grasp dataset. A real human hand is 25 DOF; the embodiment
   gap is 19 DOFs of subtle finger dynamics that the retargeting
   *cannot* recover by construction.
2. The `llm_objects` labels are **noisy by construction**: "block"
   covers 41 episodes that are physically a wooden cube, a lego
   brick, a foam block, etc. — none distinguished in the label.
   That inflates within-object std with label noise that has nothing
   to do with retargeting.
3. **Peak-grasp argmin is single-instant**, so it is dominated by
   one frame's noise. The 18-D trajectory aggregate softens this and
   improves the ratio from 0.81 → 0.95 — but it can't fix the
   embodiment gap.

#### What the retargeting is NOT

To prevent later overclaiming:

- **No physics validation.** We never closed the retargeted hand
  around a virtual object and checked grasp force closure. The joint
  angles are kinematically plausible; whether they would *hold* an
  object is untested.
- **Open-loop, no contact correction.** No tactile or force feedback;
  identical to the original MimicDreamer recipe (so we are at parity,
  not better).
- **Stage 2 (arm IK) and Stage 3 (finger retargeting) are not
  jointly optimized.** The wrist pose Stage 2 produces and the wrist
  pose Stage 3 implicitly assumes (via the dummy free joint) can drift
  apart at the mount point. Practical drift is small but it is not
  zero.
- **Distribution is narrow** — 17.9 minutes of `basic_pick_place`
  in a single environment. Generalization to other tasks / wearers /
  scenes is unknown until Stage 4 evals.

#### Why this is enough for Stage 4

For behavioral cloning on this dataset, the action signal needs to
encode **how to grasp**, not **which object** — object identity comes
from the RGB observation. The retargeting providing consistent
affordance-level encodings (close hand for compact items, wider grip
for flat items, thumb opposition for things that need pinching)
plus 5–10 mm fingertip precision is *enough signal to learn from*.
The BC policy will absorb residual retargeting noise the way it
absorbs visual noise.

The bar this needs to clear is set in `initial_plan.md` §4.4:
**outperform the binary-gripper baseline in the ablation table.**
That is the experiment that decides whether the dexterous hand was
worth the engineering. It has not run yet — the verdict on
"is the retargeting useful for the policy?" is **TBD until Stage 4**.

#### What would change this assessment

In order of effort:

1. **(easy)** Re-run with a richer trajectory-summary signature
   (already done — 18-D version, shifts ratio 0.81 → 0.95).
2. **(medium)** Drop the retargeted hand into MuJoCo, place a
   primitive object at the right location each frame, close the hand,
   and check whether the contact set is force-closure stable. This
   bridges kinematic plausibility to physical feasibility.
3. **(hard, on-roadmap)** Stage 4 ablation: full pipeline vs binary
   gripper baseline. Strongest possible validation.

### 8.7.9 Artifact format

`outputs/stage3/<idx>_fingers.npz`:

| key | shape | dtype | notes |
|-----|-------|-------|-------|
| `q_finger` | `(T, 6)` | float32 | Inspire 6 target proximals, action-vector input for Stage 4. Order: `[index, middle, ring, pinky, thumb_yaw, thumb_pitch]`. |
| `q_full` | `(T, 18)` | float32 | Full robot qpos (6 dummy free-joint + 12 hand DOFs) — for MuJoCo visualization. |
| `joint_names_target` | `(6,)` | str | names for `q_finger` columns |
| `joint_names_full` | `(18,)` | str | names for `q_full` columns (pinocchio order) |
| `tips_rel` | `(T, 5, 3)` | float32 | wrist-relative fingertip positions (the retargeter's input) |
| `wrist_world` | `(T, 3)` | float32 | absolute wrist position (debugging) |
| `tips_world` | `(T, 5, 3)` | float32 | absolute fingertip positions (debugging) |
| `tips_conf` | `(T, 5)` | float32 | per-fingertip ARKit confidence |
| `active_hand` | scalar | str | `"left"` or `"right"` |
| `hand_model` | scalar | str | `"inspire"` |

The Stage 4 action vector per episode is assembled as:

    a_t = concatenate([stage2[q][t], stage2[gripper][t], stage3[q_finger][t]])
        # shape (13,) = 6 UR5e + 1 gripper + 6 Inspire proximals

---

## 8.8 Stage 4 — policy training + first eval (R-009)

`notebooks/{08_to_lerobot,09_train_act,10_eval_act}.py` plus
`src/mimicdreamer_egodex/{lerobot_pyav_patch,eval_env}.py`. End-to-end
pipeline complete; **first eval result is 10% success rate** — proof-
of-life only, NOT the §4.4 deliverable. The full assessment lives in
`plan.md` R-009; this section documents the *how*.

### 8.8.1 LeRobot dataset format + Stage 1+2+3 → LeRobot conversion

`notebooks/08_to_lerobot.py` builds a v3.0 LeRobotDataset from the
per-episode artifacts produced by Stages 1–3:

| input | source | shape |
|---|---|---|
| stabilized RGB | `outputs/stage1/<idx>_stabilized.mp4` (resized to 224×224) | (T, 224, 224, 3) uint8 |
| `q_arm` | `outputs/stage2/<idx>_actions.npz` → `q` | (T, 6) float32 |
| `gripper` | same npz, `gripper` | (T,) float32 |
| `q_finger` | `outputs/stage3/<idx>_fingers.npz` → `q_finger` | (T, 6) float32 |
| language | EgoDex HDF5 file attr `description` | str |

The 13-D action vector is built per frame as
`concat([q_arm, gripper, q_finger])`. **State equals action** because
this is an offline-converted dataset, not a live recording — the IK
targets ARE the trajectory the robot would have followed; using the
same vector for both gives the policy a "current pose → next pose
chunk" mapping.

LeRobotDataset feature schema:

```python
features = {
    "observation.image": {"dtype": "video",   "shape": (224, 224, 3), "names": ["height", "width", "channels"]},
    "observation.state": {"dtype": "float32", "shape": (13,),         "names": None},
    "action":            {"dtype": "float32", "shape": (13,),         "names": None},
}
```

Build call: `LeRobotDataset.create(repo_id=..., fps=30, features=...,
robot_type="ur5e+inspire")`. Per frame: `dataset.add_frame({...,
"task": description})`. Per episode: `dataset.save_episode()`. At end:
`dataset.finalize()`.

Full-task batch result: **277/277 episodes, 0 failures, 75 MB on disk,
1397 s wall time (~5 s/ep — dominated by SVT-AV1 video encoding)**.
Final dataset metadata:

| | value |
|---|---:|
| codebase_version | v3.0 |
| total_episodes | 277 |
| total_frames | 32 271 |
| total_tasks (unique language strings) | 200 |
| fps | 30 |

### 8.8.2 PyAV monkey-patch for the lerobot video decoder (D-012)

`src/mimicdreamer_egodex/lerobot_pyav_patch.py`. lerobot 0.5.1's
`decode_video_frames` dispatch is broken on our torch 2.12 nightly +
torchvision 0.27 nightly + cu128 environment:

- `torchcodec 0.10` (the version lerobot pins): built against torch
  2.10's c10 ABI → undefined symbol
  `c10::MessageLogger::MessageLogger(const char*, int, int, bool)` at
  `torch.ops.load_library` time.
- `torchcodec 0.11`: built against CUDA 13 → wants `libnvrtc.so.13`
  which we don't have (cu128 ships nvrtc 12).
- The "pyav fallback" inside lerobot's `decode_video_frames_torchvision`
  calls `torchvision.io.VideoReader(path, "video")` which **does not
  exist** in torchvision 0.27 (the legacy video API was removed).

The patch installs a third path: `decode_video_frames_pyav_only` that
uses `av` (PyAV) directly. PyAV is mature, ABI-independent of torch,
and is already a transitive dep of lerobot. The replacement preserves
the original contract: `(N, C, H, W)` float32 in `[0, 1]`, with
timestamp-tolerance frame matching.

**Usage** — call `apply()` BEFORE any `from lerobot...` import that
touches the dataset reader. The patch monkey-patches both
`lerobot.datasets.video_utils.decode_video_frames` AND
`lerobot.datasets.dataset_reader.decode_video_frames` (the latter
binds the function name at import time, so without patching that
reference too, the dataset reader keeps the old broken function).

Verified end-to-end: loading frame 0 of the smoke-test dataset returns
`observation.image` of shape `(3, 224, 224)` float32 in `[0.000, 0.949]`,
no errors.

### 8.8.3 ACT training setup (`notebooks/09_train_act.py`)

Custom training loop, NOT lerobot's `train.py` CLI (which uses
`accelerate` + `draccus` config injection — too much extra machinery
for a small replication run that needs ablation flexibility).

Pipeline:
1. `apply_pyav_patch()` BEFORE importing lerobot
2. Load `LeRobotDataset` for stats only (`meta.stats`)
3. Episodes 0:222 = train, 222:277 = val (deterministic last-20% split)
4. Build train/val datasets with `delta_timestamps={"action": [i/30 for
   i in range(chunk_size)]}` so each frame yields a chunk-length action
   sequence
5. `ACTConfig(chunk_size=10, n_action_steps=8, n_obs_steps=1)` with our
   13-D action + 224×224 image + 13-D state features
6. `make_act_pre_post_processors(cfg, dataset_stats)` builds the
   per-feature normalize/unnormalize pipeline
7. AdamW (lr=1e-4, wd=1e-4) + grad-clip 10.0 + bf16 autocast
8. Loop: `batch = preprocessor(batch_to_device); loss, _ =
   policy.forward(batch); loss.backward(); optimizer.step()`

**D-014 — ACT eval-mode VAE workaround**: in the val pass, do **NOT**
call `policy.eval()`. ACT's `forward()` unconditionally computes the
VAE KL loss but skips the VAE encoder in `eval()` mode, so
`log_sigma_x2_hat` is `None` and the formula crashes with
`TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'`.
Wrap with `torch.no_grad()` only and keep the model in `train()`. This
is safe — the VAE encoder still sees the val batch and produces a
well-defined loss; the per-batch number is still a valid across-split
comparison metric.

**Inference is fine in `eval()` mode** — `select_action()` doesn't
touch the buggy KL term.

#### 8.8.3.1 Training curves (full pipeline, 3000 steps, batch 64)

| step | train loss | val loss |
|---:|---:|---:|
| 100 | 4.77 | — |
| 500 | — | 0.36 |
| 1000 | 0.241 | 0.229 |
| 1500 | 0.192 | 0.199 |
| 2000 | 0.158 | 0.182 |
| **2500** | **0.137** | **0.165** ← best |
| 3000 (final) | 0.126 | 0.173 |

- 51.6 M-param ACT (ResNet18 vision backbone, 4 encoder layers, 1
  decoder layer)
- ~25 min wall time on RTX 5090
- ~10 epochs over 25 673 train samples
- val loss ≤ train loss at every step until step 2500 → no
  overfitting; mild train/val gap of 0.05 at the end is expected on
  17.9 min of footage
- Best val at step 2500 → mild early-stopping signal but not severe
  enough to stop the run

### 8.8.4 MuJoCo eval env (`src/mimicdreamer_egodex/eval_env.py`)

Builds a UR5e + Inspire-hand + tabletop + primitive-cube scene by
**`MjSpec.attach()`-merging** the menagerie UR5e MJCF and the dex-urdf
Inspire URDF (with the D-010 `<mujoco><compiler strippath="false"
discardvisual="true" meshdir="..."/></mujoco>` injection). The merged
model has:

| | value |
|---|---:|
| njnt | 19 (6 UR5e + 12 Inspire + 1 free joint for the object) |
| nq | 25 (incl. 7 for the object's free joint) |
| nbody | 29 |
| ngeom | 52 |
| ncam | 1 (egocentric, auto-orienting at the object via `targetbody` mode) |

Scene contents:

- UR5e at world origin, base on the floor
- Inspire hand attached at UR5e `attachment_site` with the `ins_`
  prefix on all joint names
- Brown tabletop at `(0.5, 0.0, -0.025)` with half-extents
  `(0.30, 0.30, 0.025)` — top surface flush with `z = 0`, matching
  Stage 2's H2R `table_y → robot z = 0` convention
- Blue 4 cm primitive cube on top of the table at `(0.5, 0.0, 0.02)`
  with a free joint and 50 g mass — physics-driven
- Egocentric camera at `(-0.05, 0.05, 1.10)` (rough AVP head height,
  not yet calibrated to a specific episode's `camera/intrinsic`)

**D-013 — kinematic control, not torque**: `EvalEnv.step` writes
`data.qpos[:6] = action[:6]` and `data.qpos[inspire_target_qadr] =
action[7:13]` directly each tick, then `mj_step()`. The 6 Inspire
mimic joints are driven by the URDF mimic equality constraints during
`mj_forward`. `action[6]` (gripper scalar) is a no-op — the Inspire
DOFs already encode the full hand pose. This sidesteps controller
tuning and keeps the rollout fast (~350 ms each end-to-end).

`get_obs()` returns the same dict shape the policy expects:

```python
{
    "observation.image": (3, 224, 224) float32 in [0, 1],
    "observation.state": (13,) float32  # current arm + zero gripper + Inspire proximals
}
```

`reset()` randomizes the object position over a 20×20 cm patch around
the workspace center (`x ∈ [0.4, 0.6]`, `y ∈ [-0.1, 0.1]`).

### 8.8.5 First-pass rollout results (20 rollouts, full pipeline)

`notebooks/10_eval_act.py`. Loads `latest.pt` (= step 3000 state_dict)
and runs N rollouts in `EvalEnv` with up to 120 env steps each
(4 s @ 30 Hz). Inference path: `obs → preprocessor → policy.select_action
→ postprocessor → action → env.step`.

Per-rollout outcomes (seed=0):

```
[  1/20] success=0  max_lift=   0.0 mm
[  2/20] success=1  max_lift=  51.3 mm  ← clean lift, terminated at step 97
[  3/20] success=0  max_lift=   0.0 mm
[  4/20] success=1  max_lift=  50.0 mm  ← clean lift, terminated at step 111
[  5/20] success=0  max_lift=   0.2 mm
[  6/20] success=0  max_lift=   0.0 mm
[  7/20] success=0  max_lift=   0.0 mm
[  8/20] success=0  max_lift=   0.0 mm
[  9/20] success=0  max_lift=   0.0 mm
[ 10/20] success=0  max_lift=  20.2 mm  ← partial touch
[ 11/20] success=0  max_lift=   0.6 mm
[ 12/20] success=0  max_lift=  13.8 mm  ← partial touch
[ 13/20] success=0  max_lift=   1.5 mm
[ 14/20] success=0  max_lift=   0.0 mm
[ 15/20] success=0  max_lift=   7.1 mm  ← partial touch
[ 16/20] success=0  max_lift=   0.0 mm
[ 17/20] success=0  max_lift=  14.3 mm  ← partial touch
[ 18/20] success=0  max_lift=   0.0 mm
[ 19/20] success=0  max_lift=  12.5 mm  ← partial touch
[ 20/20] success=0  max_lift=   0.0 mm
```

Aggregate:

| metric | value |
|---|---:|
| **success rate** | **2 / 20 = 10 %** |
| mean max lift | 8.6 mm |
| median max lift | 0.08 mm |
| max max lift | 51.3 mm |
| wall time | 7.1 s (~350 ms/rollout) |

Distribution of outcomes:
- **2 clean successes** — full reach + close + lift sequence
- **6 partial touches** (1–20 mm of lift) — fingers reach the object but
  lose grip
- **12 no-movement** — commanded pose misses the object entirely

### 8.8.6 Honest interpretation of 10% — distribution shift hypothesis

This is **proof-of-life**, not a final result. Three things to keep in
mind when reading the number:

1. **A truly random or dead policy would be ~0%.** The 6 partial
   touches confirm the policy is meaningfully *directing* the hand
   toward the object's neighborhood, and the 2 clean successes prove
   the full reach → close → lift sequence is reproducible. The policy
   is alive.
2. **A polished BC policy on a matched-distribution eval would be
   50–90%.** We are not there.
3. **The training metrics (val loss 0.165) say the model learned the
   data well.** This is not a training failure — it is an
   evaluation-domain mismatch.

**The dominant residual error is visual distribution shift**:

| | training | eval |
|---|---|---|
| RGB source | 1080p egocentric AVP video | 224×224 procedural MuJoCo render |
| Hand visible in frame | real human hand | UR5e + Inspire (collision-mesh primitives) |
| Background | varied real environments (metal table, white wall, etc) | checkerboard floor, single light |
| Object | 99 distinct real objects | one bright primitive cube |
| Camera | calibrated AVP intrinsics, head-mounted | hand-positioned `MjvCamera`, NOT calibrated |
| Lighting | real-world directional + ambient | mujoco default |

The ResNet18 vision backbone is sensitive to all of those statistics.
The Stage 3 R-008 verdict ("affordance class adequate, not within-class
precision") is independently true and would still be a ceiling on
performance, but the **Stage 4 first-cut bottleneck is upstream of the
retargeting** — the policy isn't seeing the same kind of image at eval
that it saw at train.

### 8.8.7 What the 10 % is NOT

To prevent later overclaiming:

- **NOT a § 4.4 ablation table cell**. The §4.4 deliverable requires
  4 conditions × 100 rollouts. We have 1 condition × 20 rollouts.
  Use this number as a placeholder for the "Full pipeline" cell, but
  **regenerate** after the camera-intrinsics fix before quoting it
  publicly.
- **NOT torque-controlled rollouts**. The eval env writes joint
  positions directly via `data.qpos` (D-013). If a future eval needs
  torque-controlled rollouts (e.g., for sim-to-real readiness), swap
  `step()` to write `data.ctrl[]`.
- **NOT a measurement of grasp quality at the contact level**. The
  success metric is "object lifted > 5 cm". A policy that closes the
  hand 1 cm from the object and then drags it via friction would
  count as "no movement"; a policy that gets one finger under the
  object and flips it 5 cm would count as "success". Both happen.
- **NOT compared against a baseline**. A binary-gripper baseline
  ("no Inspire fingers", action shape 7) might do better OR worse —
  we don't know yet.

### 8.8.8 What would change this assessment (ranked by ROI)

1. **(easy, ~30 min)** **Calibrate the eval `MjvCamera`** to one of the
   277 EgoDex episodes' `camera/intrinsic` + `transforms/camera`.
   Single highest-leverage fix — should drop the visual distribution
   shift dramatically. Estimate: 2–4× success rate improvement.
2. **(easy, ~30 min)** Texture the table top + tune lighting in the
   eval scene to match EgoDex's metal-table-on-white-background look.
3. **(medium, ~80 min)** Train longer (10 000 steps instead of 3000).
   Best val was at step 2500, suggesting we may be slightly
   under-trained rather than over-trained.
4. **(hard, ~4 hours)** § 4.4 ablation table proper: build the 3 other
   dataset variants and train each one. This is the real Stage 4
   deliverable.

### 8.8.9 Artifact layout

```
outputs/
├── lerobot/mimicdreamer_egodex_basic_pick_place_full/  # Phase B output (75 MB)
│   ├── meta/{info,stats}.json          # dataset metadata + per-feature stats
│   ├── data/chunk-000/file-000.parquet # 32 271 frames × {state, action, ...}
│   └── videos/observation.image/chunk-000/file-000.mp4  # AV1, all eps in one file
└── stage4/act_full_pipeline/                # Phase C+D output
    ├── args.json, config.json
    ├── ckpt_step{1000,2000,3000}.pt    # 619 MB each (state + optimizer)
    ├── latest.pt                       # 206 MB (state_dict only — for inference)
    ├── train_log.jsonl                 # per-100-step train metrics
    ├── val_log.jsonl                   # per-500-step val metrics
    └── eval/rollouts.jsonl             # 20 rollouts × {success, max_lift_m, n_steps} + summary
```

---

## 9. File / artifact index

Code (read these to verify any claim above):

| file | what it does |
|------|--------------|
| `notebooks/00_explore_egodex.py` | Single-episode HDF5 dump + per-hand variance/confidence printout. Auto-finds a `basic_pick_place` episode. |
| `notebooks/01_variance_report.py` | Aggregates wrist-range statistics across every episode in a task folder. Writes CSV. The Stage 0 deliverable. |
| `notebooks/02_calibrate_open_questions.py` | Computes the confidence distribution and the per-frame camera-to-table distance across a task. Writes CSV. The R-001 / R-002 calibration source. |
| `notebooks/03_stage2_batch.py` | Full-task batch driver. Aggregates existing Stage 1 metrics JSONs into a CSV + aggregate JSON, then runs Stage 2 `process_episode` over every episode in a single process (UR5e MJCF cached), writing the Stage 2 CSV + aggregate JSON. Single source of the batch distributions in §8.5.5 / §8.6.7 / §8.6.8. |
| `src/mimicdreamer_egodex/egostabilizer.py` | Stage 1 deliverable. Plane-induced homography stabilizer with R-001/R-002/R-003/R-004 baked in. CLI: `uv run python -m mimicdreamer_egodex.egostabilizer <hdf5> --out-dir outputs/stage1`. Writes stabilized MP4 + metrics JSON. |
| `src/mimicdreamer_egodex/action_alignment.py` | Stage 2 deliverable. UR5e IK (mink FrameTask + smoothness PostureTask) on H2R-aligned wrist poses, plus fingertip-spread gripper signal. CLI: `uv run python -m mimicdreamer_egodex.action_alignment <hdf5> --out-dir outputs/stage2`. Writes `<idx>_actions.npz` + metrics JSON. |
| `src/mimicdreamer_egodex/finger_retargeting.py` | Stage 3 deliverable. Inspire 6-DOF finger retargeting via `dex-retargeting`'s `PositionOptimizer` on wrist-relative EgoDex fingertip positions. Caches one `SeqRetargeting` per side. CLI: `uv run python -m mimicdreamer_egodex.finger_retargeting <hdf5> --out-dir outputs/stage3`. Writes `<idx>_fingers.npz` + metrics JSON. |
| `notebooks/04_stage3_batch.py` | Stage 3 full-task batch driver. Runs `process_episode` over every `basic_pick_place` episode (retargeter cached) and writes `outputs/stage3_summary_basic_pick_place.csv` + `outputs/stage3_aggregate.json`. Source of §8.7.5 / §8.7.6 distributions. |
| `notebooks/05_stage3_visualize.py` | Stage 3 static visualizations: per-episode joint-angle time series, 3D fingertip overlay (human vs Inspire FK), per-finger retarget error, 2x2 overview. Run on any episode index. Source of the 5–10 mm per-finger error numbers in §8.7.8. |
| `notebooks/06_stage3_animate.py` | Stage 3 animations: stick-figure Inspire MP4, side-by-side EgoDex-vs-Inspire skeleton MP4, MuJoCo offscreen mesh-render MP4. Headless via EGL. Reads from `outputs/stage3/<idx>_fingers.npz`, writes to `outputs/stage3/viz/`. |
| `notebooks/07_grasp_clustering.py` | Stage 3 quality-assessment: per-object grasp-shape clustering (peak-grasp 6-D and trajectory-aggregate 18-D signatures), separation ratio + silhouette + PCA scatter. Source of the §8.7.8 verdict ("affordance class, not within-class precision"). |
| `notebooks/08_to_lerobot.py` | Stage 4 Phase B — convert per-episode Stage 1+2+3 outputs into a v3.0 LeRobotDataset (75 MB, 277 eps, 32 271 frames). Image at 224×224, action = `[arm_6, gripper_1, finger_6]`. CLI: `uv run python notebooks/08_to_lerobot.py --force` (full task). |
| `notebooks/09_train_act.py` | Stage 4 Phase C — custom training loop for `ACTPolicy` on the LeRobotDataset, NOT lerobot's CLI. Applies the PyAV monkey-patch first. Saves checkpoints + train/val JSONL logs. CLI flags for steps/batch/lr/etc. |
| `notebooks/10_eval_act.py` | Stage 4 Phase D — load a trained ACT checkpoint and run N rollouts in `EvalEnv`. Reports success rate + per-rollout `max_lift_m`. Same script will be used per-condition for the §4.4 ablation. |
| `src/mimicdreamer_egodex/lerobot_pyav_patch.py` | Monkey-patch for `lerobot.datasets.video_utils.decode_video_frames`. Required because torchcodec is unbuildable on torch 2.12 cu128 and torchvision 0.27 dropped `VideoReader`. See D-012 / §8.8.2. **Must be `apply()`-ed before any `from lerobot...` import that touches the dataset reader.** |
| `src/mimicdreamer_egodex/eval_env.py` | UR5e + Inspire merged scene built via `MjSpec.attach()`, with table + primitive cube + egocentric camera. `EvalEnv.reset/step/get_obs` API. Kinematic control via direct `qpos` writes (D-013). See §8.8.4. |

Artifacts:

| path | content |
|------|---------|
| `data/test.zip` | 16.1 GB raw archive |
| `data/test/<task>/<idx>.hdf5` | extracted episode metadata |
| `data/test/<task>/<idx>.mp4` | extracted egocentric video |
| `outputs/variance_report_basic_pick_place.csv` | per-episode wrist ranges + active hand label |
| `outputs/calibration_basic_pick_place.csv` | per-episode confidence percentiles + camera-to-table distances |
| `outputs/stage1/<idx>_stabilized.mp4` | Stage 1 stabilized clips (one per processed episode) |
| `outputs/stage1/<idx>_metrics.json` | Stage 1 metrics dump per episode |
| `outputs/stage2/<idx>_actions.npz` | Stage 2 UR5e joint-angle trajectory + gripper signal per episode |
| `outputs/stage2/<idx>_metrics.json` | Stage 2 IK + variance metrics per episode |
| `outputs/stage1_summary_basic_pick_place.csv` | Per-episode Stage 1 metrics aggregated from all 277 `outputs/stage1/<idx>_metrics.json` files |
| `outputs/stage1_aggregate.json` | Stage 1 headline distributions (method mix, active-hand split, per-metric percentiles, pass fractions) — source for §8.5.5 |
| `outputs/stage2_summary_basic_pick_place.csv` | Per-episode Stage 2 IK + variance metrics (1 row per episode) |
| `outputs/stage2_aggregate.json` | Stage 2 headline distributions — source for §8.6.7 / §8.6.8 |
| `outputs/stage3/<idx>_fingers.npz` | Stage 3 Inspire joint trajectory + wrist-relative tip inputs per episode |
| `outputs/stage3/<idx>_metrics.json` | Stage 3 per-episode retargeting metrics + variance report |
| `outputs/stage3_summary_basic_pick_place.csv` | Per-episode Stage 3 metrics (1 row per episode) |
| `outputs/stage3_aggregate.json` | Stage 3 headline distributions — source for §8.7.5 / §8.7.6 |
| `outputs/stage3_grasp_clustering.json` | Stage 3 grasp-clustering analysis (per-object signatures, separation ratio, silhouette) — source for §8.7.8 verdict |
| `outputs/stage3/viz/<idx>_*.{png,mp4}` | Per-episode static plots (joint angles, fingertip 3D, retarget error, overview) and animations (Inspire stick figure, side-by-side, MuJoCo mesh render) |
| `outputs/stage3/viz/grasp_clustering_*.png` | Cross-episode clustering plots: PCA scatter (peak + trajectory signatures) and per-object signature bar chart |
| `outputs/lerobot/mimicdreamer_egodex_basic_pick_place_full/` | Stage 4 Phase B output. v3.0 LeRobotDataset of all 277 episodes (75 MB). `meta/` + `data/chunk-000/file-000.parquet` + single AV1 video. |
| `outputs/stage4/act_full_pipeline/` | Stage 4 Phase C output. ACT training artifacts: `args.json`, `config.json`, `ckpt_step{1000,2000,3000}.pt` (619 MB each, state + optimizer), `latest.pt` (206 MB, state-dict only), `train_log.jsonl`, `val_log.jsonl`. |
| `outputs/stage4/act_full_pipeline/eval/rollouts.jsonl` | Stage 4 Phase D output. 20 rollouts with per-episode `success`/`max_lift_m`/`n_steps` + summary line. Source of the 10% number. |
| `third_party/dex-urdf/` | Vendored dex-urdf repo @ `7304c7f` (gitignored). Provides Inspire/Allegro/Shadow/LEAP URDFs for Stage 3. Re-clone per README if missing. |
| `logs/session_2026-04-07.md` | narrative session log |
| `logs/runs/2026-04-07_*.log` | captured stdout/stderr from every script run |

Decisions:

- `plan.md` D-004 — schema correction
- `plan.md` D-005 — test-split-only scope
- `plan.md` R-001 — table-distance + y-up resolution
- `plan.md` R-002 — confidence threshold recalibration
- `plan.md` R-003 — `transforms/camera` confirmed cam-to-world
- `plan.md` R-004 — plane-homography sign correction
- `plan.md` R-005 — UR5e chosen as Stage 2 arm URDF
- `plan.md` R-006 — 6 Stage 1 episodes fall through to the RANSAC fallback stub
- `plan.md` R-007 — Inspire hand chosen as Stage 3 dexterous target
- `plan.md` D-007 — `SeqRetargeting.retarget` needs a pre-sliced `(5, 3)` input
- `plan.md` D-008 — `uv sync --group <stage>` replaces the active group set
- `plan.md` R-008 — Stage 3 retargeting quality: affordance-class adequate, not within-class precision
- `plan.md` D-009 — RunPod needs `apt install libegl1` for MuJoCo headless EGL
- `plan.md` D-010 — MuJoCo URDF loader needs `strippath="false" discardvisual="true"` injection
- `plan.md` D-011 — MuJoCo URDF `<mujoco>` extension silently ignores `<visual>` (use 640×480 framebuffer)
- `plan.md` R-009 — Stage 4 first-cut policy + 10% rollout success: pipeline alive but distribution-shift-limited; § 4.4 ablation NOT yet built
- `plan.md` D-012 — PyAV monkey-patch for lerobot's video decoder (torchcodec/torchvision.io both unusable on torch 2.12 nightly)
- `plan.md` D-013 — Stage 4 eval env uses kinematic `qpos` writes, not torque actuators
- `plan.md` D-014 — ACT's `forward()` crashes in `eval()` mode; keep `train()` mode for the val pass

If any of the above goes stale, update both this file *and* the matching
entry in `plan.md` so they don't drift.
