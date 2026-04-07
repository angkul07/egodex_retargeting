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

### 3.1 — `transforms/camera` is camera-to-world

Almost certainly camera-to-world (`T_world_cam`), based on:

- The translation column is in the same scale as the wrist-pose translations.
- Camera y (~1.07) sits at head height in the same world frame in which the
  wrist sits at hand height. If it were world-to-camera, the camera "origin"
  would be at the camera's projection of the world origin, which would not
  generically land at head height.

**To go to camera-frame**: invert. For an SE(3) matrix `T = [[R, t], [0, 1]]`,
the inverse is `T_inv = [[R.T, -R.T @ t], [0, 1]]` — cheap, no `np.linalg.inv`
needed.

**Action items for Stage 1 / Stage 2 first runs**: confirm this convention
on the very first script that depends on it. If the magnitude check fails,
fall back to treating it as world-to-camera.

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

## 9. File / artifact index

Code (read these to verify any claim above):

| file | what it does |
|------|--------------|
| `notebooks/00_explore_egodex.py` | Single-episode HDF5 dump + per-hand variance/confidence printout. Auto-finds a `basic_pick_place` episode. |
| `notebooks/01_variance_report.py` | Aggregates wrist-range statistics across every episode in a task folder. Writes CSV. The Stage 0 deliverable. |
| `notebooks/02_calibrate_open_questions.py` | Computes the confidence distribution and the per-frame camera-to-table distance across a task. Writes CSV. The R-001 / R-002 calibration source. |

Artifacts:

| path | content |
|------|---------|
| `data/test.zip` | 16.1 GB raw archive |
| `data/test/<task>/<idx>.hdf5` | extracted episode metadata |
| `data/test/<task>/<idx>.mp4` | extracted egocentric video |
| `outputs/variance_report_basic_pick_place.csv` | per-episode wrist ranges + active hand label |
| `outputs/calibration_basic_pick_place.csv` | per-episode confidence percentiles + camera-to-table distances |
| `logs/session_2026-04-07.md` | narrative session log |
| `logs/runs/2026-04-07_*.log` | captured stdout/stderr from every script run |

Decisions:

- `plan.md` D-004 — schema correction
- `plan.md` D-005 — test-split-only scope
- `plan.md` R-001 — table-distance + y-up resolution
- `plan.md` R-002 — confidence threshold recalibration

If any of the above goes stale, update both this file *and* the matching
entry in `plan.md` so they don't drift.
