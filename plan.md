# plan.md — working plan

This is the **living** plan. It tracks current state, deviations from
`initial_plan.md`, open questions, and decisions. Update it as work progresses.
The original 4-stage roadmap (Stages 0–4 + contributions + timeline) lives in
`initial_plan.md` and should not be edited.

---

## Current state — 2026-04-08 (Day 1)

**Stage**: 1 and 2 first-cut complete on the full 277-episode
`basic_pick_place` test split.
**Last action**: Wrote `notebooks/03_stage2_batch.py` (full-task batch
driver + Stage 1 metrics aggregator in one pass), ran it: 277/277 episodes,
0 failures, **62.7 s total wall time** (~0.23 s per episode). Outputs at
`outputs/stage{1,2}_summary_basic_pick_place.csv` and
`outputs/stage{1,2}_aggregate.json`. Two findings worth flagging:
**(a)** 6 Stage 1 episodes fell into the `ransac_fallback` stub path
(contradicting R-004's "no episode expected to fail" — see R-006); **(b)**
8 Stage 2 episodes have IK `pos_err_p95 > 50 mm`, with a clean worst case
of ep 190 (median 100 mm, all 171 frames tracked — real IK edge, not a data
bug). The two tails do **not** overlap, so the Stage 2 tail is not caused
by bad wrist tracking. Run log:
`logs/runs/2026-04-08_092549_stage2_batch.log`.
**Next action**: (a) start Stage 3 (`finger_retargeting.py`) — the blocker
was Stage 2, not the tail; (b) investigate the Stage 2 IK tail in a side
session if it matters for the Stage 4 ablation; (c) implement the real
vidstab fallback for the 6 R-006 episodes if Stage 4 metrics regress on
them.

### Environment status
- [x] uv project initialized, Python 3.13. (Reinstalled uv on 2026-04-08:
      RunPod container restart wiped `/root/.local/`, but `/workspace/.venv`
      and `uv.lock` survived; `curl install.sh | sh` + `uv python install 3.13`
      restored everything in <30 s.)
- [x] Base deps installed (`uv sync`): h5py, numpy, opencv-python, scipy, jupyter, etc.
- [x] Stage 1 deps installed (`uv sync --group stage1`): vidstab 1.7.4.
- [x] Stage 2 deps installed (`uv sync --group stage2` + `uv add --group stage2
      robot_descriptions`): mink 1.1.0, mujoco 3.6.0, quadprog 0.1.13,
      qpsolvers 4.11.0, daqp 0.8.5, robot_descriptions 1.23.0.
- [ ] Stage 3 deps: dex-retargeting.
- [ ] Stage 4: torch nightly (cu128) + lerobot — installed manually, see below.
- [x] EgoDex **test split only** downloaded (`data/test.zip`, 16.1 GB) and
      extracted to `data/test/` — 111 task subdirs, 6,598 file entries.
      See **D-005** below: we are not downloading any other EgoDex split.

### Hardware confirmed
- RunPod, RTX 5090 (Blackwell, sm_120). Stable PyTorch cannot target this; must use
  cu128 nightly.

---

## Decisions and deviations from `initial_plan.md`

### D-001: torch / lerobot are installed out-of-band, not via pyproject
**Why**: torch nightly cu128 requires `triton==3.7.0+git9c288bc5` (a local-version
wheel hosted on the pytorch nightly index). uv's resolver does not surface that wheel
through source mapping, even with `prerelease = "allow"`. Lerobot additionally caps
torch at `<2.11`, conflicting with the nightlies.

**How to apply**: After `uv sync`, run

```bash
uv pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 \
    torch torchvision
```

For lerobot, when Stage 4 starts: try `uv pip install --no-deps lerobot` and pull in
its remaining deps manually, OR fork lerobot and bump the torch ceiling. Decide
which path at the time based on which is less invasive.

### D-003: Every step is logged to `/workspace/logs/`
**Why**: The user is often away and reviews logs after the fact to catch errors and
understand what happened. Without disciplined logging, surprises (failed downloads,
silent IK divergence, off-by-one episode indexing) get lost in scrollback.

**How to apply**: See the **Logging** section in `CLAUDE.md` for the full rules.
Two log types:
- `logs/session_YYYY-MM-DD.md` — append-only per-day narrative; lab-notebook style.
- `logs/runs/YYYY-MM-DD_HHMMSS_<slug>.log` — captured stdout/stderr from any
  non-trivial command, linked from the session entry.
Errors get full tracebacks; decisions get mirrored back into this file.

### D-005: EgoDex test split is the *only* data we use
**Why**: User decision, 2026-04-07. The test split (16.1 GB, 111 tasks,
~6,600 files, ~277 `basic_pick_place` episodes worth ~17.9 minutes of
footage at 30 Hz) is enough for the partial replication and the variance
report has already validated it. Avoiding the multi-hundred-GB Part 1/2
downloads keeps disk usage and iteration time small.

**How to apply**:
- Stage 1 (`egostabilizer.py`) — develop and benchmark on `data/test/<task>/`.
- Stage 2 (`action_alignment.py`) — same. Use the active hand of each
  episode (from `notebooks/01_variance_report.py` logic).
- Stage 3 (`finger_retargeting.py`) — same.
- Stage 4 (policy training) — train and evaluate on the `basic_pick_place`
  split *of the test set* (277 episodes). The original §4.2 plan mentioned
  Part 2 explicitly; that is overridden. The dataset is small for a behavior
  cloning policy, so expect the §4.4 ablation table to be more about
  pipeline-quality deltas than absolute success rates.
- `initial_plan.md` §0.3 / §4.2 references to `part2.zip` and Part 2 data
  are now obsolete — treat them as historical context, not active plan.

### D-004: EgoDex HDF5 schema differs from `initial_plan.md` §0.3
**Why**: The schema documented in `initial_plan.md` (a `/hand/{left,right}/joints`
group with packed `(T,25,3)` arrays plus a `/language/annotation` string and a
`/camera/extrinsic` array) does not match the real `test.zip`. Actual layout
(verified on `data/test/basic_pick_place/0.hdf5`):

- `/camera/intrinsic` — `(3,3)` ✓ matches
- `/transforms/camera` — `(T,4,4)` per-frame camera-to-world (NOT under
  `/camera/extrinsic`)
- `/transforms/leftHand`, `/transforms/rightHand` — wrist `(T,4,4)` SE(3)
- `/transforms/<jointName>` — every other joint is its own `(T,4,4)`. There
  are 69 such datasets per file: camera, hip, spine1..7, neck1..4, both
  shoulders/arms/forearms, and 25 joints per hand (1 wrist + 4 thumb +
  4 fingers × 5 each). 25-joint hand layout is preserved, just unpacked.
- `/confidences/<jointName>` — `(T,)` per-joint ARKit confidence (one per
  transform; spine/neck/etc. included)
- Language metadata lives in **file-level HDF5 attributes**, not a dataset:
  `task`, `description`, `llm_description`, `llm_objects`, `llm_verbs`,
  `environment` (e.g. `'hand:right'`), `session_name`, etc.

**How to apply**: Treat `initial_plan.md` §0.3 code snippets as pseudocode for
the *intent* (load wrist trajectory + finger joints + confidence + intrinsics)
but use `transforms/<joint>` paths in real code. The 25-joint-per-hand list is
hard-coded in `notebooks/00_explore_egodex.py::hand_joint_names`. When Stage 2
needs finger spreads, stack the per-joint datasets explicitly. The Stage 0.3
exploration script and Stage 0 variance report both follow this corrected
schema.

Sub-finding: ARKit confidence values appear to be in roughly the 0.3–0.7 range,
not the [0,1]-with-most-frames-near-1 distribution implied by §0.3. On
`basic_pick_place/0.hdf5` no frame had a per-hand mean confidence above 0.8.
This means a confidence threshold of 0.5 (per `initial_plan.md` §2.1) will
reject most data unless the threshold is recalibrated. Open question for
Stage 2 — investigate whether 0.5 is right for this dataset, or whether
confidence is meant to be interpreted relatively per joint.

### D-002: Stage groups are gated behind explicit `uv sync --group <stage>`
**Why**: Each stage adds heavy deps (mink pulls quadprog + a QP solver chain;
dex-retargeting pulls SAPIEN). Keeping them off the default sync makes onboarding
fast on a fresh box and avoids resolution churn during Stages 0–1.

**How to apply**: When you start a new stage, run the matching `uv sync --group stageN`
*before* writing code for it. Update the README setup section if a new group is added.

---

## Resolved questions

### R-001: Stage 1 — table-distance assumption  *(answered 2026-04-07)*
**Original question**: Is the §1.1 hard-coded `table_dist = 0.5 m` acceptable, or
do we need per-episode estimation?

**Answer**: Both the constant *and* the up-axis assumption in `initial_plan.md`
§1.1 are wrong.
- ARKit / EgoDex world frame is **y-up**, not z-up. Verified empirically:
  on `basic_pick_place/0.hdf5` the camera y is essentially constant at
  ~1.07 m while wrist y varies between 0.80–0.98 m. The §1.1 snippet's
  `n = [0, 0, 1]` would point along the depth axis, not the table normal.
- The actual camera-to-(estimated)-table distance across 277
  `basic_pick_place` episodes: median 0.243 m, p10/p90 = 0.171 / 0.312 m,
  global frame range 0.105 – 0.727 m. Only 0.4% of episodes have a per-episode
  mean within ±20% of 0.5 m. So 0.5 m is wrong by roughly 2×.
- The "estimated table plane" here is `5th-percentile-wrist-y` over the
  active hand (the lowest the hand goes, on the assumption that the wrist
  briefly touches the table during pick/place). The real surface is 5–10 cm
  below that because the wrist sits above the contact point. Even after
  that correction, true camera-to-table is ~0.30–0.35 m, still well below
  0.5 m.

**How to apply (Stage 1)**:
1. Use **y-up** in EgoStabilizer. The table plane normal in world frame is
   `n_world = (0, 1, 0)`. To get the normal in the camera-1 frame for the
   homography formula, transform by the inverse of the camera-1 rotation:
   `n_cam = R_w2c @ n_world` where `R_w2c = (cam[:3, :3]).T` if
   `transforms/camera` is camera-to-world (verify on first run).
2. Do **not** use a hard-coded distance constant. We have per-frame camera
   poses, so compute the distance exactly per frame:
   - Estimate `table_y` once per episode as `5th percentile of active-hand
     wrist world-y` (cheap, robust). Optionally subtract a small bias
     (~5 cm) to account for the wrist-above-fingertips offset, but for
     stabilization the bias mostly washes out.
   - At frame `t`, `d_t = |cam_y_world[t] - table_y|`.
3. The §1.1 RANSAC fallback (vidstab) is still useful for episodes whose
   wrist tracking is too noisy to estimate `table_y` confidently — gate the
   exact-homography path on having ≥ 5 frames where the active hand's mean
   confidence is above the threshold from R-002.

Investigation script: `notebooks/02_calibrate_open_questions.py`. Numeric
output captured at `logs/runs/2026-04-07_111200_calibrate_open_questions_v2.log`.
Per-episode CSV at `outputs/calibration_basic_pick_place.csv`.

### R-003: Stage 1 — confirmed `transforms/camera` is camera-to-world  *(answered 2026-04-08)*
**Original question** (action item from `doc.md` §3.1): the convention was
inferred from the magnitude of the translation column on `basic_pick_place/0`
but never empirically pinned down. Stage 1 depends on it for sign correctness.

**Answer**: camera-to-world. Verified on `basic_pick_place/0.hdf5`:
- `transforms/camera[t][:3, 3]` mean = `(-0.077, 1.069, -0.395)`, std
  `(0.022, 0.015, 0.010)` over 126 frames. The y-component sits at head
  height for a seated AVP wearer (~1.07 m) and is ~constant. If this were
  world-to-camera the translation would be the world origin in the camera's
  coordinate system, which would not generically land at head height.
- The cam y-mean (1.069) is above the right-wrist y-mean (0.877), matching
  "head above hand".
- `R R^T - I` is at machine precision; `det(R) = +1`. So the rotation block
  is a proper rotation, not a left-handed flip.

Verification log: `logs/runs/2026-04-08_*_verify_camera_convention.log`.

**How to apply**: in code, treat `transforms/camera[t]` as `T_world_cam_t`
(`X_world = T @ X_cam`). To go to camera frame, invert by transposing the
rotation and applying `-R^T t` — see `egostabilizer.plane_homography`.

### R-004: Stage 1 — sign of the plane-induced homography  *(answered 2026-04-08)*
**Original question**: while smoke-testing `egostabilizer.py` on
`basic_pick_place/1`, stabilization barely reduced inter-frame ORB
displacement (1.04× ratio) and the inlier H-RMSE was ~125 px. The geometry
matched the textbook formula `H = K (R - t n^T / d) K^-1`, but something was
clearly wrong.

**Answer**: the textbook minus-sign assumes a different (src, dst) ordering.
Re-deriving from scratch with our convention (`X_dst = R_ds X_src + t_ds` and
`n_s^T X_s = d_s` describing the plane in the source frame) gives:

    X_d = R_ds X_s + t_ds * (n_s.T X_s / d_s) = (R_ds + t_ds n_s.T / d_s) X_s

i.e. **plus**, not minus. The minus form in Hartley & Zisserman corresponds to
the opposite ordering / inward-pointing normal convention. With the plus sign,
on `basic_pick_place/1`:

| metric                       | before fix | after fix |
|-----------------------------|-----------:|----------:|
| stab inter-frame disp (px)  |       6.65 |      2.11 |
| reduction ratio             |       1.35 |      4.25 |
| inlier H-RMSE (px, median)  |     125.6  |       5.5 |

And on `basic_pick_place/0` (mostly static): 6.05 → 0.92 px (6.6× ratio),
inlier H-RMSE 1.5 px.

**How to apply**: the corrected formula lives at
`egostabilizer.plane_homography`. The `+` sign is non-negotiable for our
(src→dst, world-y-up, camera-to-world `transforms/camera`) convention.
Re-derive from scratch (do not copy from a textbook) if any of those
assumptions ever change.

A second metric subtlety surfaced at the same time: the *raw* H-RMSE
(measured against all ORB matches between frame 0 and a far frame t) is
dominated by features on the moving hand and the manipulated object, which
no plane-induced homography can predict. The metric is only meaningful after
inlier filtering — keep the best 50% of per-pair reprojection errors. The
implementation in `homography_reprojection_rmse` does this.

### R-006: Stage 1 — 6 episodes actually fall through to the RANSAC fallback  *(answered 2026-04-08)*
**Original claim (R-004 / doc.md §8.5.4)**: the `CONF_TRACKED = 0.10`
confidence floor is loose enough that "no episode in the test split is
expected to fail it for the wrist joint" — so the vidstab fallback can
stay a stub.

**Answer**: that claim was wrong. The full-task Stage 1 batch over 277
`basic_pick_place` episodes classified 6 of them as `ransac_fallback`
(i.e. fewer than 5 active-hand wrist frames above `CONF_TRACKED`):

| idx | frames | active hand | stab reduction |
|----:|------:|-------------|---------------:|
|  15 |    51 | right       |         1.00 × |
|  26 |   127 | left        |         1.00 × |
| 102 |   100 | left        |         1.00 × |
| 131 |   142 | left        |         1.00 × |
| 192 |   125 | left        |         1.00 × |
| 251 |   113 | left        |         1.00 × |

5 of 6 are left-hand — the ARKit left-hand wrist-confidence distribution on
this task has a longer low tail than the right. The `1.00×` reduction is
expected: the fallback path is currently a stub that writes raw frames
unchanged, so raw disp == stab disp by construction.

Importantly, **these 6 episodes do not overlap with the Stage 2 IK tail**
(see `doc.md` §8.6.8) — Stage 2's tail is an IK convergence issue, not a
data quality issue.

**How to apply**:
1. Update `doc.md` §8.5.4 to state "6 episodes do fall through; the
   fallback is currently a stub" instead of "none expected".
2. Wire in a real vidstab RANSAC path when either: (a) Stage 4 ablation
   metrics regress on these 6 episodes vs. the 271 `exact` ones, or (b)
   the Stage 2 batch ends up needing them. Neither condition holds yet.
3. For now, these 6 episodes still produce stabilized MP4s (just
   unchanged) and produce valid Stage 2 joint trajectories, so they do
   not block anything.

### R-005: Stage 2 — arm URDF  *(answered 2026-04-08)*
**Original question**: Which arm URDF to target for IK? `initial_plan.md` §2.3
is agnostic; FIVER used UR5e.

**Answer**: **UR5e**, loaded via
`robot_descriptions.loaders.mujoco.load_robot_description("ur5e_mj_description")`.
Reasons:
- 6-DOF standard serial arm; matches the `(arm_6, gripper_1, finger_N)`
  action vector `initial_plan.md` §3.3 already writes.
- Default in mink's own arm examples — minimal integration risk.
- `attachment_site` on the tool flange is the natural frame for the
  Stage 2 `FrameTask` target and is also exactly where a Stage 3 dexterous
  hand would be mounted. Arm and hand remain decoupled: Stage 2 solves for
  the 6 arm DOFs to put the wrist where the human wrist is, Stage 3
  retargets fingers independently against the same episodes, and the two
  are concatenated at action-assembly time.

**How to apply (Stage 2/3/4)**:
- Stage 2: IK target is `attachment_site`; home keyframe `home` (the MJCF's
  only keyframe) is a safe seed. Joint order is
  `[shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]`; all
  revolute; joint limits `±2π` except elbow `±π`.
- Stage 3: treat the arm as solved; choose the dex hand without having to
  revisit the arm.
- Stage 4: the LeRobot action head predicts 6 arm DOFs + gripper + N finger
  DOFs. Dimensionality is fixed at 7+N.

The Stage 3 dex-hand choice (Inspire vs. Allegro vs. LEAP) is still an open
question; see "Open questions" above.

### R-002: Stage 2 — ARKit confidence threshold recalibration  *(answered 2026-04-07)*
**Original question** (surfaced 2026-04-07 from the §0.3 single-episode dump):
The `initial_plan.md` §2.1 plan filters at `min_confidence = 0.5`. On episode 0
no per-hand mean confidence reached even 0.7. Is the threshold wrong?

**Answer**: Yes, 0.5 is too aggressive — it throws out 42% of all samples,
much of which is real tracking. The distribution is **bimodal** with a clear
"untracked floor" mode at 0.000–0.003 (17.9% of samples) and a broad tracked
mode spanning 0.2 – 1.0.

Numbers (806,775 confidence samples — 25 joints × all frames × 277 episodes,
active hand only, basic_pick_place):

| stat | value |
|------|-------|
| min  | 0.000 |
| p05  | 0.003 |
| p10  | 0.003 |
| p25  | 0.257 |
| p50  | 0.614 |
| p75  | 0.816 |
| p90  | 0.895 |
| p99  | 0.996 |
| max  | 1.000 |
| mean | 0.529 |
| std  | 0.327 |

Single biggest unique value: **0.0032 with 15.58% of all samples** — this is
ARKit's "no signal" floor for joints it failed to track that frame. Anything
above ~0.05 is in the genuinely-tracked mode. The full discrete-floor +
broad-distribution shape is bimodal, valley between 0.05 and 0.20.

Threshold sweep — fraction of samples kept:
- `> 0.1`: **82.0%** ← recommended "tracked at all" threshold
- `> 0.2`: 78.8%
- `> 0.3`: 72.0%
- `> 0.5`: 58.0% ← original §2.1 plan; throws out a lot of real data
- `> 0.7`: 42.0%
- `> 0.8`: 27.9% ← too aggressive

**Recalibrated thresholds**:
- **`min_confidence = 0.1`** for "is this joint tracked at all?" — use this
  as the default reject filter in Stage 2 IK preprocessing. It removes the
  floor mode without dipping into real measurements.
- **`min_confidence = 0.5`** for "is this a *high quality* observation?" —
  use this for confidence-aware weighting (contribution pitch #1 in
  `initial_plan.md`), not for hard rejection.
- Compute confidence per-joint (not averaged across the whole hand). The
  wrist is generally tracked better than distal finger tips, so a hand-wide
  average would over-penalize "good wrist + flaky pinky tip" frames.

**How to apply (Stage 2)**: in `extract_wrist_trajectory` (and the analogous
finger extractor), gate on `confidences/<jointName> > 0.1`. When implementing
contribution #1 (confidence-aware filtering), use the per-joint score
directly as an IK task weight rather than thresholding twice.

---

## Open questions (resolve before the relevant stage)
- **Stage 3**: Does `dex-retargeting` ship an Inspire-hand config in `assets/`? If
  not, write a YAML for the Allegro hand instead — the dexterity story is the same.
- **Stage 4**: MuJoCo eval env for `basic_pick_place` — build from scratch, or reuse
  an existing one (mujoco_menagerie? lerobot envs)? Defer until Stage 4 actually
  starts; estimate time then.

---

## Stage-by-stage status

| Stage | Status      | Notes |
|-------|-------------|-------|
| 0     | done — 2026-04-07 | Test split downloaded + extracted; exploration + variance report; R-001/R-002 calibrated. (`part2` download is obsolete per D-005; concept reading is on the user, not Claude.) |
| 1     | done — 2026-04-08 | Full batch over all 277 `basic_pick_place` episodes: 271 `exact`, **6 `ransac_fallback` (stub)**, 0 failures. Reduction ratio median 2.50×, p95 8.28×, 61.7% of episodes > 2×, 24.5% > 4×. Inlier H-RMSE median 2.66 px, 92.4% < 10 px. Raw cam angle median 0.17°/frame (low-motion AVP wearer). See `doc.md` §8.5.5 for distributions. The 6 fallback episodes are documented under R-006; the vidstab path is still a stub and will be wired in only if Stage 4 metrics regress on them. |
| 2     | done — 2026-04-08 (first cut) | Full batch over all 277 episodes: **0 failures**, 62.7 s wall total (~0.23 s/ep). pos_err median distribution: mean 2.48 mm, p95 4.53 mm, **max 99.6 mm** (one bad ep). 96.8% of episodes have pos_err median < 5 mm. ori_err median 0.29° (median of medians). FIVER-collapse guard: **90.6%** of episodes clear ≥5/6 joints > 0.3 rad, 49.5% clear 6/6. No systemic FIVER collapse, but the 0.3-rad-on-all-6-joints threshold is tighter than necessary — per the Stage 4 §4.4 ablation we care about the reaching joints clearing it, which ~100% do. Stage 2 IK tail (8 episodes with p95 > 50 mm) does not overlap with the Stage 1 R-006 episodes — it's an IK-convergence issue, not a data issue. Investigation deferred; see `doc.md` §8.6.8. |
| 3     | not started | Dexterous finger retargeting via dex-retargeting. |
| 4     | not started | Train policy on `basic_pick_place`, ablation table. |

Mark each stage `in progress` when you start it and `done — YYYY-MM-DD` when the
deliverable from `initial_plan.md` exists.

---

## Things we are explicitly NOT doing (yet)

- **H2R visual diffusion**: requires paired real-robot data we don't have. Skipped
  per `initial_plan.md` rationale.
- **Training on all 194 EgoDex tasks**: stick to `basic_pick_place` for the
  ablation, per §4.2. Scale experiments belong to "potential original
  contributions" §4.
