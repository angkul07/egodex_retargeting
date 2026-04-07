# plan.md — working plan

This is the **living** plan. It tracks current state, deviations from
`initial_plan.md`, open questions, and decisions. Update it as work progresses.
The original 4-stage roadmap (Stages 0–4 + contributions + timeline) lives in
`initial_plan.md` and should not be edited.

---

## Current state — 2026-04-07 (Day 0)

**Stage**: 0 — Foundation + Data Exploration (in progress, deliverable produced)
**Last action**: Downloaded EgoDex `test.zip` (16.1 GB), extracted, wrote
`notebooks/00_explore_egodex.py` (single-episode dump) and
`notebooks/01_variance_report.py` (whole-task aggregate). Variance report on
277 `basic_pick_place` episodes: 0 collapsed, 70.4% within the 0.2–0.5 m target
range. CSV at `outputs/variance_report_basic_pick_place.csv`. See decision
**D-004** below — the schema in `initial_plan.md` §0.3 was wrong; the real
layout is everything-as-`/transforms/<joint>` and metadata as file attrs.
**Next action**: Stage 0 §0.1–0.2 conceptual reading (homography + DLS IK),
then move to Stage 1 — write `egostabilizer.py`. **No further EgoDex
downloads** (D-005); the test split is the only data we use.

### Environment status
- [x] uv project initialized, Python 3.13.
- [x] Base deps installed (`uv sync`): h5py, numpy, opencv-python, scipy, jupyter, etc.
- [ ] Stage 1 deps (`uv sync --group stage1`): vidstab.
- [ ] Stage 2 deps: mink, quadprog.
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
- **Stage 2**: Which arm URDF to target for IK? UR5e is the default in the mink
  examples and matches what FIVER used. Confirm with Sanskar before locking in.
- **Stage 3**: Does `dex-retargeting` ship an Inspire-hand config in `assets/`? If
  not, write a YAML for the Allegro hand instead — the dexterity story is the same.
- **Stage 4**: MuJoCo eval env for `basic_pick_place` — build from scratch, or reuse
  an existing one (mujoco_menagerie? lerobot envs)? Defer until Stage 4 actually
  starts; estimate time then.

---

## Stage-by-stage status

| Stage | Status      | Notes |
|-------|-------------|-------|
| 0     | in progress | Test split downloaded + extracted; exploration script + variance report done; concept reading + part2 download outstanding. |
| 1     | not started | EgoStabilizer (exact homography from extrinsics + RANSAC fallback). |
| 2     | not started | Action alignment (wrist trajectory → IK joint angles). |
| 3     | not started | Dexterous finger retargeting via dex-retargeting. |
| 4     | not started | Train policy on `basic_pick_place`, ablation table. |

Mark each stage `in progress` when you start it and `done — YYYY-MM-DD` when the
deliverable from `initial_plan.md` exists.

---

## Things we are explicitly NOT doing (yet)

- **H2R visual diffusion**: requires paired real-robot data we don't have. Skipped
  per `initial_plan.md` rationale. Revisit only if Sanskar provides paired data.
- **Training on all 194 EgoDex tasks**: stick to `basic_pick_place` for the
  ablation, per §4.2. Scale experiments belong to "potential original
  contributions" §4.
- **Confidence-aware filtering, finger-contact-aware IK weighting, skill
  segmentation, scale experiments, cross-embodiment transfer**: these are the
  five contribution pitches in `initial_plan.md`. Pick at most one to attempt
  before the internship — most likely **confidence-aware filtering**, since it's
  the cheapest and falls naturally out of Stage 2.
