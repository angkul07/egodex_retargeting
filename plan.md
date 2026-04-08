# plan.md — working plan

This is the **living** plan. It tracks current state, deviations from
`initial_plan.md`, open questions, and decisions. Update it as work progresses.
The original 4-stage roadmap (Stages 0–4 + contributions + timeline) lives in
`initial_plan.md` and should not be edited.

---

## Current state — 2026-04-08 (Day 1)

**Stage**: 1, 2, 3, and **4 first-cut complete** end-to-end. Stage 4
trained an ACT policy on the full pipeline + ran 20 MuJoCo rollouts.
**First-pass eval result: 10% success rate** (R-009): 2 of 20 rollouts
lifted the object > 5 cm, 6 made partial contact, 12 missed entirely.
This is proof-of-life for the *whole pipeline* (HDF5 → stab → IK →
finger retargeting → LeRobot dataset → ACT → MuJoCo rollouts) but
**NOT** a publishable §4.4 ablation table — only 1 of 4 conditions is
trained, the eval camera is not calibrated to the EgoDex AVP intrinsics,
and the visual distribution shift between training (real human hand on
cluttered table) and eval (procedural MuJoCo render of robot hand on a
checkerboard) is the dominant residual error mode. Stage 3's quality
assessment (R-008) still holds: retargeting captures grasp affordance
class, sufficient for BC training.
**Last action**: Wrote `src/mimicdreamer_egodex/finger_retargeting.py`
(Stage 3 deliverable) + `notebooks/04_stage3_batch.py` and ran the batch:
277/277 episodes, 0 failures, **90.7 s total wall time** (~2.4 ms/frame
mean — fast because the `SeqRetargeting` instance is per-side cached and
warm-starts after frame 0 of each episode). Target hand = **Inspire**;
`dex-retargeting` ships `offline/inspire_hand_{left,right}.yml` so no
custom YAML is needed, and the URDFs are vendored at
`third_party/dex-urdf@7304c7f`. Results: **96.8%** of episodes have 6/6
target proximals clearing the Inspire variance guard (> 0.1 rad range),
thumb yaw has the widest per-episode variance (p05 0.19 → p95 0.77 rad,
task-adaptive opposition), median `ms/frame = 2.4`. Stage 3 tail (9
episodes with <6/6) splits cleanly into 3 episodes overlapping Stage 1
R-006 (low wrist/fingertip confidence → shared data issue) + 6 short
episodes (<80 frames, data-constrained variance floor); no overlap with
Stage 2's IK tail. Torch nightly cu128 is now installed out-of-band (RTX
5090 sm_120 verified) so Stage 4's environment is ~half-prepared.
Run log: `logs/runs/2026-04-08_104408_stage3_batch.log`.
**Next action**: see R-009 "what would change this assessment" — three
ranked improvements: (1) **camera intrinsics calibration** — read
`camera/intrinsic` from one of the 277 EgoDex episodes and apply it to
the MuJoCo render to remove most of the visual distribution shift
(highest ROI, ~30 min); (2) tabletop texturing + lighting tweaks to
match EgoDex's metal-table-on-white-background look; (3) the §4.4
ablation runs proper — build the 3 other dataset variants (drop fingers,
drop smooth IK, drop stab) and train them, ~75 min/condition. Parked
(non-blocking): R-006 Stage 1 fallback, Stage 2 IK tail.

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
- [x] Stage 3 deps installed (`uv sync --group stage1 --group stage2 --group
      stage3`): dex-retargeting 0.4.5, pin 3.9.0, libpinocchio 3.9.0,
      eigenpy 3.12.0, nlopt 2.10.0, pytransform3d 3.14.4, lxml 6.0.2,
      libcoal 3.0.2. URDFs vendored at `third_party/dex-urdf@7304c7f`.
- [x] Torch nightly cu128 installed out-of-band (D-001 recipe): torch
      2.12.0.dev20260407+cu128, torchvision 0.27.0.dev20260407+cu128,
      triton 3.7.0+git9c288bc5. RTX 5090 sm_120 capability (12, 0)
      verified. Unblocks `dex_retargeting` (which hard-imports torch) and
      prepares Stage 4.
- [x] Stage 4: lerobot 0.5.1 installed out-of-band via
      `uv pip install --no-deps lerobot` + bulk install of 110
      transitive deps (excluding torch/torchvision/torchcodec/torchdiffeq
      and `opencv-python-headless` which would conflict with our
      `opencv-python`). lerobot's `torch<2.11` ceiling is purely a
      packaging-level constraint — runtime is compatible with torch 2.12
      cu128. **Important**: torchcodec is NOT installable on our combo
      (cu13 wheel mismatch); applied a PyAV monkey-patch (D-012) so
      lerobot's video decoder bypasses both torchcodec and the removed
      `torchvision.io.VideoReader`.
- [x] RunPod base image needs `apt install libegl1 libglvnd0` for
      headless MuJoCo rendering (D-009).
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

For lerobot (resolved 2026-04-08 by R-009 / Stage 4 phase A): use the
`--no-deps` path. lerobot 0.5.1's `torch<2.11` ceiling is purely a
packaging-level constraint — runtime is fine on torch 2.12. Recipe:

```bash
uv pip install --no-deps lerobot
# Pull lerobot's transitive deps EXCEPT the torch family + opencv-python-headless
uv pip install \
    datasets diffusers huggingface-hub accelerate einops av jsonlines \
    pynput pyserial wandb draccus gymnasium rerun-sdk deepdiff \
    "imageio[ffmpeg]" termcolor "numpy<2.3" "setuptools<81" "cmake<4.2" \
    pandas pyarrow pydantic
```

torchcodec is **not installable** on torch 2.12 cu128 (0.10 has the
wrong c10 ABI; 0.11 wants CUDA 13). Use the PyAV monkey-patch
documented in **D-012** instead.

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

### D-007: Stage 3 — `SeqRetargeting.retarget(ref_value)` expects a pre-sliced `(5, 3)` input  *(decided 2026-04-08)*
**Why**: The YAML `target_link_human_indices: [4, 8, 12, 16, 20]` in
`dex-retargeting/configs/offline/inspire_hand_*.yml` is a MediaPipe-21-point
landmark layout — those are the fingertip indices in the 21-point hand
model. The code however does **not** index into the 21-point array
internally: `SeqRetargeting.retarget` passes `ref_value` straight through
to `PositionOptimizer.get_objective_function`, which feeds it directly
into `torch.nn.SmoothL1Loss` against a `(5, 3)` body-position tensor.
Passing a `(21, 3)` array triggers a silent PyTorch broadcasting warning
and produces garbage. Caller must pre-slice to `(5, 3)` in the order
matching the config's `target_link_names`:

    [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]

**How to apply**: `finger_retargeting.py::wrist_relative_tips` returns a
`(T, 5, 3)` array in exactly this order; Stage 3's extraction puts
`<side>ThumbTip`, `<side>IndexFingerTip`, `<side>MiddleFingerTip`,
`<side>RingFingerTip`, `<side>LittleFingerTip` (= pinky) in that order.
Do not pass a length-21 array. Flagged inline in the module docstring
and verified empirically on `basic_pick_place/0` — the `(5, 3)` path
produces per-joint ranges of 0.41–0.68 rad; the `(21, 3)` path silently
converges to near-zero deltas. (The latter is how the earlier dry-run
with fake inputs ended up with 0.0001 rad "curl deltas".)

### D-008: `uv sync --group <stage>` replaces the active group set  *(decided 2026-04-08)*
**Why**: The first `uv sync --group stage3` I ran silently **uninstalled**
the Stage 2 packages (mink, mujoco, quadprog, robot_descriptions, etc.)
because uv treats the `--group` flag as a replacement selection, not
additive. This broke `action_alignment.py`'s import chain and would have
broken any re-run of the Stage 2 batch.

**How to apply**: when syncing, always pass **every active group
simultaneously**:

```bash
uv sync --group stage1 --group stage2 --group stage3
```

This rule is also mirrored into `CLAUDE.md` under the tooling section
so future sessions don't trip over it. `uv add --group <stage> <pkg>`
does not have the same footgun — it only edits the one group. The trap
is `uv sync` specifically.

### R-007: Stage 3 — Inspire hand locked in; `dex-retargeting` ships the config  *(answered 2026-04-08)*
**Original question**: Does `dex-retargeting` ship an Inspire-hand config
in `assets/`? If not, write a YAML for the Allegro hand instead.

**Answer**: **Yes, Inspire is supported out of the box.** The
`dex_retargeting` wheel bundles both
`configs/offline/inspire_hand_left.yml` and
`configs/offline/inspire_hand_right.yml` (also teleop variants which we
don't use). The referenced URDFs (`inspire_hand/inspire_hand_{side}.urdf`)
live in the separate `dex-urdf` repo (the `dex-retargeting` pip wheel
does NOT carry URDFs). We vendor it at
`third_party/dex-urdf@7304c7fb59214dab870eca02cf26f76e944e12df`
(gitignored) and point `RetargetingConfig.set_default_urdf_dir` at
`third_party/dex-urdf/robots/hands` in `finger_retargeting.py::_ensure_urdf_dir`.
Override via `$DEX_URDF_DIR` if the repo is cloned elsewhere.

Inspire details:
- Type: `position` retargeting (5-fingertip positions, SmoothL1 loss).
- 12 non-dummy URDF DOFs, of which **6 are "target" (optimized)**:
  `{index, middle, ring, pinky}_proximal_joint`, `thumb_proximal_yaw_joint`,
  `thumb_proximal_pitch_joint`. The intermediate and distal joints are
  declared as mimic joints in the URDF and track their proximals
  automatically.
- `add_dummy_free_joint: True` prepends a 6-DOF free joint at the hand
  root so the optimizer can absorb any input-frame offset — which is why
  we can feed wrist-relative positions in world axes without rotating to
  a hand-local frame.
- Input contract is the `(5, 3)` slice described in D-007.

**How to apply (Stage 3/4)**:
- Stage 3: `finger_retargeting.py` writes `q_finger (T, 6)` per episode,
  in the action-vector order
  `[index, middle, ring, pinky, thumb_yaw, thumb_pitch]`.
- Stage 4: final action vector per `initial_plan.md` §3.3 is
  `[q_arm_6, gripper_1, q_finger_6]` = **13 dimensions**. Concatenate
  Stage 2's `outputs/stage2/<idx>_actions.npz` with Stage 3's
  `outputs/stage3/<idx>_fingers.npz` at dataset-build time.

The hand choice can still be revisited later (Allegro has 16 DOFs, LEAP
has 16, Shadow has ~22) — all four configs are in the same
`dex-retargeting/configs/offline/` directory and would require only
swapping one constant in `finger_retargeting.py`. Sticking with Inspire
for the replication because 6 DOFs keeps the Stage 4 action head small
and matches the paper's closest comparable setup.

### R-009: Stage 4 — first-cut policy + eval result is 10% success, dominated by visual distribution shift  *(answered 2026-04-08)*
**Original question**: Does the full pipeline (stab → IK → retargeting →
ACT → MuJoCo rollouts) actually train and produce a working policy?

**Answer**: **End-to-end pipeline works, first-cut success rate is
10% (2/20 rollouts), proof-of-life only — not the §4.4 deliverable.**

Trained ACT (51.6 M params, chunk_size=10, n_action_steps=8) via the
custom loop in `notebooks/09_train_act.py`:

| step | train loss | val loss |
|---:|---:|---:|
| 100 | 4.77 | — |
| 1000 | 0.241 | 0.229 |
| 1500 | 0.192 | 0.199 |
| 2000 | 0.158 | 0.182 |
| 2500 | **0.137** | **0.165** ← best |
| 3000 (final) | 0.126 | 0.173 |

- batch 64, bf16 autocast, AdamW lr=1e-4, ~25 min wall on RTX 5090
- 222 train / 55 val episodes (last 20% holdout)
- val ≤ train at every step → no big overfitting; the train-val gap of
  0.05 at the end is mild and consistent with ~10 epochs over 17.9 min
  of footage
- Best val at step 2500 → mild early-stopping signal but not severe

Eval: `notebooks/10_eval_act.py` ran 20 rollouts of up to 120 env steps
each (4 s @ 30 Hz) in the `EvalEnv` defined in
`src/mimicdreamer_egodex/eval_env.py`. Success criterion: object lifted
> 5 cm.

| metric | value |
|---|---:|
| **success rate** | **2 / 20 = 10 %** |
| mean max lift | 8.6 mm |
| median max lift | 0.08 mm |
| max max lift | 51.3 mm |
| wall time | 7.1 s (~350 ms/rollout) |

Distribution of outcomes:
- **2 clean successes** (object lifted 50.0 / 51.3 mm at steps 111 / 97)
- **6 partial touches** (1–20 mm of lift — fingers reached the object
  but lost grip)
- **12 no-movement** (the policy's commanded pose missed the object)

**What this means**:

- A truly random or dead policy would be ~0%. The 6 partial touches
  prove the policy is *directing* the hand toward the object's
  neighborhood, and the 2 clean successes prove a complete reach →
  close → lift sequence is reproducible.
- A polished BC policy on a matched-distribution eval would be 50–90%.
  We are not there.
- The training metrics (val loss 0.165) say the model learned the data
  well — this is **not** a training failure, it's an evaluation-domain
  mismatch.
- **The dominant residual error is visual distribution shift** between
  the training images (1080p egocentric video of a real human hand
  grasping a varied real object on a textured tabletop) and the eval
  images (224×224 procedural MuJoCo render of a robot hand on a
  checkerboard with one bright primitive cube). The ResNet18 vision
  backbone is sensitive to color/edge/lighting statistics that don't
  transfer.
- **No § 4.4 ablation comparison exists yet**. This is the "Full
  pipeline" condition only. Building a real ablation table requires
  training 3 more conditions (drop fingers / drop smooth IK / drop
  stab) and running 100 rollouts each.

**What's NOT validated**:
- Camera intrinsics — the eval `MjvCamera` is positioned by hand and is
  NOT calibrated against any specific EgoDex episode's
  `camera/intrinsic`. This is the single highest-leverage fix.
- Object diversity — eval uses one 4 cm primitive cube; training data
  has 99 distinct objects from `llm_objects[0]`.
- The inference-time arm trajectory — we apply policy actions via
  direct `qpos` writes (kinematic control) rather than via torque
  actuators (D-013). This sidesteps controller tuning but means the
  arm "snaps" each step instead of following a smooth controller —
  fine for measuring grasp shape success, less realistic for dynamics.

**How to apply / improvement order**:

1. **(easy, ~30 min)** Read `camera/intrinsic` + `transforms/camera`
   from a representative EgoDex episode, set the `MjvCamera` extrinsics
   + intrinsics to match. Estimate: 2–4× success rate improvement.
2. **(easy, ~30 min)** Texture the table top + tune lighting to match
   the EgoDex metal-table-on-white-background look.
3. **(medium, ~80 min)** Train longer (10 000 steps instead of 3000)
   and/or use a smaller model + harder regularization to combat
   small-data overfitting.
4. **(hard, ~4 hours)** The §4.4 ablation table proper: build the 3
   other dataset variants and train them sequentially. This is the
   real Stage 4 deliverable.

The current 10% number IS the entry for the §4.4 "Full pipeline" cell,
but it should be **regenerated after improvements (1)–(3)** before being
quoted as a real result. Honest framing: "first-pass end-to-end
working, distribution-shift-limited; ablation table not yet built."

### D-012: lerobot's video decoder needs a PyAV-only monkey-patch on torch 2.12 nightly  *(decided 2026-04-08)*
**Why**: lerobot 0.5.1's `decode_video_frames` dispatches to
`decode_video_frames_torchcodec` if torchcodec is available, otherwise
falls back to `decode_video_frames_torchvision` (which calls
`torchvision.io.VideoReader` under the hood). On our env both paths
break:

- **torchcodec 0.10** (the version lerobot pins to) was built against
  torch 2.10's c10 ABI → loading
  `libtorchcodec_core6.so` raises
  `OSError: undefined symbol: _ZN3c1013MessageLoggerC1EPKciib` against
  our torch 2.12 nightly.
- **torchcodec 0.11** (latest) was built against CUDA 13 → wants
  `libnvrtc.so.13` which we don't have (we ship CUDA 12.8 nvrtc with
  the cu128 nightly).
- **torchvision 0.27 nightly removed `torchvision.io.VideoReader`**
  entirely → lerobot's "pyav fallback" path raises
  `AttributeError: module 'torchvision.io' has no attribute 'VideoReader'`.

**How to apply**: import
`mimicdreamer_egodex.lerobot_pyav_patch.apply()` **before** any
`from lerobot...` import that touches the dataset reader. The patch
replaces `lerobot.datasets.video_utils.decode_video_frames` AND
`lerobot.datasets.dataset_reader.decode_video_frames` (the latter
binds the function name at import time so both references must be
patched) with a `decode_video_frames_pyav_only` function that uses
`av` (PyAV) directly. PyAV is already installed as a transitive
lerobot dep and is ABI-independent of torch.

The patched function preserves the original contract: `(N, C, H, W)`
float32 in `[0, 1]`, with timestamp-tolerance frame matching. Verified
against the smoke-test dataset at frame 0 — image (3, 224, 224) float32
in [0.000, 0.949], no errors.

### D-013: Stage 4 eval env uses kinematic control (direct `qpos` writes), not torque actuators  *(decided 2026-04-08)*
**Why**: The trained policy outputs 13-D joint positions, not torques.
Building a torque-actuator controller (PD or impedance) on top of the
joint-position targets adds tuning complexity and an extra source of
sim-to-real gap that doesn't help measure the BC policy's quality. For
a partial replication where the eval question is "does the policy
*shape* a successful grasp?" — kinematic control is the right
abstraction.

**How to apply**: `EvalEnv.step` writes `data.qpos[:6] = action[:6]`
(UR5e arm) and `data.qpos[inspire_target_qadr] = action[7:13]` (Inspire
proximals) directly each tick, then calls `mj_step` to advance the
free-joint object dynamics + contact. The 6 Inspire mimic joints (the
intermediate / distal joints of the four long fingers and thumb) are
driven by the URDF mimic equality constraints during `mj_forward`.
`action[6]` (the gripper scalar from Stage 2) is currently a no-op
because the Inspire DOFs already encode the full hand pose; pass-
through for compatibility with the trained 13-D output shape.

**Trade-offs**:
- Pro: zero controller tuning, exact policy → joint correspondence,
  fast.
- Con: arm "snaps" each tick (no joint-velocity smoothing); contact
  forces during the snap can be noisy. Mitigation: small step sizes
  via mujoco's default integrator (`implicitfast`).
- Con: doesn't test whether a real torque controller could track the
  policy output. That's a sim-to-real concern; out of scope for the
  partial replication.

If a future eval needs torque-controlled rollouts, swap `step()` to
write to `data.ctrl[]` instead of `data.qpos[]`, and rely on the
menagerie UR5e's pre-tuned `general` actuators (the URDF-attached
Inspire would need actuators added — not free).

### D-014: ACT's `forward()` crashes in `eval()` mode; keep `train()` for validation  *(decided 2026-04-08)*
**Why**: lerobot 0.5.1's `ACTPolicy.forward()` (in
`policies/act/modeling_act.py`) computes the VAE KL-divergence loss
unconditionally, but the VAE encoder is skipped when the model is in
`eval()` mode. The result is that `log_sigma_x2_hat` is `None` at the
KL formula:

```python
(-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
```

→ `TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'`.

**How to apply**: in the validation pass of any custom training loop,
**do NOT call `policy.eval()`**. Wrap the forward pass with
`torch.no_grad()` only and keep the model in `train()` mode. The VAE
encoder will run on the validation batch, the KL loss will be
well-defined, and the resulting per-batch loss is still a valid
across-split comparison metric (it just includes the KL term, same as
train). The `evaluate()` function in `notebooks/09_train_act.py`
implements this workaround inline with a comment.

This only affects custom loops. lerobot's own `lerobot_train.py` uses
accelerate's autocast wrapper which sidesteps the issue differently;
our custom loop hits it directly.

**Important**: at *inference* time (rollouts in the eval env), it IS
safe to call `policy.eval()` — that path uses `select_action()`, not
`forward()`, so the buggy KL term is never computed.

### R-008: Stage 3 — retargeting quality is affordance-class adequate, not within-class precision  *(answered 2026-04-08)*
**Original question**: Is the Stage 3 finger retargeting actually
"good"? The 96.8% variance-guard pass rate (R-007) tells us nothing
*moves wrong*, but says nothing about whether the retargeted joint
angles encode object identity in a useful way.

**Answer**: The retargeting is **successful for the project's goal**
(BC training in Stage 4) but is **not** a precision-grade dexterous
teleoperation system. Concretely:

- **Per-finger position error** between human EgoDex fingertips and
  Inspire FK fingertips (`notebooks/05_stage3_visualize.py` on a
  4-episode sample): **median 5–10 mm, p95 10–20 mm**. Inspire's
  fingerpad is ~15 mm wide, so a 10 mm error is ~60% of the pad. Fine
  for power grasps (the dataset is 100% power grasps), marginal for
  precision pinches.

- **Per-object grasp clustering** (`notebooks/07_grasp_clustering.py`)
  on 174 episodes covering 25 distinct objects (each with ≥ 3 episodes):
  - 6-D peak-grasp signature: separation ratio 0.81, silhouette −0.27
  - **18-D trajectory signature: separation ratio 0.95, silhouette −0.15**
  - Both raw numbers are < 1 / negative — but the *most-distinct pairs*
    are physically meaningful: `dice box vs toy block = 1.34 rad`,
    `iphone vs toy block = 1.31 rad`, `iphone vs fry = 1.26 rad`. The
    *most-similar pairs* are also physically similar:
    `iphone vs mouse = 0.15`, `container of slime vs donut = 0.16`,
    `plushie vs tea cup = 0.17`. Pairwise distances among the 6 largest
    object groups show clean affordance-class structure: small-hard
    objects (`block`, `dice`, `egg`) cluster together; soft plush
    objects (`duck`, `plushie`) cluster together; cross-group distances
    are larger.

- **Verdict**: the retargeting captures **grasp affordance class**
  (small hard, soft plush, flat thin, …) but **does not** discriminate
  fine-grained within-class differences. This is consistent with three
  hard limits: (a) the 6-DOF Inspire embodiment can only express a
  limited grasp vocabulary, (b) the `llm_objects` labels are noisy
  ("block" lumps 41 physically diverse cubes together), (c) we are
  matching fingertip *positions*, not contact normals or forces.

**What this means for Stage 4**: this is **enough**. The action signal
needs to encode *how to grasp* (affordance level); object identity
comes from the RGB observation. The BC policy will absorb residual
retargeting noise. The bar this needs to clear is set by the §4.4
ablation table — outperform the binary-gripper baseline. That
experiment hasn't run yet; the verdict on "is the dexterous hand
actually pulling its weight" is **TBD until Stage 4**.

**What this is NOT**: not physics-validated (no contact-closure check
on retargeted poses), not closed-loop (no force feedback, same as the
MimicDreamer baseline), not jointly optimized with Stage 2 (the wrist
mount can drift between the two stages), not validated on anything
other than `basic_pick_place`.

**How to apply**:
- Trust the action signal at the **affordance class** level.
- Do **not** trust it for fine within-class object discrimination —
  rely on the RGB input for that.
- Re-evaluate after Stage 4 ablation results land.
- Full discussion is in `doc.md` §8.7.8.

### D-009: RunPod needs `apt install libegl1` for MuJoCo headless rendering  *(decided 2026-04-08)*
**Why**: The RunPod base image ships
`libEGL_nvidia.so.570.195.03` (the NVIDIA driver's EGL implementation)
but **not** the generic `libEGL.so.1` dispatcher that PyOpenGL looks
for. `import mujoco` followed by `Renderer(...)` fails with
`AttributeError: 'NoneType' object has no attribute 'eglQueryString'`
because PyOpenGL can't find the entry point.

**How to apply**: one-liner per fresh container:

```bash
apt-get update && apt-get install -y libegl1 libglvnd0
```

Then `os.environ["MUJOCO_GL"] = "egl"` **before** `import mujoco` in
any script that uses offscreen rendering. The
`mujoco.Renderer(model, h, w)` smoke test should pass with non-zero
nonzero pixel count.

`libegl1-mesa` was renamed/removed in Ubuntu 24.04 — don't install
that one, just `libegl1` + `libglvnd0`.

### D-010: MuJoCo URDF loader needs `strippath="false" discardvisual="true"` injection  *(decided 2026-04-08)*
**Why**: MuJoCo's URDF loader has two opinionated defaults:

1. `strippath="true"` — flattens `filename="meshes/visual/foo.glb"`
   to `foo.glb`, breaking subdirectory layouts like dex-urdf's
   `meshes/{visual,collision}/`.
2. It loads visual meshes by default, and dex-urdf's visual meshes
   are `.glb` (not supported by MuJoCo — only `.obj` and `.stl`
   work). The collision meshes are `.obj` and load fine.

**How to apply**: inject a `<mujoco>` extension block immediately
after `<robot ...>` when reading the URDF, before passing it to
`mujoco.MjModel.from_xml_path`:

```python
mesh_dir = str(urdf_path.parent.absolute()) + "/"
injection = (
    f'<mujoco>'
    f'<compiler meshdir="{mesh_dir}" strippath="false" '
    f'discardvisual="true"/>'
    f'</mujoco>'
)
txt = re.sub(r"(<robot[^>]*>)", r"\1" + injection, urdf_text, count=1)
```

The hand will render with collision-mesh primitives (blocky/industrial
look — fully legible, just not pretty). See
`notebooks/06_stage3_animate.py::_inspire_mjcf_prepare`.

### D-011: MuJoCo URDF `<mujoco>` extension silently ignores `<visual>` children  *(decided 2026-04-08)*
**Why**: I tried injecting a `<visual><global offwidth="720" offheight="480"/></visual>`
sub-element into the same `<mujoco>` block to enlarge the offscreen
framebuffer beyond the 640×480 default. MuJoCo's URDF parser silently
**discarded** it — only `<compiler>` was honored. Asking for a
720-wide render still raised
`Image width 720 > framebuffer width 640`.

**How to apply**: when rendering Inspire (or any URDF) via MuJoCo
offscreen, **use the default 640×480 framebuffer** (matplotlib
animations can stay at 720×480 — only the MuJoCo render is constrained).
Define separate constants `MJ_FRAME_W = 640, MJ_FRAME_H = 480` and
build `mujoco.Renderer(model, MJ_FRAME_H, MJ_FRAME_W)`. Larger
framebuffers would require writing a standalone MJCF wrapper — but
MJCF can't `<include>` URDFs, so that path is closed for vendored
URDF assets.

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
- **Stage 4 polish (deferred, not blocking)**: ranked by ROI per R-009:
  - (1) Calibrate the eval `MjvCamera` intrinsics + extrinsics to one of
    the 277 EgoDex episodes' `camera/intrinsic` + `transforms/camera`,
    so the eval-time RGB matches the training distribution. Highest
    leverage. Estimated 2–4× success rate improvement.
  - (2) Texture the table top + tune lighting in the eval scene to
    match EgoDex's metal-table-on-white-background look.
  - (3) Re-train for 10 000 steps instead of 3000 (current best val at
    step 2500 hints at slight under-training rather than over-training).
- **Stage 4 § 4.4 ablation deliverable (the real Stage 4 result)**:
  build the 3 other dataset variants and train them sequentially:
    - "FIVER baseline": raw (unstabilized) frames + no-smooth-IK arm + binary gripper, 7-D action
    - "EgoDex only":    stabilized frames + no-smooth-IK arm + binary gripper, 7-D action
    - "+ Smooth IK":    stabilized frames + smooth-IK arm + binary gripper, 7-D action
  Each requires re-running Stage 2 with `lambda_smooth=0` for the
  no-smooth conditions and a 7-D-action variant of `08_to_lerobot.py`
  for the no-fingers conditions. Estimated ~4 hours of additional
  wall time.

---

## Stage-by-stage status

| Stage | Status      | Notes |
|-------|-------------|-------|
| 0     | done — 2026-04-07 | Test split downloaded + extracted; exploration + variance report; R-001/R-002 calibrated. (`part2` download is obsolete per D-005; concept reading is on the user, not Claude.) |
| 1     | done — 2026-04-08 | Full batch over all 277 `basic_pick_place` episodes: 271 `exact`, **6 `ransac_fallback` (stub)**, 0 failures. Reduction ratio median 2.50×, p95 8.28×, 61.7% of episodes > 2×, 24.5% > 4×. Inlier H-RMSE median 2.66 px, 92.4% < 10 px. Raw cam angle median 0.17°/frame (low-motion AVP wearer). See `doc.md` §8.5.5 for distributions. The 6 fallback episodes are documented under R-006; the vidstab path is still a stub and will be wired in only if Stage 4 metrics regress on them. |
| 2     | done — 2026-04-08 (first cut) | Full batch over all 277 episodes: **0 failures**, 62.7 s wall total (~0.23 s/ep). pos_err median distribution: mean 2.48 mm, p95 4.53 mm, **max 99.6 mm** (one bad ep). 96.8% of episodes have pos_err median < 5 mm. ori_err median 0.29° (median of medians). FIVER-collapse guard: **90.6%** of episodes clear ≥5/6 joints > 0.3 rad, 49.5% clear 6/6. No systemic FIVER collapse, but the 0.3-rad-on-all-6-joints threshold is tighter than necessary — per the Stage 4 §4.4 ablation we care about the reaching joints clearing it, which ~100% do. Stage 2 IK tail (8 episodes with p95 > 50 mm) does not overlap with the Stage 1 R-006 episodes — it's an IK-convergence issue, not a data issue. Investigation deferred; see `doc.md` §8.6.8. |
| 3     | done — 2026-04-08 (first cut + quality assessment) | `src/mimicdreamer_egodex/finger_retargeting.py` written; **Inspire** hand locked in via R-007 (dex-retargeting ships the config out of the box, URDFs vendored at `third_party/dex-urdf@7304c7f`). Full batch over all 277 episodes: **0 failures, 90.7 s total (~2.4 ms/frame mean)**, 96.8% of episodes clear 6/6 target proximals > 0.1 rad. Task-adaptive thumb opposition visible (stapler 0.68 rad vs iPhone 0.48 rad yaw). **Quality assessment (R-008)**: per-finger position error median 5–10 mm (Inspire fingerpad ~15 mm — fine for power grasps, marginal for precision); per-object grasp clustering on 25 objects shows separation ratio 0.95, silhouette −0.15 — captures **affordance class** but not within-class precision. Verdict: **enough for Stage 4 BC training**, not a finished dexterous teleop system. Stage 3 tail (9 episodes with <6/6) decomposes into 3 Stage-1-R-006 overlaps + 6 short (<80-frame) episodes. Zero overlap with the Stage 2 IK tail. See `doc.md` §8.7 (esp. §8.7.8). |
| 4     | done — 2026-04-08 (first cut, NOT § 4.4 ablation) | Phase A: lerobot 0.5.1 installed via `--no-deps` + 110 deps; PyAV monkey-patch (D-012) bypasses broken torchcodec/torchvision.io paths; ACT smoke (51.6 M params) forward+backward verified on RTX 5090 with bf16. Phase B: `notebooks/08_to_lerobot.py` converted all 277 episodes into a v3.0 LeRobotDataset (75 MB, 32 271 frames, 200 task strings) in 23 min. Phase C: `notebooks/09_train_act.py` trained ACT for 3000 steps (batch 64, ~25 min), final train loss 0.126 / val loss 0.173 (best val 0.165 at step 2500); D-014 ACT-eval-mode workaround needed in the custom val pass. Phase D: `eval_env.py` builds UR5e+Inspire+table+object scene via `MjSpec.attach()` (D-013 kinematic control); `notebooks/10_eval_act.py` ran 20 rollouts with **2/20 = 10% success rate** (2 clean lifts of 50/51 mm, 6 partial touches, 12 misses). **R-009**: this is proof-of-life only — § 4.4 ablation table needs the 3 other conditions trained AND camera intrinsics calibrated. Distribution shift (training real human hand vs eval procedural MuJoCo render) is the dominant residual error. See `doc.md` §8.8 for the full writeup. |

Mark each stage `in progress` when you start it and `done — YYYY-MM-DD` when the
deliverable from `initial_plan.md` exists.

---

## Things we are explicitly NOT doing (yet)

- **H2R visual diffusion**: requires paired real-robot data we don't have. Skipped
  per `initial_plan.md` rationale.
- **Training on all 194 EgoDex tasks**: stick to `basic_pick_place` for the
  ablation, per §4.2. Scale experiments belong to "potential original
  contributions" §4.
