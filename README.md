# mimicdreamer-egodex

Partial replication of [MimicDreamer](https://arxiv.org/abs/2509.22199) on the
[EgoDex](https://arxiv.org/abs/2505.11709) dataset, with a dexterous-finger
retargeting extension.

**Status (2026-04-08): Stages 0–4 first-cut complete on the full 277-episode
`basic_pick_place` test split.** End-to-end pipeline (HDF5 → stabilized frames
→ smooth-IK arm → Inspire-hand finger retargeting → LeRobot dataset → ACT
training → MuJoCo rollouts) is alive. First-pass eval is **10 % rollout
success**, dominated by visual distribution shift between training (real
EgoDex egocentric video) and eval (procedural MuJoCo render). The §4.4
ablation table is **not** built yet — see `plan.md` R-009 for the honest
assessment and what's left.

Documentation map:
- `initial_plan.md` — original 4-stage roadmap (read-only).
- `plan.md` — living plan, decisions (D-/R- entries), current status.
- `doc.md` — data-derived reference: real EgoDex HDF5 schema, coordinate
  frame (y-up, not z-up), calibrated confidence + table-distance thresholds,
  per-stage results. **When `doc.md` and `initial_plan.md` disagree,
  `doc.md` is correct.**
- `CLAUDE.md` — Claude Code conventions for working in this repo.

Scope: only the EgoDex **test split** (16.1 GB, 111 tasks, ~277
`basic_pick_place` episodes). No `part1.zip` / `part2.zip` downloads. See
`plan.md` D-005.

## Hardware target

NVIDIA RTX 5090 (Blackwell, sm_120) on RunPod. CUDA 12.8+. PyTorch stable
wheels do **not** support sm_120 — the cu128 nightly build is required.
Other GPUs that support a recent torch nightly should work too.

## Setup on a fresh machine

Prereqs: `uv` (https://docs.astral.sh/uv/), CUDA 12.8 driver, `git`.

### 1. System packages (one-time, root)

MuJoCo's headless rendering needs `libEGL.so.1` (the generic dispatcher),
which the RunPod base image doesn't ship even though NVIDIA's EGL backend
is present. Without this, `import mujoco` + `Renderer(...)` fails inside
Stages 3 / 4 visualizations and the Stage 4 eval rollouts.

```bash
apt-get update && apt-get install -y libegl1 libglvnd0
```

See `plan.md` D-009.

### 2. Repo + base + stage-grouped Python deps

```bash
git clone <repo-url> mimicdreamer-egodex
cd mimicdreamer-egodex

# Install Python 3.13 + base deps (Stage 0).
uv sync

# Stage-grouped deps. Pass EVERY active group on each `uv sync` — the
# `--group` flag REPLACES the active set rather than appending to it
# (see plan.md D-008). Active group set right now: stage1+stage2+stage3.
uv sync --group stage1 --group stage2 --group stage3
```

### 3. Torch nightly cu128 (Stage 4 prep + Stage 3 enablement)

`torch` and `torchvision` are NOT in `pyproject.toml` because the cu128
nightly resolver chain doesn't fit through `uv add` cleanly (see
`plan.md` D-001). Install out-of-band after the `uv sync`:

```bash
uv pip install --pre \
    --index-url https://download.pytorch.org/whl/nightly/cu128 \
    torch torchvision
```

This is also a hard requirement for `dex-retargeting` (Stage 3) — its
package init imports torch. Verify:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: 2.12.0.dev... True NVIDIA GeForce RTX 5090
```

### 4. lerobot (Stage 4)

lerobot 0.5.1's `torch<2.11` ceiling is purely a packaging-level constraint;
the runtime is fully compatible with torch 2.12 cu128. Install via
`--no-deps` and bulk-install the transitive deps that don't conflict with
our torch:

```bash
uv pip install --no-deps lerobot
uv pip install \
    datasets diffusers huggingface-hub accelerate einops av jsonlines \
    pynput pyserial wandb draccus gymnasium rerun-sdk deepdiff \
    "imageio[ffmpeg]" termcolor "numpy<2.3" "setuptools<81" "cmake<4.2" \
    pandas pyarrow pydantic
```

**Do NOT install torchcodec** — both the version lerobot pins (0.10) and
the latest (0.11) are unbuildable on torch 2.12 cu128 (c10 ABI mismatch
and CUDA 13 wheel mismatch respectively). Instead, our code installs a
PyAV-only monkey-patch over `lerobot.datasets.video_utils.decode_video_frames`
at runtime — see `src/mimicdreamer_egodex/lerobot_pyav_patch.py` and
`plan.md` D-012. Any new training/eval script must call
`mimicdreamer_egodex.lerobot_pyav_patch.apply()` **before** the first
`from lerobot...` import.

### 5. Vendored dex-urdf assets (Stage 3 hand URDFs)

`dex-retargeting` ships YAML retargeting configs but **not** the URDFs
they reference. Clone `dex-urdf` separately at the pinned commit:

```bash
mkdir -p third_party
git clone https://github.com/dexsuite/dex-urdf.git third_party/dex-urdf
git -C third_party/dex-urdf checkout 7304c7fb59214dab870eca02cf26f76e944e12df
```

`third_party/` is gitignored. Override the location with `$DEX_URDF_DIR`
if you cloned it elsewhere.

## Downloading EgoDex

Only the **test split** is needed:

```bash
mkdir -p data
curl -L -C - -o data/test.zip https://ml-site.cdn-apple.com/datasets/egodex/test.zip
# 16.1 GB; ~30–45 min on a typical RunPod box.

# RunPod images often ship without the system `unzip` binary; use Python:
uv run python -c "import zipfile; zipfile.ZipFile('data/test.zip').extractall('data/')"
```

This produces `data/test/<task>/<idx>.{hdf5,mp4}` for 111 task categories
(~6,600 files). Episode HDF5s are tiny (~0.5–1 MB each); the bulk is the MP4s.

We do **not** download `part1.zip` or `part2.zip`. Stage 4 policy training
runs on the test split's `basic_pick_place` task only (~277 episodes ≈ 18
minutes @ 30 Hz). Rationale and implications: `plan.md` D-005, `doc.md` §0.

## Repo layout

```
.
├── CLAUDE.md             # Claude Code conventions
├── README.md             # this file
├── initial_plan.md       # 4-stage roadmap (read-only reference)
├── plan.md               # living plan, decisions, status
├── doc.md                # data-derived reference (schema, conventions, results)
├── pyproject.toml        # uv-managed base + stage-grouped deps
├── uv.lock
├── src/mimicdreamer_egodex/
│   ├── egostabilizer.py       # Stage 1
│   ├── action_alignment.py    # Stage 2 (UR5e IK via mink)
│   ├── finger_retargeting.py  # Stage 3 (Inspire via dex-retargeting)
│   ├── eval_env.py            # Stage 4 MuJoCo eval scene (UR5e+Inspire+table+object)
│   └── lerobot_pyav_patch.py  # Stage 4 video-decoder monkey-patch (D-012)
├── notebooks/                  # one driver per stage / analysis
│   ├── 00_explore_egodex.py
│   ├── 01_variance_report.py
│   ├── 02_calibrate_open_questions.py
│   ├── 03_stage2_batch.py            # Stage 1+2 full-task batch
│   ├── 04_stage3_batch.py            # Stage 3 full-task batch
│   ├── 05_stage3_visualize.py        # Stage 3 static plots
│   ├── 06_stage3_animate.py          # Stage 3 animations (stick + side-by-side + mujoco)
│   ├── 07_grasp_clustering.py        # Stage 3 quality assessment (R-008)
│   ├── 08_to_lerobot.py              # Stage 4 dataset conversion
│   ├── 09_train_act.py               # Stage 4 ACT training loop
│   └── 10_eval_act.py                # Stage 4 MuJoCo rollouts
├── third_party/                # vendored upstream assets (gitignored)
│   └── dex-urdf/               # cloned separately, see Setup §5
├── data/                       # EgoDex test split (gitignored)
├── outputs/                    # CSVs, metrics, MP4s, checkpoints (gitignored)
└── logs/                       # session log + per-run stdout captures
```

## Running the pipeline

Each stage has a one-shot single-episode CLI and a full-task batch driver.
Always wrap long-running commands with `tee logs/runs/$(date +%Y-%m-%d_%H%M%S)_<slug>.log`
per the logging convention in `CLAUDE.md`.

### Stage 0 — exploration + calibration (one-time)

```bash
# Per-episode HDF5 dump + per-hand variance/confidence printout
uv run python notebooks/00_explore_egodex.py [path/to/episode.hdf5]

# Aggregate variance report across a task folder (Stage 0 deliverable)
uv run python notebooks/01_variance_report.py [data/test/<task>]

# Confidence-distribution + camera-to-table-distance calibration
uv run python notebooks/02_calibrate_open_questions.py [data/test/<task>]
```

### Stage 1 — EgoStabilizer (plane-induced homography)

```bash
# Single episode → outputs/stage1/<idx>_stabilized.mp4 + _metrics.json
uv run python -m mimicdreamer_egodex.egostabilizer \
    data/test/basic_pick_place/0.hdf5 --out-dir outputs/stage1
```

### Stage 2 — UR5e IK action alignment (mink + smooth posture task)

```bash
# Single episode → outputs/stage2/<idx>_actions.npz + _metrics.json
uv run python -m mimicdreamer_egodex.action_alignment \
    data/test/basic_pick_place/0.hdf5 --out-dir outputs/stage2
```

### Stage 3 — Inspire-hand finger retargeting (dex-retargeting)

```bash
# Single episode → outputs/stage3/<idx>_fingers.npz + _metrics.json
uv run python -m mimicdreamer_egodex.finger_retargeting \
    data/test/basic_pick_place/0.hdf5 --out-dir outputs/stage3
```

### Stage 1+2+3 full-task batches (run after the per-episode scripts work)

```bash
# Stage 1 batch — runs egostabilizer over all 277 episodes (one-time, ~25 min)
# (No dedicated script — loop with `bash for f in data/test/basic_pick_place/*.hdf5; do ... done`,
#  or run `notebooks/03_stage2_batch.py` which also aggregates Stage 1 metrics.)

# Stage 2 batch + Stage 1 metric aggregation (62.7 s wall on RTX 5090)
uv run python notebooks/03_stage2_batch.py

# Stage 3 batch (90.7 s wall)
uv run python notebooks/04_stage3_batch.py

# Optional Stage 3 visualizations (per-episode static + animation MP4s)
uv run python notebooks/05_stage3_visualize.py --episodes 0 1 50 100
uv run python notebooks/06_stage3_animate.py --episodes 0 1 50 100

# Optional Stage 3 quality assessment (per-object grasp clustering, R-008)
uv run python notebooks/07_grasp_clustering.py
```

Outputs land under `outputs/stage{1,2,3}/` plus
`outputs/stage{1,2,3}_summary_basic_pick_place.csv` +
`outputs/stage{1,2,3}_aggregate.json`.

### Stage 4 — LeRobot dataset → ACT training → MuJoCo rollouts

```bash
# B. Convert all 277 episodes into a v3.0 LeRobotDataset (~23 min, 75 MB)
uv run python notebooks/08_to_lerobot.py --force

# C. Train ACT on the full pipeline (~25 min for 3000 steps on RTX 5090)
uv run python notebooks/09_train_act.py \
    --steps 3000 --batch-size 64 --num-workers 8 \
    --eval-every 500 --save-every 1000 --log-every 100 \
    --ckpt-dir outputs/stage4/act_full_pipeline

# D. Roll out the trained policy in the MuJoCo eval env (~7 s for 20 rollouts)
uv run python notebooks/10_eval_act.py \
    --ckpt-dir outputs/stage4/act_full_pipeline \
    --n-rollouts 20 --episode-length 120
```

The eval env uses kinematic control (direct `qpos` writes, see `plan.md`
D-013). The current setup is **proof-of-life only**: 10% success rate on
20 rollouts, dominated by the visual distribution shift between training
and eval (see `plan.md` R-009 + `doc.md` §8.8 for full discussion). The
§4.4 ablation table is not yet built — that requires 3 more dataset
variants and 3 more training runs.

## Session-to-session conventions

- **Always pass `uv run`** so the right Python + venv are used.
- **Always write per-step logs** to `logs/session_YYYY-MM-DD.md` (narrative)
  and `logs/runs/<timestamp>_<slug>.log` (per-run stdout captures). The
  full convention is in `CLAUDE.md`.
- **Don't `cd` without coming back** — bash session cwd persists across
  commands and stray `uv run` from the wrong dir will create a phantom
  `.venv`. Use absolute paths.
- **`uv sync --group <stage>` REPLACES the active group selection**, it
  does NOT append. Always pass every active group (`stage1 stage2 stage3`)
  on each sync. See `plan.md` D-008.
- **Custom ACT training loops**: in any val pass, do NOT call
  `policy.eval()` — ACT's `forward()` crashes computing the VAE KL term
  when the encoder is skipped. Wrap with `torch.no_grad()` only and keep
  the model in `train()`. Inference (`select_action()`) is fine in
  `eval()` mode. See `plan.md` D-014.

For everything else, read `plan.md` (decisions + status) and `doc.md`
(data-derived reference + per-stage results).
