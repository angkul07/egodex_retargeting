# mimicdreamer-egodex

Partial replication of [MimicDreamer](https://arxiv.org/abs/2509.22199) on the
[EgoDex](https://arxiv.org/abs/2505.11709) dataset, with a dexterous-finger
retargeting extension. Built for an internship-prep timeline, not for production.

Documentation map:
- `initial_plan.md` — original 4-stage roadmap (read-only).
- `plan.md` — living plan, decisions, current status.
- `doc.md` — data-derived reference: real EgoDex HDF5 schema, coordinate frame
  (y-up, not z-up), calibrated confidence + table-distance thresholds. **When
  `doc.md` and `initial_plan.md` disagree, `doc.md` is correct.**
- `CLAUDE.md` — Claude Code conventions for working in this repo.

Scope: only the EgoDex **test split** (16.1 GB, 111 tasks, ~277 `basic_pick_place`
episodes). No `part1.zip` / `part2.zip` downloads. See `plan.md` D-005.

## Hardware target

NVIDIA RTX 5090 (Blackwell, sm_120) on RunPod. CUDA 12.8+. PyTorch stable wheels
do **not** support sm_120 — the cu128 nightly build is required.

## Setup on a fresh machine

Prereqs: `uv` (https://docs.astral.sh/uv/) and CUDA 12.8 driver.

```bash
git clone <repo-url> mimicdreamer-egodex
cd mimicdreamer-egodex

# 1. Install Python 3.13 + base deps (Stage 0 only).
uv sync

# 2. Install stage-specific deps as you reach each stage:
uv sync --group stage1                       # EgoStabilizer
uv sync --group stage1 --group stage2        # + IK / action alignment
uv sync --group stage1 --group stage2 --group stage3   # + finger retargeting

# 3. Stage 4 — torch nightly cu128 (RTX 5090 needs sm_120 support).
#    These are NOT in pyproject.toml; see plan.md decision D-001.
uv sync --group stage4
uv pip install --pre \
    --index-url https://download.pytorch.org/whl/nightly/cu128 \
    torch torchvision

# 4. Verify CUDA + sm_120 are visible to torch:
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`lerobot` is also installed manually at Stage 4 — see `plan.md` for the recipe
once that stage starts.

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

We do **not** download `part1.zip` or `part2.zip`. Stage 4 policy training runs
on the test split's `basic_pick_place` task only (~277 episodes ≈ 18 minutes
@ 30 Hz). Rationale and implications: `plan.md` D-005, `doc.md` §0.

## Repo layout

```
.
├── CLAUDE.md             # Claude Code conventions
├── README.md             # this file
├── initial_plan.md       # 4-stage roadmap (read-only reference)
├── plan.md               # living plan, decisions, status
├── doc.md                # data-derived reference (schema, conventions, thresholds)
├── pyproject.toml        # uv-managed deps, stage-grouped
├── uv.lock
├── src/mimicdreamer_egodex/   # the package
├── notebooks/            # Stage 0 exploration + calibration
├── data/                 # EgoDex test split (gitignored)
├── outputs/              # CSVs, metrics, checkpoints (gitignored)
└── logs/                 # session log + per-run stdout captures (gitignored)
```

## Stage 0 scripts (already working)

```bash
# Single-episode HDF5 dump and per-hand variance/confidence printout
uv run python notebooks/00_explore_egodex.py [path/to/episode.hdf5]

# Aggregate variance report across a task folder (Stage 0 deliverable)
uv run python notebooks/01_variance_report.py [data/test/<task>]

# Confidence-distribution + camera-to-table-distance calibration
uv run python notebooks/02_calibrate_open_questions.py [data/test/<task>]
```

CSV output lands in `outputs/`. Always run with `uv run` so the right Python
and environment are used. Wrap long-running commands with
`tee logs/runs/$(date +%Y-%m-%d_%H%M%S)_<slug>.log` per the logging convention
in `CLAUDE.md`.
