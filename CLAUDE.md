# CLAUDE.md — project conventions for Claude Code

This file is loaded automatically into Claude's context for any work in `/workspace`.
Keep it short and load-bearing; long-form notes go in `plan.md`.

## What this project is
Partial replication of **MimicDreamer** on the **EgoDex** dataset (Apple, 829h egocentric
dexterous manipulation), with a finger-retargeting extension.

**Three load-bearing docs** — read them in this order if you don't already have context:
1. `initial_plan.md` — original 4-stage roadmap (immutable reference).
2. `plan.md` — living plan, decisions, deviations, current status.
3. `doc.md` — data-derived reference for the EgoDex test split: real HDF5 schema,
   coordinate convention, calibrated thresholds. **When `doc.md` and `initial_plan.md`
   disagree, `doc.md` is correct** (the disagreements are documented inline).

**Dataset scope**: only the EgoDex `test` split is used (16.1 GB, 111 tasks, ~277
`basic_pick_place` episodes ≈ 17.9 min @ 30 Hz). No `part1.zip` / `part2.zip` /
additional splits. See `plan.md` D-005.

## Hardware & runtime
- GPU: **NVIDIA RTX 5090** on RunPod (Blackwell, sm_120).
- CUDA: 12.8+. PyTorch **stable** wheels do NOT support sm_120 — must use the
  cu128 nightly index.
- Python: **3.13** (pinned in `pyproject.toml`).

## Tooling — always use uv
- `uv` is preinstalled. **Never** call `pip` or `python -m pip` directly.
- Add a dep:        `uv add <pkg>`           (base) or `uv add --group stage2 <pkg>`
- Sync env:         `uv sync`                (base) or `uv sync --group stage1 --group stage2`
- Run a script:     `uv run python foo.py`
- One-off install:  `uv pip install <pkg>`   (use this for torch nightly — see below)

### Torch / lerobot exception
torch + torchvision are NOT in `pyproject.toml`. The cu128 nightly wheels depend on
a local-version pytorch-triton (`3.7.0+git...`) that uv's resolver can't fetch from the
nightly index. Install them out-of-band after `uv sync`:

```bash
uv pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 \
    torch torchvision
```

`lerobot` is also installed manually because its `torch<2.11` ceiling conflicts with the
nightlies. Recipe lives in `plan.md` once Stage 4 starts.

## Repo layout
```
/workspace
├── CLAUDE.md             # this file
├── README.md             # external setup instructions
├── initial_plan.md       # original 4-stage roadmap (do NOT edit)
├── plan.md               # living working plan, decisions, deviations
├── doc.md                # data-derived reference (schema, conventions, thresholds)
├── pyproject.toml        # uv-managed; deps grouped by stage
├── uv.lock
├── src/mimicdreamer_egodex/   # python package
├── notebooks/            # exploration + calibration scripts (Stage 0)
├── data/test/            # extracted EgoDex test split (gitignored)
├── outputs/              # variance/calibration CSVs, metrics, checkpoints (gitignored)
└── logs/                 # session + run logs (see "Logging" below)
```

## Logging — REQUIRED for every step

The user is often away and reviews logs afterwards. **Log every step to files in
`/workspace/logs/`.** Non-optional. Two log types:

1. **Session log** — `logs/session_YYYY-MM-DD.md` (one per day, append-only).
   Open `## HH:MM — <topic>` blocks; under them, numbered entries: what you tried,
   the command, the result, the decision. Lab-notebook style. Paste full tracebacks,
   never paraphrase. Append as you go, never rewrite history (corrections are
   new entries).

2. **Run log** — `logs/runs/YYYY-MM-DD_HHMMSS_<slug>.log`. Capture stdout+stderr
   for any command that takes more than a few seconds or can fail (downloads,
   training, evals, data scripts). Use `tee`:

   ```bash
   uv run python notebooks/00_explore.py 2>&1 | tee logs/runs/$(date +%Y-%m-%d_%H%M%S)_explore.log
   ```

   Reference the run-log filename from the corresponding session-log entry.

**Log**: side-effecting commands, errors with full text, decisions that deviate
from `initial_plan.md`/`plan.md`/`doc.md` (mirror into the relevant doc),
file paths and dataset IDs touched, hyperparameters.
**Don't log**: trivial reads, secrets, full HDF5 dumps (paths + shapes only).

## Doc maintenance rules
- `initial_plan.md` is the original roadmap — read-only reference.
- `plan.md` is the **working** plan: update it when scope changes, a stage
  starts/finishes, or a decision deviates from `initial_plan.md`. Always note *why*.
- `doc.md` is the data-derived reference (schema, coordinate frame, thresholds).
  Update it whenever the data tells us something `initial_plan.md` got wrong.
  Mirror every `doc.md` change into a matching `plan.md` D-/R- entry so they don't drift.
- `README.md` is for a future collaborator setting this up on a fresh box.
  Keep it install-focused, not narrative.
- This `CLAUDE.md` should stay under ~100 lines. Move long content into `doc.md` or `plan.md`.

## Behavior expectations
- Follow `initial_plan.md` stage ordering. Don't jump ahead.
- Prefer editing existing files over creating new ones; don't sprawl the repo.
- **EgoDex coordinate frame is y-up, not z-up.** `initial_plan.md` §1.1 is wrong on
  this. See `doc.md` §3.
- **HDF5 schema**: every joint is `transforms/<jointName>` `(T,4,4)` with confidence
  at `confidences/<jointName>` `(T,)`. Language metadata is on file-level attrs,
  not a `/language/annotation` dataset. See `doc.md` §2.
- **Confidence thresholds**: `0.10` for hard reject (untracked floor),
  `0.50` only as a *weight*, never a hard cut. See `doc.md` §5.
- Always write a "variance check" output when producing trajectories — the FIVER v1
  failure mode (collapsed joint ranges) is the thing we're explicitly trying to avoid.
