"""
Monkey-patch lerobot's `decode_video_frames` to use **PyAV directly**,
bypassing both `torchcodec` (CUDA-13 ABI mismatch on cu128) and
`torchvision.io.VideoReader` (removed in torchvision 0.27 nightly).

Why this exists
---------------
On torch 2.12.0.dev cu128 + torchvision 0.27.0.dev:
- `torchcodec 0.10` was built against torch 2.10's c10 ABI → undefined symbol
  `c10::MessageLogger::MessageLogger(const char*, int, int, bool)` at load time.
- `torchcodec 0.11` (latest) was built against CUDA 13 → wants `libnvrtc.so.13`
  which we don't have (we have CUDA 12.8 nvrtc bundled with torch).
- lerobot's "pyav" fallback path in `decode_video_frames_torchvision` calls
  `torchvision.io.VideoReader(...)` which **does not exist** in torchvision
  0.27 (it was removed when they deprecated the legacy video API).

So neither the primary (`torchcodec`) nor the fallback (`torchvision.io`) path
works on our nightly torch combo. This patch installs a third path: PyAV
(`av` package) directly. PyAV is mature, ABI-independent of torch, and is
already a transitive dependency of lerobot.

Usage
-----
Import this module **before** instantiating any `LeRobotDataset`:

    from mimicdreamer_egodex.lerobot_pyav_patch import apply
    apply()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id="...", root="...")
    ds[0]   # → goes through our PyAV decoder, no torchcodec/torchvision

Patches both `lerobot.datasets.video_utils.decode_video_frames` and
`lerobot.datasets.dataset_reader.decode_video_frames` (the latter binds the
function name at import time, so the dataset_reader-side reference would
otherwise still point at the broken original).
"""

from __future__ import annotations

import logging

import av
import torch

logger = logging.getLogger(__name__)

_PATCH_APPLIED = False


def decode_video_frames_pyav_only(
    video_path,
    timestamps,
    tolerance_s,
    backend=None,
) -> torch.Tensor:
    """PyAV-only video frame decoder. Drop-in replacement for
    `lerobot.datasets.video_utils.decode_video_frames`.

    Returns a `(N, C, H, W)` float32 tensor in `[0, 1]`, matching the
    contract of the original function.
    """
    container = av.open(str(video_path))
    try:
        if not container.streams.video:
            raise RuntimeError(f"no video stream in {video_path}")
        stream = container.streams.video[0]
        time_base = float(stream.time_base) if stream.time_base else 1.0 / 30.0

        first_ts = float(min(timestamps))
        last_ts = float(max(timestamps))

        # Seek to the keyframe immediately before `first_ts`. PyAV's seek
        # uses stream-time-base units, not seconds.
        try:
            seek_target = max(0, int(first_ts / time_base))
            container.seek(
                seek_target, stream=stream, any_frame=False, backward=True
            )
        except av.error.PyAVError:
            # Some containers don't support seek; fall back to a full scan.
            container.close()
            container = av.open(str(video_path))
            stream = container.streams.video[0]

        loaded_frames: list[torch.Tensor] = []
        loaded_ts: list[float] = []
        for frame in container.decode(stream):
            ts = float(frame.pts * time_base) if frame.pts is not None else 0.0
            arr = frame.to_ndarray(format="rgb24")               # (H, W, 3) uint8
            t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3, H, W)
            loaded_frames.append(t)
            loaded_ts.append(ts)
            if ts >= last_ts:
                break
    finally:
        container.close()

    if not loaded_frames:
        raise RuntimeError(f"PyAV decoded no frames from {video_path}")

    query_ts = torch.tensor(timestamps, dtype=torch.float64)
    loaded_ts_t = torch.tensor(loaded_ts, dtype=torch.float64)

    # |query - loaded| pairwise
    dist = torch.cdist(query_ts[:, None], loaded_ts_t[:, None], p=1)
    min_, argmin_ = dist.min(1)

    if not (min_ < tolerance_s).all():
        bad = min_[min_ >= tolerance_s]
        raise RuntimeError(
            f"PyAV: query timestamps outside tolerance "
            f"({bad} > {tolerance_s=}).\n"
            f"  video: {video_path}\n"
            f"  query_ts: {query_ts.tolist()}\n"
            f"  loaded_ts: {loaded_ts}"
        )

    closest_frames = torch.stack([loaded_frames[i] for i in argmin_])  # (N, 3, H, W)
    return closest_frames.float() / 255.0


def apply() -> None:
    """Install the PyAV decoder into lerobot's video_utils + dataset_reader.

    Idempotent: calling multiple times is a no-op.
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    import lerobot.datasets.video_utils as _vu

    _vu.decode_video_frames = decode_video_frames_pyav_only

    # `dataset_reader` does `from .video_utils import decode_video_frames`
    # at import time, which binds the name into its own namespace. We have
    # to patch that bound reference too.
    try:
        import lerobot.datasets.dataset_reader as _dr

        if hasattr(_dr, "decode_video_frames"):
            _dr.decode_video_frames = decode_video_frames_pyav_only
    except ImportError:
        pass

    _PATCH_APPLIED = True
    logger.info(
        "lerobot_pyav_patch: replaced decode_video_frames with PyAV-only path"
    )
