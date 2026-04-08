"""
07_grasp_clustering.py — does the Stage 3 retargeting preserve object identity?

For each `basic_pick_place` episode:
  - load `q_finger` (T,6) + `tips_rel` (T,5,3) from outputs/stage3/<idx>_fingers.npz
  - find the "peak grasp" frame as `argmin(mean fingertip-to-wrist distance)`
  - take `q_finger[peak]` as a 6-D **grasp signature** for the episode
  - read `llm_objects[0]` from the HDF5 attrs as the "primary object"

Then:
  - group episodes by primary object
  - for objects with ≥ 3 episodes, compute:
      - per-object mean signature + within-object std
      - between-object spread (std of group means)
      - per-DOF separation ratio = between_std / within_std
      - silhouette-like score per episode (own-cluster vs nearest-other distance)
  - PCA-2D scatter of all signatures, colored by object

A separation ratio > 1 + a positive mean silhouette indicates the retargeting
captures object identity, not just produces undifferentiated noise.

Outputs
-------
- `outputs/stage3_grasp_clustering.json` — all numbers
- `outputs/stage3/viz/grasp_clustering_pca.png` — 2D scatter
- `outputs/stage3/viz/grasp_clustering_per_object.png` — per-object signature bar chart
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
TASK_DIR = REPO / "data/test/basic_pick_place"
STAGE3_DIR = REPO / "outputs/stage3"
OUT_DIR = REPO / "outputs"
VIZ_DIR = REPO / "outputs/stage3/viz"

# Display order for the 6 Inspire target DOFs (matches q_finger column order
# in finger_retargeting.INSPIRE_TARGET_FINGER_JOINTS).
DOF_NAMES = ("index", "middle", "ring", "pinky", "th_yaw", "th_pitch")


def normalize_object_name(raw) -> str:
    """`llm_objects` is a numpy object-dtype array. Take the first entry,
    decode bytes if needed, and lowercase + strip."""
    if raw is None:
        return "unknown"
    try:
        v = raw[0]
    except (TypeError, IndexError, KeyError):
        v = raw
    if isinstance(v, bytes):
        v = v.decode()
    return str(v).strip().lower()


def load_episode_signatures(idx: int):
    """Returns both a 6-D peak-grasp signature and an 18-D trajectory
    signature (per-joint min, max, mean) for the same episode, plus the
    object label. Both are computed because the peak-frame is single-instant
    noisy and the trajectory aggregate is smoother but less interpretable."""
    npz = STAGE3_DIR / f"{idx}_fingers.npz"
    h5 = TASK_DIR / f"{idx}.hdf5"
    if not npz.exists() or not h5.exists():
        return None
    d = np.load(npz, allow_pickle=True)
    q_finger = d["q_finger"]              # (T, 6)
    tips_rel = d["tips_rel"]              # (T, 5, 3)
    active = str(d["active_hand"])

    # 6-D peak grasp (frame where mean fingertip-to-wrist distance is min)
    openness = np.linalg.norm(tips_rel, axis=-1).mean(axis=1)  # (T,)
    t_peak = int(openness.argmin())
    sig_peak = q_finger[t_peak].astype(np.float32)

    # 18-D trajectory aggregate (per-joint min/max/mean)
    sig_traj = np.concatenate(
        [q_finger.min(axis=0), q_finger.max(axis=0), q_finger.mean(axis=0)]
    ).astype(np.float32)

    with h5py.File(h5, "r") as f:
        obj_raw = f.attrs.get("llm_objects", None)
    obj = normalize_object_name(obj_raw)
    return {
        "idx": idx,
        "object": obj,
        "hand": active,
        "sig_peak": sig_peak,
        "sig_traj": sig_traj,
    }


def silhouette_per_episode(
    rows: list[dict],
    means: dict[str, np.ndarray],
    sig_key: str,
) -> list[float]:
    """For each episode, compute a silhouette-like score on the chosen
    signature variant (`sig_peak` or `sig_traj`):

        s = (nearest_other_d - own_d) / max(own_d, nearest_other_d)

    Range [-1, +1]: +1 = perfectly clustered, 0 = ambiguous, -1 = misclustered.
    Episodes whose object isn't in `means` (singletons) are skipped.
    """
    out = []
    obj_keys = list(means.keys())
    for r in rows:
        if r["object"] not in means or len(obj_keys) < 2:
            continue
        sig = r[sig_key]
        own = means[r["object"]]
        own_d = float(np.linalg.norm(sig - own))
        other_d = min(
            float(np.linalg.norm(sig - means[k]))
            for k in obj_keys
            if k != r["object"]
        )
        denom = max(own_d, other_d)
        if denom < 1e-12:
            continue
        out.append((other_d - own_d) / denom)
    return out


def cluster_analysis(
    rows: list[dict], sig_key: str, min_n: int
) -> dict:
    """Run a full per-object clustering analysis on the given signature.
    Returns a dict containing per-DOF and overall spread, silhouette,
    pair distances, and PCA-2D coords."""
    by_obj: dict[str, list[np.ndarray]] = defaultdict(list)
    for r in rows:
        by_obj[r["object"]].append(r[sig_key])
    common = {o: np.stack(s) for o, s in by_obj.items() if len(s) >= min_n}
    if not common:
        return {}
    means = {o: s.mean(axis=0) for o, s in common.items()}
    within_stds = {o: s.std(axis=0) for o, s in common.items()}

    means_arr = np.stack(list(means.values()))
    within_arr = np.stack(list(within_stds.values()))
    between_std_per_dof = means_arr.std(axis=0)
    within_std_per_dof = within_arr.mean(axis=0)
    overall_within = float(within_std_per_dof.mean())
    overall_between = float(between_std_per_dof.mean())
    overall_ratio = overall_between / max(1e-9, overall_within)

    silh = silhouette_per_episode(rows, means, sig_key)
    silh_mean = float(np.mean(silh)) if silh else 0.0
    silh_pos_frac = float(np.mean(np.array(silh) > 0)) if silh else 0.0

    objs = list(common.keys())
    pairs = []
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            d = float(np.linalg.norm(means[objs[i]] - means[objs[j]]))
            pairs.append((objs[i], objs[j], d))
    pairs.sort(key=lambda x: -x[2])

    # PCA 2D
    all_sigs = np.stack([s for sigs in common.values() for s in sigs])
    all_objs = [obj for obj, sigs in common.items() for _ in sigs]
    Xc = all_sigs - all_sigs.mean(axis=0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ Vt[:2].T
    explained = (s[:2] ** 2) / (s ** 2).sum()

    return {
        "common": common,
        "means": means,
        "within_stds": within_stds,
        "n_episodes": int(sum(s.shape[0] for s in common.values())),
        "n_objects": len(common),
        "between_std_per_dof": between_std_per_dof,
        "within_std_per_dof": within_std_per_dof,
        "overall_within": overall_within,
        "overall_between": overall_between,
        "overall_ratio": overall_ratio,
        "silhouette_mean": silh_mean,
        "silhouette_pos_frac": silh_pos_frac,
        "n_silhouette": len(silh),
        "pairs_sorted": pairs,
        "all_sigs": all_sigs,
        "all_objs": all_objs,
        "pca_coords": coords,
        "pca_explained": explained,
    }


def _plot_pca(res: dict, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    cmap = plt.cm.tab20
    sorted_objs = sorted(res["common"].keys(), key=lambda o: -res["common"][o].shape[0])
    coords = res["pca_coords"]
    all_objs = res["all_objs"]
    for i, obj in enumerate(sorted_objs):
        mask = np.array([o == obj for o in all_objs])
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            color=cmap(i % 20),
            label=f"{obj} (n={int(mask.sum())})",
            s=55,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.5,
        )
    ax.set_xlabel(f"PC1 ({res['pca_explained'][0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({res['pca_explained'][1] * 100:.1f}% var)")
    ax.set_title(
        f"{title}\n"
        f"separation ratio = {res['overall_ratio']:.2f}, "
        f"mean silhouette = {res['silhouette_mean']:+.3f} "
        f"({res['silhouette_pos_frac'] * 100:.0f}% positive)"
    )
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_per_object_bars(res: dict, title: str, out_path: Path) -> None:
    sorted_objs = sorted(res["common"].keys(), key=lambda o: -res["common"][o].shape[0])
    n_obj = len(sorted_objs)
    width = 0.13
    x = np.arange(n_obj)
    fig, ax = plt.subplots(figsize=(max(12, n_obj * 0.55), 6))
    for i, name in enumerate(DOF_NAMES):
        vals = [res["means"][obj][i] for obj in sorted_objs]
        errs = [res["within_stds"][obj][i] for obj in sorted_objs]
        ax.bar(x + i * width, vals, width, yerr=errs, label=name, alpha=0.85, capsize=2)
    ax.set_xticks(x + 2.5 * width)
    ax.set_xticklabels(sorted_objs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("peak-grasp angle (rad)")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=6)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _print_analysis(res: dict, label: str) -> None:
    print(f"\n--- {label} ---")
    print(f"  n_objects (>= min): {res['n_objects']}, n_episodes: {res['n_episodes']}")
    print(f"  overall within std (rad): {res['overall_within']:.4f}")
    print(f"  overall between std (rad): {res['overall_between']:.4f}")
    print(f"  separation ratio:         {res['overall_ratio']:.3f}")
    print(f"  silhouette mean:          {res['silhouette_mean']:+.3f}  "
          f"({res['silhouette_pos_frac'] * 100:.1f}% positive over {res['n_silhouette']} eps)")
    print(f"  top 5 most DISTINCT pairs (rad):")
    for a, b, d in res["pairs_sorted"][:5]:
        print(f"    {d:6.3f}  {a:25s} vs {b}")
    print(f"  top 5 most SIMILAR pairs (rad):")
    for a, b, d in res["pairs_sorted"][-5:]:
        print(f"    {d:6.3f}  {a:25s} vs {b}")


def _result_to_json(res: dict) -> dict:
    return {
        "n_objects": res["n_objects"],
        "n_episodes": res["n_episodes"],
        "overall_within_std_rad": res["overall_within"],
        "overall_between_std_rad": res["overall_between"],
        "overall_separation_ratio": res["overall_ratio"],
        "silhouette_mean": res["silhouette_mean"],
        "silhouette_pos_fraction": res["silhouette_pos_frac"],
        "n_silhouette_samples": res["n_silhouette"],
        "pca_explained_variance_2d": [
            float(res["pca_explained"][0]),
            float(res["pca_explained"][1]),
        ],
        "most_distinct_pairs": [[a, b, d] for a, b, d in res["pairs_sorted"][:10]],
        "most_similar_pairs": [[a, b, d] for a, b, d in res["pairs_sorted"][-10:]],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 3 grasp clustering")
    ap.add_argument("--min-episodes-per-object", type=int, default=3)
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    paths = sorted(TASK_DIR.glob("*.hdf5"), key=lambda p: int(p.stem))
    print(f"Found {len(paths)} episodes")

    rows = []
    for h5 in paths:
        idx = int(h5.stem)
        result = load_episode_signatures(idx)
        if result is not None:
            rows.append(result)
    print(f"Loaded {len(rows)} episode signatures")

    objects_n = sorted(
        ((obj, sum(1 for r in rows if r["object"] == obj)) for obj in {r["object"] for r in rows}),
        key=lambda x: -x[1],
    )
    print(f"\n{len(objects_n)} unique objects total. Top 25 by count:")
    for obj, n in objects_n[:25]:
        print(f"  {n:3d}  {obj}")

    # Two analyses on the same data
    res_peak = cluster_analysis(rows, "sig_peak", args.min_episodes_per_object)
    res_traj = cluster_analysis(rows, "sig_traj", args.min_episodes_per_object)

    _print_analysis(res_peak, "6-D PEAK-GRASP signature (single most-closed frame)")
    _print_analysis(res_traj, "18-D TRAJECTORY signature (per-joint min/max/mean over episode)")

    # Plots — both signatures, side-by-side files
    _plot_pca(
        res_peak,
        "Stage 3 peak-grasp signature PCA (6-D, single frame)",
        VIZ_DIR / "grasp_clustering_pca.png",
    )
    _plot_pca(
        res_traj,
        "Stage 3 trajectory signature PCA (18-D, min/max/mean per joint)",
        VIZ_DIR / "grasp_clustering_pca_traj.png",
    )
    _plot_per_object_bars(
        res_peak,
        f"Per-object PEAK-grasp signature (mean ± within-object std)  — "
        f"{res_peak['n_objects']} objects, ≥ {args.min_episodes_per_object} eps each",
        VIZ_DIR / "grasp_clustering_per_object.png",
    )
    print(f"\n  wrote 3 PCA/bar plots to {VIZ_DIR}/grasp_clustering_*.png")

    # JSON summary — both analyses
    summary = {
        "n_episodes_total": len(rows),
        "n_objects_total": len(objects_n),
        "min_episodes_per_object": args.min_episodes_per_object,
        "object_counts_top25": [[obj, n] for obj, n in objects_n[:25]],
        "peak_grasp_6d": _result_to_json(res_peak),
        "trajectory_18d": _result_to_json(res_traj),
    }
    out_json = OUT_DIR / "stage3_grasp_clustering.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"  wrote {out_json}")

    # Verdict
    print(f"\n=== INTERPRETATION ===")
    ratio = res_traj["overall_ratio"]
    silh = res_traj["silhouette_mean"]
    print(
        f"  Using the more reliable 18-D trajectory signature: "
        f"separation ratio = {ratio:.2f}, silhouette = {silh:+.3f}."
    )
    if ratio > 1.0 and silh > 0:
        verdict = (
            "Retargeting CLEANLY preserves object identity. Different\n"
            "  objects produce clearly different grasp trajectories."
        )
    elif ratio > 0.7:
        verdict = (
            "Retargeting captures AFFORDANCE-CLASS structure but does not\n"
            "  cleanly separate within-class objects. Look at the most-distinct\n"
            "  vs most-similar pairs above: large/blocky objects vs small/round\n"
            "  ones are far apart, but objects within the same physical\n"
            "  affordance class collapse together. This is consistent with the\n"
            "  6-DOF Inspire hand only being able to express a limited grasp\n"
            "  vocabulary on power-grasp data."
        )
    else:
        verdict = (
            "Retargeting does NOT clearly preserve task identity — within-\n"
            "  object noise dominates between-object spread."
        )
    print(f"  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
