"""Roll up metrics_raw → per-model, shot-scale, difficulty, and tier summary."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from pipeline import config
from pipeline.io_utils import write_csv

logger = logging.getLogger(__name__)

METRIC_KEYS = [
    "lse_d",
    "lse_c",
    "lse_d_z",
    "lse_c_z",
    "lse_offset_corrected",
    "fid",
    "fvd",
    "csim",
    "csim_late",
    "jitter_score",
    "warping_error",
    "seam_score",
    "oral_artifact_score",
    "bilabial_score",
    "bilabial_closure_fail_rate",
    "upper_face_csim",
    "expr_drift",
    "av_timing_ratio",
]

# Metrics that only make sense on speaking segments
SPEAKING_ONLY = {
    "lse_d",
    "lse_c",
    "lse_d_z",
    "lse_c_z",
    "lse_offset_corrected",
    "fid",
    "fvd",
    "seam_score",
    "oral_artifact_score",
    "bilabial_score",
    "bilabial_closure_fail_rate",
    "upper_face_csim",
    "expr_drift",
    "av_timing_ratio",
}


def _f(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _mean(vals: List[float]) -> Optional[float]:
    return sum(vals) / len(vals) if vals else None


def _fmt(v: Optional[float]) -> str:
    return f"{v:.6f}" if v is not None else ""


def aggregate(metrics_rows: Sequence[dict]) -> Dict[str, List[dict]]:
    """
    Returns dict with keys:
      by_model — one row per model
      by_scale — one row per (model, shot_scale)
      by_difficulty — one row per (model, difficulty_tag)
    """
    by_model: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    by_scale: Dict[tuple, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    by_diff: Dict[tuple, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    counts_model: Dict[str, int] = defaultdict(int)
    counts_scale: Dict[tuple, int] = defaultdict(int)
    counts_diff: Dict[tuple, int] = defaultdict(int)

    for r in metrics_rows:
        model = r.get("model_name") or r.get("model") or "unknown"
        scale = r.get("shot_scale") or "unknown"
        is_speaking = r.get("segment_type", "speaking") == "speaking"
        tags_raw = str(r.get("difficulty_tags") or "").strip()
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()] or ["untagged"]

        for k in METRIC_KEYS:
            val = _f(r.get(k))
            if val is None:
                continue
            if k in SPEAKING_ONLY and not is_speaking:
                continue
            by_model[model][k].append(val)
            by_scale[(model, scale)][k].append(val)
            for tag in tags:
                by_diff[(model, tag)][k].append(val)
        if is_speaking:
            counts_model[model] += 1
            counts_scale[(model, scale)] += 1
            for tag in tags:
                counts_diff[(model, tag)] += 1

    model_rows = []
    for model in sorted(by_model.keys()):
        row = {"model": model, "n_speaking_segments": counts_model[model]}
        for k in METRIC_KEYS:
            row[f"mean_{k}"] = _fmt(_mean(by_model[model][k]))
        model_rows.append(row)

    scale_rows = []
    for (model, scale) in sorted(by_scale.keys()):
        row = {
            "model": model,
            "shot_scale": scale,
            "n_speaking_segments": counts_scale[(model, scale)],
        }
        for k in METRIC_KEYS:
            row[f"mean_{k}"] = _fmt(_mean(by_scale[(model, scale)][k]))
        scale_rows.append(row)

    diff_rows = []
    for (model, tag) in sorted(by_diff.keys()):
        row = {
            "model": model,
            "difficulty_tag": tag,
            "n_speaking_segments": counts_diff[(model, tag)],
        }
        for k in METRIC_KEYS:
            row[f"mean_{k}"] = _fmt(_mean(by_diff[(model, tag)][k]))
        diff_rows.append(row)

    return {"by_model": model_rows, "by_scale": scale_rows, "by_difficulty": diff_rows}


def write_scale_csv(scale_rows: List[dict]) -> None:
    fields = [
        "model",
        "shot_scale",
        "n_speaking_segments",
        *[f"mean_{k}" for k in METRIC_KEYS],
    ]
    write_csv(config.METRICS_BY_SCALE_CSV, scale_rows, fields)


def write_difficulty_csv(diff_rows: List[dict]) -> None:
    fields = [
        "model",
        "difficulty_tag",
        "n_speaking_segments",
        *[f"mean_{k}" for k in METRIC_KEYS],
    ]
    write_csv(config.METRICS_BY_DIFFICULTY_CSV, diff_rows, fields)


def write_tier_summary_json(metrics_rows: Sequence[dict], agg: Dict[str, List[dict]]) -> None:
    speaking = [r for r in metrics_rows if r.get("segment_type") == "speaking"]
    n_speak = len(speaking)
    qc_pass = sum(1 for r in speaking if r.get("qc_pass") == "1")
    proxy = sum(1 for r in speaking if r.get("fvd_mode") == "proxy")
    tier2_keys = [
        "seam_score",
        "oral_artifact_score",
        "bilabial_score",
        "upper_face_csim",
        "expr_drift",
        "av_timing_ratio",
    ]
    tier2_means: Dict[str, Dict[str, float]] = {}
    for row in agg.get("by_model", []):
        model = row["model"]
        tier2_means[model] = {}
        for k in tier2_keys:
            v = row.get(f"mean_{k}")
            if v:
                try:
                    tier2_means[model][k] = float(v)
                except ValueError:
                    pass
    payload = {
        "n_metric_rows": len(metrics_rows),
        "n_speaking_segments": n_speak,
        "qc_pass_rate": (qc_pass / n_speak) if n_speak else None,
        "n_fvd_proxy_segments": proxy,
        "tier2_mean_by_model": tier2_means,
        "by_model": agg.get("by_model", []),
    }
    config.METRICS_TIER_SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with config.METRICS_TIER_SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote tier summary → %s", config.METRICS_TIER_SUMMARY_JSON)


def print_summary(agg: Dict[str, List[dict]], metrics_rows: Sequence[dict] | None = None) -> None:
    speaking = list(metrics_rows or []) if metrics_rows is not None else []
    if metrics_rows is not None:
        speaking = [r for r in metrics_rows if r.get("segment_type") == "speaking"]
    n_speak = len(speaking)
    qc_pass = sum(1 for r in speaking if r.get("qc_pass") == "1")
    proxy = sum(1 for r in speaking if r.get("fvd_mode") == "proxy")

    print("\n========== SUMMARY (speaking segments) ==========", flush=True)
    if n_speak:
        print(
            f"QC pass: {qc_pass}/{n_speak} ({100.0 * qc_pass / n_speak:.1f}%)",
            flush=True,
        )
    if proxy:
        print(
            f"WARNING: {proxy} speaking segment(s) used FVD proxy (not I3D) — see fvd_mode column",
            flush=True,
        )
    header = (
        f"{'model':<12} {'LSE-D↓':>9} {'LSE-C↑':>9} {'FID↓':>9} "
        f"{'FVD↓':>9} {'CSIM↑':>8} {'late↑':>8} {'jit↓':>8} {'warp↓':>8} {'n':>4}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for r in agg["by_model"]:
        print(
            f"{r['model']:<12} "
            f"{r.get('mean_lse_d') or 'n/a':>9} "
            f"{r.get('mean_lse_c') or 'n/a':>9} "
            f"{r.get('mean_fid') or 'n/a':>9} "
            f"{r.get('mean_fvd') or 'n/a':>9} "
            f"{r.get('mean_csim') or 'n/a':>8} "
            f"{r.get('mean_csim_late') or 'n/a':>8} "
            f"{r.get('mean_jitter_score') or 'n/a':>8} "
            f"{r.get('mean_warping_error') or 'n/a':>8} "
            f"{r.get('n_speaking_segments', 0):>4}",
            flush=True,
        )
    print("\n--- Tier 2 (means) ---", flush=True)
    for r in agg["by_model"]:
        print(
            f"  {r['model']:<10} seam={r.get('mean_seam_score') or 'n/a'} "
            f"oral={r.get('mean_oral_artifact_score') or 'n/a'} "
            f"bilabial={r.get('mean_bilabial_score') or 'n/a'} "
            f"upper_csim={r.get('mean_upper_face_csim') or 'n/a'}",
            flush=True,
        )
    print("\n--- By shot scale (do not blend close-up with wide) ---", flush=True)
    for r in agg["by_scale"]:
        print(
            f"  {r['model']:<10} {r['shot_scale']:<10} "
            f"LSE-D={r.get('mean_lse_d') or 'n/a'} "
            f"LSE-C={r.get('mean_lse_c') or 'n/a'} "
            f"FID={r.get('mean_fid') or 'n/a'} "
            f"n={r.get('n_speaking_segments', 0)}",
            flush=True,
        )
    print("=================================================\n", flush=True)
