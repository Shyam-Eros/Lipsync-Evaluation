#!/usr/bin/env python3
"""Export stratified MOS clips and base vs finetuned A/B pairs for human raters."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import config
from pipeline.io_utils import export_clip, read_csv, write_csv

MOS_SCORECARD_FIELDS = [
    "rater_id",
    "clip_id",
    "video_id",
    "model_name",
    "shot_scale",
    "clip_path",
    "sync_mos",
    "naturalness_mos",
    "quality_mos",
    "notes",
]

AB_PAIR_FIELDS = [
    "rater_id",
    "pair_id",
    "video_id",
    "segment_id",
    "model_a",
    "model_b",
    "clip_a",
    "clip_b",
    "winner",
    "preference_dimension",
    "notes",
]

README_RATERS = """# Human evaluation — Tamil lip-sync (MOS + A/B)

You are rating **dubbed Tamil** lip-sync quality. Do not compare to any English source video.

## MOS clips (`mos_scorecard.csv`)

For each row with an empty `rater_id`, watch `clip_path` and fill:

| Column | Scale | Guidance |
|--------|-------|----------|
| `sync_mos` | 1–5 | Lip motion matches dubbed audio |
| `naturalness_mos` | 1–5 | Looks like natural speech |
| `quality_mos` | 1–5 | Visual quality (identity, blending, artifacts) |

Use your `rater_id` (e.g. `r01`) on every row you score.

## A/B pairs (`ab_pairs.csv`)

Watch `clip_a` then `clip_b` (same scene, two models). Set `winner` to `A` or `B`.
`preference_dimension`: `sync`, `quality`, or `overall`.

## Return

Send completed CSVs back to the eval owner. Aggregation: `python aggregate_human_mos.py`.
"""


def _jobs_index_from_metrics(rows: List[dict]) -> Dict[Tuple[str, str], Path]:
    """Best-effort: cannot recover paths from metrics alone."""
    return {}


def _weight_row(r: dict) -> float:
    w = 1.0
    if r.get("shot_scale") == "close-up":
        w *= float(getattr(config, "QC_CLOSE_UP_WEIGHT", 2.0))
    if r.get("qc_pass") == "0":
        w *= 1.5
    return w


def _export_segment_clip(
    jobs_index: Dict[Tuple[str, str], Path],
    r: dict,
    out_dir: Path,
    prefix: str,
) -> str:
    key = (str(r.get("video_id")), str(r.get("model_name")))
    video_path = jobs_index.get(key)
    if not video_path or not Path(video_path).is_file():
        return ""
    out = out_dir / f"{prefix}_{r.get('model_name')}_{r.get('video_id')}_{r.get('segment_id')}.mp4"
    try:
        export_clip(
            Path(video_path),
            out,
            float(r["start_time"]),
            float(r["end_time"]),
        )
        return str(out)
    except Exception:
        return ""


def build_pack(
    metrics_path: Optional[Path] = None,
    qc_path: Optional[Path] = None,
    jobs_index: Optional[Dict[Tuple[str, str], Path]] = None,
    mos_per_model: int = 8,
    seed: int = 42,
) -> Path:
    metrics_path = Path(metrics_path or config.METRICS_RAW_CSV)
    jobs_index = jobs_index or {}
    out_root = Path(config.HUMAN_EVAL_DIR)
    clips_dir = out_root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics not found: {metrics_path}")

    rows = read_csv(metrics_path)
    speaking = [r for r in rows if r.get("segment_type") == "speaking"]
    if qc_path and Path(qc_path).is_file():
        qc_by_id = {r.get("segment_id"): r for r in read_csv(qc_path)}
        for r in speaking:
            q = qc_by_id.get(r.get("segment_id"), {})
            r.setdefault("qc_pass", q.get("qc_pass", ""))

    rng = random.Random(seed)
    by_model: Dict[str, List[dict]] = {}
    for r in speaking:
        by_model.setdefault(str(r.get("model_name")), []).append(r)

    mos_rows: List[dict] = []
    clip_i = 0
    for model, segs in sorted(by_model.items()):
        weights = [_weight_row(s) for s in segs]
        n = min(mos_per_model, len(segs))
        chosen: List[dict] = []
        pool = list(segs)
        pool_w = list(weights)
        for _ in range(n):
            if not pool:
                break
            idx = rng.choices(range(len(pool)), weights=pool_w, k=1)[0]
            chosen.append(pool.pop(idx))
            pool_w.pop(idx)
        for r in chosen:
            clip_i += 1
            clip_id = f"mos_{clip_i:04d}"
            path = _export_segment_clip(jobs_index, r, clips_dir, clip_id)
            mos_rows.append(
                {
                    "rater_id": "",
                    "clip_id": clip_id,
                    "video_id": r.get("video_id", ""),
                    "model_name": model,
                    "shot_scale": r.get("shot_scale", ""),
                    "clip_path": path,
                    "sync_mos": "",
                    "naturalness_mos": "",
                    "quality_mos": "",
                    "notes": "",
                }
            )

    ab_rows: List[dict] = []
    base_name = getattr(config, "HUMAN_AB_BASE_MODEL", "xdub_base")
    ft_name = getattr(config, "HUMAN_AB_FT_MODEL", "xdub_ft")
    base_segs = [r for r in speaking if r.get("model_name") == base_name]
    ft_by_vid: Dict[str, List[dict]] = {}
    for r in speaking:
        if r.get("model_name") == ft_name:
            ft_by_vid.setdefault(str(r.get("video_id")), []).append(r)
    pair_i = 0
    for br in base_segs:
        vid = str(br.get("video_id"))
        candidates = ft_by_vid.get(vid, [])
        if not candidates:
            continue
        b_start = float(br.get("start_time", 0))
        b_end = float(br.get("end_time", b_start))
        best = None
        best_overlap = 0.0
        for fr in candidates:
            f_start = float(fr.get("start_time", 0))
            f_end = float(fr.get("end_time", f_start))
            overlap = max(0.0, min(b_end, f_end) - max(b_start, f_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best = fr
        if best is None or best_overlap <= 0:
            continue
        pair_i += 1
        pair_id = f"ab_{pair_i:04d}"
        clip_a = _export_segment_clip(jobs_index, br, clips_dir, f"{pair_id}_a")
        clip_b = _export_segment_clip(jobs_index, best, clips_dir, f"{pair_id}_b")
        ab_rows.append(
            {
                "rater_id": "",
                "pair_id": pair_id,
                "video_id": vid,
                "segment_id": br.get("segment_id", ""),
                "model_a": base_name,
                "model_b": ft_name,
                "clip_a": clip_a,
                "clip_b": clip_b,
                "winner": "",
                "preference_dimension": "overall",
                "notes": "",
            }
        )

    write_csv(out_root / "mos_scorecard.csv", mos_rows, MOS_SCORECARD_FIELDS)
    write_csv(out_root / "ab_pairs.csv", ab_rows, AB_PAIR_FIELDS)
    (out_root / "README_raters.md").write_text(README_RATERS, encoding="utf-8")
    print(
        f"Human eval pack: {len(mos_rows)} MOS, {len(ab_rows)} A/B → {out_root}",
        flush=True,
    )
    return out_root


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build human MOS/A/B evaluation pack")
    p.add_argument("--metrics", type=Path, default=config.METRICS_RAW_CSV)
    p.add_argument("--qc", type=Path, default=config.QC_FLAGS_CSV)
    p.add_argument("--mos-per-model", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    try:
        build_pack(
            metrics_path=args.metrics,
            qc_path=args.qc,
            mos_per_model=args.mos_per_model,
            seed=args.seed,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
