"""
Reference-free dubbed-movie lip-sync evaluation.

CRITICAL: Never compares generated frames to English-synced source / GT video.
Layer 1 builds segments.csv; Layer 2 scores only from that table.

Setup
-----
1. cd Evalution && source .venv/bin/activate
2. bash syncnet/download_weights.sh
3. python calibrate_syncnet_tamil.py --data-root /path/to/tamil/clips
4. python run_pipeline.py --video out.mp4 --model xdub
   python run_pipeline.py --layer1-only   # segmentation debug
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import config
from pipeline.aggregation import (
    aggregate,
    print_summary,
    write_difficulty_csv,
    write_scale_csv,
    write_tier_summary_json,
)
from pipeline.io_utils import (
    cache_key,
    ensure_dirs,
    export_clip,
    extract_audio,
    extract_frames,
    read_csv,
    write_csv,
)
from pipeline.metrics import (
    identity_eval,
    oral_artifact_eval,
    pose_expression_eval,
    quality_eval,
    seam_eval,
    speech_timing_eval,
    stability_eval,
    sync_eval,
    viseme_eval,
)
from pipeline.metrics.qc_gate import apply_qc_flags, rollup_qc_shots
from pipeline.segmentation.active_speaker import SEGMENT_FIELDS, run_active_speaker
from pipeline.segmentation.face_tracking import run_face_tracking
from pipeline.segmentation.identity_clustering import run_identity_clustering
from pipeline.segmentation.shot_detection import run_shot_detection


def _setup_logging() -> None:
    ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.LOG_DIR / "run.log", mode="a", encoding="utf-8"),
        ],
        force=True,
    )


def _list_videos(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in config.VIDEO_EXTS
    )


def _infer_video_id_from_path(path: Path) -> str | None:
    """UI-UX Inference layout: .../data/jobs/<job_id>/infer_{base,ft}/run_final.mp4."""
    parts = [p.lower() for p in path.resolve().parts]
    if "jobs" not in parts:
        return None
    i = parts.index("jobs")
    if i + 1 >= len(parts):
        return None
    job_id = path.resolve().parts[i + 1]
    if job_id in ("infer_base", "infer_ft"):
        return None
    return str(job_id)


def _iter_jobs(
    cli_videos: List[Path] | None = None,
    cli_models: List[str] | None = None,
    video_id: str | None = None,
) -> List[Tuple[str, str, Path]]:
    """Build (video_id, model_name, path) jobs.

    CLI --video/--model pairs take precedence over EVAL_ITEMS / INPUT_DIRS.
    """
    jobs: List[Tuple[str, str, Path]] = []
    if cli_videos:
        models = cli_models or []
        if len(models) not in (0, 1, len(cli_videos)):
            raise ValueError(
                "Provide one --model per --video, a single --model for all, or omit --model"
            )
        for i, path in enumerate(cli_videos):
            path = Path(path)
            if len(models) == 0:
                model = "model"
            elif len(models) == 1:
                model = models[0]
            else:
                model = models[i]
            vid = video_id or _infer_video_id_from_path(path) or path.stem
            jobs.append((vid, model, path))
        return jobs

    items = getattr(config, "EVAL_ITEMS", None) or []
    if items:
        for i, item in enumerate(items):
            vid = str(item.get("video_id") or f"clip_{i}")
            # Prefer known MODEL_KEYS, but also accept any other path keys
            keys = list(config.MODEL_KEYS)
            for k in item:
                if k not in ("video_id",) and k not in keys and item.get(k):
                    keys.append(k)
            for model in keys:
                raw = item.get(model)
                if raw:
                    jobs.append((vid, model, Path(raw)))
        return jobs
    for model, d in config.INPUT_DIRS.items():
        for path in _list_videos(Path(d)):
            jobs.append((path.stem, model, path))
    return jobs


def run_layer1(video_id: str, model_name: str, video_path: Path) -> List[dict]:
    """Full segmentation → append/write segments for this video."""
    key = cache_key(video_path)
    work = config.TEMP_DIR / f"{model_name}_{video_id}_{key}"
    work.mkdir(parents=True, exist_ok=True)

    frames_dir = work / "frames"
    wav_path = work / "audio_full.wav"
    print(f"{model_name}/{video_id} — extracting frames + audio …", flush=True)
    extract_frames(video_path, frames_dir)
    extract_audio(video_path, wav_path)

    shots = run_shot_detection(
        video_path, frames_dir, video_id=video_id, model_name=model_name
    )
    tracklets, shot_scales = run_face_tracking(
        frames_dir, shots, work / "tracks", video_id=video_id, model_name=model_name
    )
    # Update shot_scale on shot rows in memory
    for s in shots:
        s["shot_scale"] = shot_scales.get(int(s["shot_id"]), "pending")

    identity_map = (
        run_identity_clustering(tracklets, video_id, model_name) if tracklets else {}
    )
    segments = run_active_speaker(
        video_id=video_id,
        model_name=model_name,
        shots=shots,
        tracklets=tracklets,
        identity_map=identity_map,
        shot_scales=shot_scales,
        full_wav=wav_path,
        work_dir=work / "segments",
    )
    return segments


def run_layer2_for_video(
    video_id: str,
    model_name: str,
    video_path: Path,
    segments: List[dict],
) -> List[dict]:
    """Run independent metric branches; failures logged, other branches continue."""
    log_path = config.LOG_DIR / f"{model_name}_{video_id}_metrics.log"
    rows = []

    # FID references: full-face + mouth/lower-face
    ref_dir = quality_eval.build_reference_pool(
        segments,
        config.TEMP_DIR / f"fid_ref_{model_name}_{video_id}",
        real_face_dir=config.REAL_FACE_REF_DIR,
    )
    mouth_ref_dir = quality_eval.build_mouth_reference_pool(
        segments,
        config.TEMP_DIR / f"fid_mouth_ref_{model_name}_{video_id}",
    )
    identity_refs = identity_eval.build_identity_references(segments)
    upper_refs = pose_expression_eval.build_upper_face_refs(segments)

    # Video-level I3D FVD (broadcast to speaking segments)
    video_fvd = ""
    fvd_notes = ""
    video_fvd_mode = ""
    try:
        fvd_res = quality_eval.compute_video_fvd(segments)
        video_fvd = fvd_res.get("fvd", "")
        fvd_notes = fvd_res.get("fvd_notes", "")
        video_fvd_mode = fvd_res.get("fvd_mode", "")
    except Exception as exc:
        fvd_notes = f"fvd_video_failed:{exc}"
        logging.warning("Video FVD failed %s/%s: %s", model_name, video_id, exc)

    for seg in segments:
        row = {
            "segment_id": seg.get("segment_id"),
            "segment_type": seg.get("segment_type"),
            "video_id": video_id,
            "model_name": model_name,
            "shot_id": seg.get("shot_id"),
            "tracklet_id": seg.get("tracklet_id"),
            "identity_id": seg.get("identity_id"),
            "shot_scale": seg.get("shot_scale"),
            "difficulty_tags": seg.get("difficulty_tags", ""),
            "start_time": seg.get("start_time"),
            "end_time": seg.get("end_time"),
            "lse_d": "",
            "lse_c": "",
            "lse_offset": "",
            "lse_d_z": "",
            "lse_c_z": "",
            "lse_offset_corrected": "",
            "fid": "",
            "fid_face": "",
            "fvd": "",
            "fvd_mode": video_fvd_mode if seg.get("segment_type") == "speaking" else "",
            "csim": "",
            "csim_late": "",
            "jitter_score": "",
            "warping_error": "",
            "seam_score": "",
            "oral_artifact_score": "",
            "bilabial_frames": "",
            "bilabial_closure_fail_rate": "",
            "bilabial_score": "",
            "upper_face_csim": "",
            "expr_drift": "",
            "av_timing_ratio": "",
            "timing_stress_flag": "",
            "qc_pass": "",
            "qc_fail_reasons": "",
            "notes": "",
        }
        notes = []

        # Sync
        try:
            sres = sync_eval.evaluate_segment(seg)
            for k in (
                "lse_d", "lse_c", "lse_offset",
                "lse_d_z", "lse_c_z", "lse_offset_corrected",
            ):
                if k in sres:
                    row[k] = sres[k]
            if sres.get("sync_notes"):
                notes.append(sres["sync_notes"])
        except Exception as exc:
            notes.append(f"sync_failed:{exc}")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"SYNC {seg.get('segment_id')}: {exc}\n{traceback.format_exc()}\n")

        # Quality (mouth FID + I3D FVD)
        try:
            q = quality_eval.evaluate_segment(
                seg,
                ref_pool_dir=ref_dir,
                mouth_ref_pool_dir=mouth_ref_dir,
                video_fvd=video_fvd if seg.get("segment_type") == "speaking" else "",
            )
            row["fid"] = q.get("fid", "")
            row["fid_face"] = q.get("fid_face", "")
            row["fvd"] = q.get("fvd", "")
            if q.get("quality_notes"):
                notes.append(q["quality_notes"])
            if fvd_notes and seg.get("segment_type") == "speaking":
                notes.append(fvd_notes)
        except Exception as exc:
            notes.append(f"quality_failed:{exc}")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"QUAL {seg.get('segment_id')}: {exc}\n{traceback.format_exc()}\n")

        # Identity
        try:
            i = identity_eval.evaluate_segment(seg, identity_refs)
            row["csim"] = i.get("csim", "")
            row["csim_late"] = i.get("csim_late", "")
            if i.get("identity_notes"):
                notes.append(i["identity_notes"])
        except Exception as exc:
            notes.append(f"identity_failed:{exc}")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"ID {seg.get('segment_id')}: {exc}\n{traceback.format_exc()}\n")

        # Stability (jitter + warping)
        try:
            s = stability_eval.evaluate_segment(seg)
            row["jitter_score"] = s.get("jitter_score", "")
            row["warping_error"] = s.get("warping_error", "")
            if s.get("stability_notes"):
                notes.append(s["stability_notes"])
        except Exception as exc:
            notes.append(f"stability_failed:{exc}")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"STAB {seg.get('segment_id')}: {exc}\n{traceback.format_exc()}\n")

        if seg.get("segment_type") == "speaking":
            for label, fn, extra in (
                ("seam", seam_eval.evaluate_segment, {}),
                ("oral", oral_artifact_eval.evaluate_segment, {}),
                ("viseme", viseme_eval.evaluate_segment, {}),
                ("timing", speech_timing_eval.evaluate_segment, {}),
            ):
                try:
                    res = fn(seg, **extra)
                    for k, v in res.items():
                        if k.endswith("_notes"):
                            if v:
                                notes.append(f"{label}:{v}")
                        elif k in row:
                            row[k] = v
                except Exception as exc:
                    notes.append(f"{label}_failed:{exc}")
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(
                            f"{label.upper()} {seg.get('segment_id')}: {exc}\n"
                            f"{traceback.format_exc()}\n"
                        )
            try:
                p = pose_expression_eval.evaluate_segment(seg, upper_refs)
                row["upper_face_csim"] = p.get("upper_face_csim", "")
                row["expr_drift"] = p.get("expr_drift", "")
                if p.get("pose_notes"):
                    notes.append(p["pose_notes"])
            except Exception as exc:
                notes.append(f"pose_failed:{exc}")
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(f"POSE {seg.get('segment_id')}: {exc}\n{traceback.format_exc()}\n")

        row["notes"] = ";".join(notes)
        rows.append(row)
        print(
            f"{model_name}/{video_id} — metrics {seg.get('segment_id')} "
            f"[{seg.get('segment_type')}] done",
            flush=True,
        )
    return rows


def qc_sample(metrics_rows: List[dict], jobs_index: Dict[tuple, Path]) -> None:
    """Stratified QC sample weighted toward close-ups."""
    speaking = [r for r in metrics_rows if r.get("segment_type") == "speaking"]
    if not speaking:
        write_csv(
            config.QC_SAMPLE_CSV,
            [],
            [
                "segment_id", "model", "video_id", "shot_scale", "clip_path",
                "sync_naturalness", "identity_fidelity", "blending_artifacts", "notes",
            ],
        )
        return

    rng = random.Random(config.QC_SAMPLE_SEED)
    # Weight close-ups
    weights = []
    for r in speaking:
        w = config.QC_CLOSE_UP_WEIGHT if r.get("shot_scale") == "close-up" else 1.0
        weights.append(w)
    total_w = sum(weights)
    probs = [w / total_w for w in weights]
    n = max(1, int(round(len(speaking) * config.QC_SAMPLE_RATE)))
    n = min(n, len(speaking))
    chosen_idx = set()
    while len(chosen_idx) < n:
        chosen_idx.add(rng.choices(range(len(speaking)), weights=probs, k=1)[0])

    qc_rows = []
    for i in sorted(chosen_idx):
        r = speaking[i]
        key = (r["video_id"], r["model_name"])
        video_path = jobs_index.get(key)
        clip_path = ""
        if video_path and video_path.is_file():
            out = (
                config.QC_CLIPS_DIR
                / f"{r['model_name']}_{r['video_id']}_{r['segment_id']}.mp4"
            )
            try:
                export_clip(
                    video_path,
                    out,
                    float(r["start_time"]),
                    float(r["end_time"]),
                )
                clip_path = str(out)
            except Exception as exc:
                logging.warning("QC clip export failed: %s", exc)
        qc_rows.append(
            {
                "segment_id": r["segment_id"],
                "model": r["model_name"],
                "video_id": r["video_id"],
                "shot_scale": r.get("shot_scale", ""),
                "clip_path": clip_path,
                "sync_naturalness": "",
                "identity_fidelity": "",
                "blending_artifacts": "",
                "notes": "",
            }
        )
    write_csv(
        config.QC_SAMPLE_CSV,
        qc_rows,
        [
            "segment_id", "model", "video_id", "shot_scale", "clip_path",
            "sync_naturalness", "identity_fidelity", "blending_artifacts", "notes",
        ],
    )
    print(f"QC sample: {len(qc_rows)} segments → {config.QC_SAMPLE_CSV}", flush=True)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reference-free movie lip-sync eval")
    parser.add_argument(
        "--video",
        action="append",
        default=[],
        type=Path,
        help="Inferred model output video (repeatable). Overrides EVAL_ITEMS/INPUT_DIRS.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model name for each --video (or one name for all). Any string allowed.",
    )
    parser.add_argument(
        "--video-id",
        default=None,
        help="Shared video_id for CLI videos (default: each file stem)",
    )
    parser.add_argument(
        "--layer1-only",
        action="store_true",
        help="Run segmentation only (inspect segments.csv without metrics)",
    )
    parser.add_argument(
        "--reuse-segments",
        action="store_true",
        help="Skip Layer 1 if segments.csv already has rows for these videos",
    )
    args = parser.parse_args(argv)
    _setup_logging()

    # Fresh Layer-1 aggregate tables each full run (unless reuse)
    if not args.reuse_segments:
        for p in (config.SHOTS_CSV, config.TRACKLETS_CSV, config.SEGMENTS_CSV):
            if p.is_file():
                p.unlink()

    try:
        jobs = _iter_jobs(
            cli_videos=args.video or None,
            cli_models=args.model or None,
            video_id=args.video_id,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    if not jobs:
        print(
            "ERROR: no videos. Pass --video/--model or set INPUT_DIRS / EVAL_ITEMS "
            "in pipeline/config.py",
            flush=True,
        )
        return 1

    jobs_index = {(vid, model): path for vid, model, path in jobs}
    all_segments: List[dict] = []
    all_metrics: List[dict] = []

    for video_id, model_name, path in jobs:
        if not path.is_file():
            print(f"{model_name}/{video_id} — SKIP missing {path}", flush=True)
            continue
        print(f"\n=== {model_name}/{video_id} ===", flush=True)
        try:
            if args.reuse_segments and config.SEGMENTS_CSV.is_file():
                segs = [
                    r for r in read_csv(config.SEGMENTS_CSV)
                    if r.get("video_id") == video_id and r.get("model_name") == model_name
                ]
                if not segs:
                    segs = run_layer1(video_id, model_name, path)
            else:
                segs = run_layer1(video_id, model_name, path)
            all_segments.extend(segs)
        except Exception as exc:
            logging.error("Layer1 failed %s/%s: %s\n%s", model_name, video_id, exc, traceback.format_exc())
            print(f"{model_name}/{video_id} — Layer1 FAILED: {exc}", flush=True)
            continue

        if args.layer1_only:
            continue

        try:
            mrows = run_layer2_for_video(video_id, model_name, path, segs)
            all_metrics.extend(mrows)
        except Exception as exc:
            logging.error("Layer2 failed %s/%s: %s\n%s", model_name, video_id, exc, traceback.format_exc())
            print(f"{model_name}/{video_id} — Layer2 FAILED: {exc}", flush=True)

    print(f"\nLayer 1 segments written → {config.SEGMENTS_CSV}", flush=True)

    if args.layer1_only:
        print("Stopped after Layer 1 (--layer1-only).", flush=True)
        return 0

    all_metrics = apply_qc_flags(all_metrics)

    metric_fields = [
        "segment_id", "segment_type", "video_id", "model_name", "shot_id",
        "tracklet_id", "identity_id", "shot_scale", "difficulty_tags",
        "start_time", "end_time",
        "lse_d", "lse_c", "lse_offset",
        "lse_d_z", "lse_c_z", "lse_offset_corrected",
        "fid", "fid_face", "fvd", "fvd_mode",
        "csim", "csim_late",
        "jitter_score", "warping_error",
        "seam_score", "oral_artifact_score",
        "bilabial_frames", "bilabial_closure_fail_rate", "bilabial_score",
        "upper_face_csim", "expr_drift",
        "av_timing_ratio", "timing_stress_flag",
        "qc_pass", "qc_fail_reasons",
        "notes",
    ]
    write_csv(config.METRICS_RAW_CSV, all_metrics, metric_fields)
    print(f"Wrote {len(all_metrics)} metric rows → {config.METRICS_RAW_CSV}", flush=True)

    qc_fields = [
        "segment_id", "video_id", "model_name", "shot_id", "shot_scale",
        "difficulty_tags", "segment_type", "qc_pass", "qc_fail_reasons",
        "lse_d", "lse_c", "csim", "fvd_mode",
    ]
    qc_rows = [
        {k: r.get(k, "") for k in qc_fields}
        for r in all_metrics
        if r.get("segment_type") == "speaking"
    ]
    write_csv(config.QC_FLAGS_CSV, qc_rows, qc_fields)
    shot_qc = rollup_qc_shots(all_metrics)
    write_csv(
        config.QC_SHOTS_CSV,
        shot_qc,
        [
            "video_id", "model_name", "shot_id",
            "n_speaking_segments", "n_fail", "shot_qc_pass", "fail_reasons",
        ],
    )
    print(f"QC flags → {config.QC_FLAGS_CSV} ({len(qc_rows)} speaking rows)", flush=True)

    agg = aggregate(all_metrics)
    write_scale_csv(agg["by_scale"])
    write_difficulty_csv(agg["by_difficulty"])
    write_tier_summary_json(all_metrics, agg)
    print_summary(agg, all_metrics)
    qc_sample(all_metrics, jobs_index)

    try:
        from build_human_eval_pack import build_pack

        build_pack(metrics_path=config.METRICS_RAW_CSV, qc_path=config.QC_FLAGS_CSV, jobs_index=jobs_index)
    except Exception as exc:
        logging.warning("Human eval pack skipped: %s", exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
