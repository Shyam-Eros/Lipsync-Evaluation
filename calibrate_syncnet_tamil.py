#!/usr/bin/env python3
"""
Calibrate SyncNet LSE-D / LSE-C / AV offset on real Tamil talking-face clips.

Stock SyncNet is English-heavy; this script measures score distributions on
known-synced Tamil speech so eval can report z-scores and offset bias.

Layouts supported under --data-root:
  1) Sample folders:  <root>/<id>/{target.mp4, audio.wav}
  2) Flat videos:     <root>/*.mp4  (audio muxed in the file)

Usage:
  cd Evalution && source .venv/bin/activate
  python calibrate_syncnet_tamil.py --data-root /path/to/tamil/clips
  python calibrate_syncnet_tamil.py --data-root /path/a --data-root /path/b --max-clips 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import config
from pipeline.io_utils import ensure_dirs, extract_audio, extract_frames
from pipeline.metrics import sync_eval
from pipeline.metrics.shared_face_crop import (
    detect_faces_bgr,
    ensure_mouth_crops,
    get_s3fd,
    write_track_crops,
)

logger = logging.getLogger(__name__)


def _discover_clips(roots: Sequence[Path]) -> List[Tuple[str, Path, Optional[Path]]]:
    """Return list of (clip_id, video_path, optional_wav_path)."""
    found: List[Tuple[str, Path, Optional[Path]]] = []
    seen = set()
    for root in roots:
        root = Path(root)
        if not root.exists():
            logger.warning("data root missing: %s", root)
            continue
        # Sample-dir layout
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            video = None
            for name in ("target.mp4", "video.mp4", "clip.mp4"):
                cand = child / name
                if cand.is_file():
                    video = cand
                    break
            if video is None:
                vids = [
                    p for p in child.iterdir()
                    if p.is_file() and p.suffix.lower() in config.VIDEO_EXTS
                ]
                if vids:
                    video = sorted(vids)[0]
            if video is None:
                continue
            wav = None
            for name in ("audio.wav", "target_audio.wav", "audio.mp3"):
                cand = child / name
                if cand.is_file() and cand.suffix.lower() == ".wav":
                    wav = cand
                    break
            key = str(video.resolve())
            if key in seen:
                continue
            seen.add(key)
            found.append((child.name, video, wav))
        # Flat layout
        for p in sorted(root.iterdir()):
            if p.is_file() and p.suffix.lower() in config.VIDEO_EXTS:
                key = str(p.resolve())
                if key in seen:
                    continue
                seen.add(key)
                found.append((p.stem, p, None))
    return found


def _score_clip(
    clip_id: str,
    video_path: Path,
    wav_path: Optional[Path],
    work_root: Path,
    mouth_ref_dir: Optional[Path],
) -> Optional[Dict[str, float]]:
    work = work_root / clip_id
    work.mkdir(parents=True, exist_ok=True)
    frames_dir = work / "frames"
    out_wav = work / "audio.wav"
    crop_dir = work / "face_crops"

    extract_frames(video_path, frames_dir)
    # Prefer dedicated wav when present; always re-encode to 16 kHz mono
    audio_src = Path(wav_path) if wav_path and Path(wav_path).is_file() else video_path
    extract_audio(audio_src, out_wav)

    flist = sorted(frames_dir.glob("*.jpg"))
    if len(flist) < 25:
        raise RuntimeError(f"too few frames ({len(flist)})")

    detector = get_s3fd()
    frame_idxs: List[int] = []
    bboxes = []
    for i, fp in enumerate(flist):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        dets = detect_faces_bgr(img, detector=detector)
        if len(dets) == 0:
            continue
        # Largest face
        areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
        j = int(np.argmax(areas))
        frame_idxs.append(i)
        bboxes.append(dets[j, :4])

    if len(frame_idxs) < 25:
        raise RuntimeError(f"too few face frames ({len(frame_idxs)})")

    # Keep a contiguous run (longest streak of detections with gaps ≤ 2)
    bboxes_arr = np.asarray(bboxes, dtype=np.float64)
    n_written = write_track_crops(frames_dir, frame_idxs, bboxes_arr, crop_dir)
    if n_written < 25:
        raise RuntimeError(f"wrote only {n_written} crops")

    lse_d, lse_c, offset = sync_eval.evaluate_sync_track(crop_dir, out_wav)

    if mouth_ref_dir is not None:
        mouth_dir = ensure_mouth_crops(crop_dir)
        mouth_ref_dir.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(sorted(mouth_dir.glob("*.jpg"))[:40]):
            img = cv2.imread(str(p))
            if img is None:
                continue
            out = mouth_ref_dir / f"{clip_id}_{i:04d}.jpg"
            cv2.imwrite(str(out), img)

    return {
        "lse_d": float(lse_d),
        "lse_c": float(lse_c),
        "lse_offset": float(offset),
        "n_crops": float(n_written),
    }


def _percentile(vals: List[float], q: float) -> float:
    if not vals:
        return float("nan")
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate SyncNet on Tamil clips")
    parser.add_argument(
        "--data-root",
        action="append",
        type=Path,
        default=[],
        help="Root of Tamil clip dataset (repeatable)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Calibration JSON path (default: config.TAMIL_CALIBRATION_PATH)",
    )
    parser.add_argument(
        "--mouth-ref-dir",
        type=Path,
        default=None,
        help="Optional dir to write real Tamil mouth crops for FID/FVD refs",
    )
    parser.add_argument("--max-clips", type=int, default=0, help="0 = all")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Temp work dir (default: tmp/tamil_calib)",
    )
    args = parser.parse_args(argv)

    ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    roots = list(args.data_root)
    if not roots and config.TAMIL_CALIB_DATA_ROOT:
        roots = [Path(config.TAMIL_CALIB_DATA_ROOT)]
    if not roots:
        print(
            "ERROR: pass --data-root /path/to/tamil/clips "
            "(or set TAMIL_CALIB_DATA_ROOT in pipeline/config.py)",
            flush=True,
        )
        return 1

    clips = _discover_clips(roots)
    if args.max_clips and args.max_clips > 0:
        clips = clips[: args.max_clips]
    if not clips:
        print(f"ERROR: no clips found under {roots}", flush=True)
        return 1

    work_root = Path(args.work_dir or (config.TEMP_DIR / "tamil_calib"))
    work_root.mkdir(parents=True, exist_ok=True)
    mouth_ref = args.mouth_ref_dir
    if mouth_ref is None:
        mouth_ref = config.RESULTS_DIR / "tamil_mouth_ref"
    mouth_ref = Path(mouth_ref)

    print(f"Calibrating SyncNet on {len(clips)} Tamil clips …", flush=True)
    per_clip: List[Dict] = []
    lse_ds, lse_cs, offsets = [], [], []
    fvd_real_dirs: List[str] = []

    for clip_id, video, wav in clips:
        print(f"  {clip_id} — {video}", flush=True)
        try:
            scores = _score_clip(clip_id, video, wav, work_root, mouth_ref)
        except Exception as exc:
            logger.warning("skip %s: %s", clip_id, exc)
            logger.debug(traceback.format_exc())
            continue
        if scores is None:
            continue
        face_dir = work_root / clip_id / "face_crops"
        per_clip.append(
            {
                "clip_id": clip_id,
                "video": str(video),
                "face_crop_dir": str(face_dir),
                **scores,
            }
        )
        if face_dir.is_dir():
            fvd_real_dirs.append(str(face_dir))
        lse_ds.append(scores["lse_d"])
        lse_cs.append(scores["lse_c"])
        offsets.append(scores["lse_offset"])
        print(
            f"    LSE-D={scores['lse_d']:.3f} LSE-C={scores['lse_c']:.3f} "
            f"offset={scores['lse_offset']:.1f}",
            flush=True,
        )

    if len(lse_ds) < 3:
        print(
            f"ERROR: need ≥3 successful clips for calibration, got {len(lse_ds)}",
            flush=True,
        )
        return 1

    calib = {
        "n_clips": len(lse_ds),
        "data_roots": [str(r) for r in roots],
        "lse_d_mean": float(np.mean(lse_ds)),
        "lse_d_std": float(np.std(lse_ds)) or 1.0,
        "lse_d_p10": _percentile(lse_ds, 10),
        "lse_d_p50": _percentile(lse_ds, 50),
        "lse_d_p90": _percentile(lse_ds, 90),
        "lse_c_mean": float(np.mean(lse_cs)),
        "lse_c_std": float(np.std(lse_cs)) or 1.0,
        "lse_c_p10": _percentile(lse_cs, 10),
        "lse_c_p50": _percentile(lse_cs, 50),
        "lse_c_p90": _percentile(lse_cs, 90),
        "offset_median": float(np.median(offsets)),
        "offset_mean": float(np.mean(offsets)),
        "offset_std": float(np.std(offsets)) or 1.0,
        "mouth_ref_dir": str(mouth_ref),
        "fvd_real_dirs": fvd_real_dirs,
        "healthy_lse_d_max": _percentile(lse_ds, 90),
        "healthy_lse_c_min": _percentile(lse_cs, 10),
        "qc_lse_d_max": _percentile(lse_ds, 90),
        "qc_lse_c_min": _percentile(lse_cs, 10),
        "qc_offset_abs_max": max(
            5.0,
            abs(float(np.median(offsets))) + 2.0 * (float(np.std(offsets)) or 1.0),
        ),
        "qc_csim_min": 0.45,
        "av_timing_ratio_min": 0.55,
        "av_timing_ratio_max": 1.85,
        "per_clip": per_clip,
    }

    out_path = Path(args.out or config.TAMIL_CALIBRATION_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)
    # Invalidate cached calib in this process
    sync_eval.load_tamil_calibration(out_path, force=True)

    print("\n========== Tamil SyncNet calibration ==========", flush=True)
    print(f"clips:     {calib['n_clips']}", flush=True)
    print(
        f"LSE-D:     mean={calib['lse_d_mean']:.3f} std={calib['lse_d_std']:.3f} "
        f"p50={calib['lse_d_p50']:.3f} (healthy ≤ p90={calib['healthy_lse_d_max']:.3f})",
        flush=True,
    )
    print(
        f"LSE-C:     mean={calib['lse_c_mean']:.3f} std={calib['lse_c_std']:.3f} "
        f"p50={calib['lse_c_p50']:.3f} (healthy ≥ p10={calib['healthy_lse_c_min']:.3f})",
        flush=True,
    )
    print(f"AV offset: median bias={calib['offset_median']:.2f} frames", flush=True)
    print(f"wrote:     {out_path}", flush=True)
    print(f"mouth ref: {mouth_ref}", flush=True)
    print("================================================\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
