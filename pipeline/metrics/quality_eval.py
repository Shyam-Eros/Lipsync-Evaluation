"""
Mouth FID + I3D FVD — reference-free visual quality.

CRITICAL: Do NOT accept or use an English-synced source/GT video as the FID
reference. References come from non_speaking mouth crops and/or Tamil real
mouth refs (REAL_MOUTH_REF_DIR / calibration).
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from pipeline import config
from pipeline.metrics.shared_face_crop import (
    ensure_mouth_crops,
    list_crop_paths,
)

logger = logging.getLogger(__name__)


def _assert_no_gt_reference(**kwargs) -> None:
    banned = ("gt_video", "source_video", "ground_truth", "original_video", "gt_frames")
    for k in banned:
        if kwargs.get(k) is not None:
            raise RuntimeError(
                f"quality_eval refused banned reference argument '{k}'. "
                "Do not compare to English-synced source frames."
            )


def _gather_face_ref_dirs(
    segments: Sequence[dict], real_face_dir: Optional[Path]
) -> List[Path]:
    dirs = []
    for s in segments:
        if s.get("segment_type") == "non_speaking_face" and s.get("face_crop_dir"):
            p = Path(s["face_crop_dir"])
            if p.is_dir() and list_crop_paths(p):
                dirs.append(p)
    if real_face_dir and Path(real_face_dir).is_dir():
        dirs.append(Path(real_face_dir))
    return dirs


def _gather_mouth_ref_dirs(
    segments: Sequence[dict],
    real_mouth_dir: Optional[Path] = None,
) -> List[Path]:
    """Mouth crop dirs: non_speaking faces + optional Tamil REAL_MOUTH_REF_DIR."""
    dirs: List[Path] = []
    for s in segments:
        if s.get("segment_type") == "non_speaking_face" and s.get("face_crop_dir"):
            try:
                md = ensure_mouth_crops(Path(s["face_crop_dir"]))
                if list_crop_paths(md):
                    dirs.append(md)
            except Exception as exc:
                logger.debug("mouth ref skip %s: %s", s.get("segment_id"), exc)
    # Flat Tamil mouth ref dir (images directly inside)
    candidates = []
    if real_mouth_dir:
        candidates.append(Path(real_mouth_dir))
    if config.REAL_MOUTH_REF_DIR:
        candidates.append(Path(config.REAL_MOUTH_REF_DIR))
    calib_path = Path(config.TAMIL_CALIBRATION_PATH)
    if calib_path.is_file():
        try:
            with calib_path.open(encoding="utf-8") as f:
                calib = json.load(f)
            mref = calib.get("mouth_ref_dir")
            if mref:
                candidates.append(Path(mref))
        except Exception:
            pass
    for c in candidates:
        if c.is_dir() and (list_crop_paths(c) or any(c.glob("*.png"))):
            dirs.append(c)
    return dirs


def _pool_images_to_dir(dirs: List[Path], out_dir: Path, max_images: int = 2000) -> int:
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for d in dirs:
        paths = list_crop_paths(d)
        if not paths:
            paths = sorted(Path(d).glob("*.png"))
        for p in paths:
            shutil.copy2(p, out_dir / f"{n:06d}{p.suffix}")
            n += 1
            if n >= max_images:
                return n
    return n


def compute_fid(gen_dir: Path, ref_dir: Path, device: Optional[str] = None) -> float:
    from pytorch_fid.fid_score import calculate_fid_given_paths

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return float(
        calculate_fid_given_paths(
            [str(ref_dir), str(gen_dir)],
            batch_size=config.FID_BATCH_SIZE,
            device=device,
            dims=config.FID_DIMS,
            num_workers=0,
        )
    )


def compute_fvd_proxy(crop_dir: Path) -> float:
    """Lightweight temporal proxy (fallback when I3D unavailable)."""
    import cv2

    paths = list_crop_paths(crop_dir)
    n = int(config.FVD_NUM_FRAMES)
    if len(paths) < config.FVD_MIN_FRAMES:
        raise RuntimeError(f"too few frames for FVD proxy ({len(paths)})")
    if len(paths) > n:
        start = (len(paths) - n) // 2
        paths = paths[start : start + n]
    feats = []
    prev = None
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        small = cv2.resize(img, (64, 64)).astype(np.float32) / 255.0
        if prev is not None:
            feats.append((small - prev).reshape(-1))
        prev = small
    if len(feats) < 4:
        raise RuntimeError("insufficient frame deltas for FVD proxy")
    X = np.stack(feats)
    mu = X.mean(axis=0)
    return float(np.mean(mu ** 2) * 1000.0)


def _fvd_real_face_dirs() -> List[Path]:
    dirs: List[Path] = []
    calib_path = Path(config.TAMIL_CALIBRATION_PATH)
    if not calib_path.is_file():
        return dirs
    try:
        with calib_path.open(encoding="utf-8") as f:
            calib = json.load(f)
    except Exception:
        return dirs
    for p in calib.get("fvd_real_dirs") or []:
        path = Path(p)
        if path.is_dir():
            dirs.append(path)
    for row in calib.get("per_clip") or []:
        p = row.get("face_crop_dir")
        if p and Path(p).is_dir() and Path(p) not in dirs:
            dirs.append(Path(p))
    return dirs


def compute_video_fvd(
    speaking_segments: Sequence[dict],
    *,
    device: Optional[str] = None,
) -> Dict[str, str]:
    """
    True I3D FVD for a video's speaking segments vs Tamil real mouth clips.
    Returns dict with fvd + notes. Same score broadcast to all speaking rows.
    """
    _assert_no_gt_reference()
    fake_dirs = [
        Path(s["face_crop_dir"])
        for s in speaking_segments
        if s.get("segment_type") == "speaking" and s.get("face_crop_dir")
    ]
    real_dirs = _fvd_real_face_dirs()
    if len(fake_dirs) < 1:
        return {"fvd": "", "fvd_mode": "", "fvd_notes": "no_speaking_crops"}
    try:
        from pipeline.metrics import fvd_i3d

        if len(real_dirs) < 2:
            # Fall back: use non_speaking face dirs from same video as "real" motion prior
            # is weak — prefer proxy if no Tamil calib
            raise RuntimeError("need Tamil calibration fvd_real_dirs (≥2 clips)")
        score = fvd_i3d.compute_fvd_i3d(fake_dirs, real_dirs, device=device)
        return {"fvd": f"{score:.6f}", "fvd_mode": "i3d", "fvd_notes": "i3d"}
    except Exception as exc:
        logger.warning("I3D FVD failed (%s); trying proxy on first speaking segment", exc)
        try:
            mouth = ensure_mouth_crops(fake_dirs[0])
            proxy = compute_fvd_proxy(mouth)
            return {
                "fvd": f"{proxy:.6f}",
                "fvd_mode": "proxy",
                "fvd_notes": f"fvd_proxy:{exc}",
            }
        except Exception as exc2:
            return {"fvd": "", "fvd_mode": "", "fvd_notes": f"fvd_failed:{exc};proxy_failed:{exc2}"}


def evaluate_segment(
    segment: dict,
    *,
    ref_pool_dir: Optional[Path] = None,
    mouth_ref_pool_dir: Optional[Path] = None,
    video_fvd: Optional[str] = None,
    **kwargs,
) -> dict:
    """
    Mouth FID for speaking segments; optional face FID; FVD filled from video_fvd.
    """
    _assert_no_gt_reference(**kwargs)
    out = {
        "fid": "",
        "fid_face": "",
        "fvd": video_fvd if video_fvd is not None else "",
        "quality_notes": "",
    }
    if segment.get("segment_type") != "speaking":
        out["quality_notes"] = "skipped_non_speaking"
        return out
    crop = Path(segment["face_crop_dir"]) if segment.get("face_crop_dir") else None
    if crop is None or not crop.is_dir():
        out["quality_notes"] = "missing_crops"
        return out

    # Per-segment proxy only if video-level FVD not provided
    if video_fvd is None:
        try:
            mouth = ensure_mouth_crops(crop)
            out["fvd"] = f"{compute_fvd_proxy(mouth):.6f}"
            out["quality_notes"] += "fvd_proxy_segment;"
        except Exception as exc:
            out["quality_notes"] += f"fvd_failed:{exc};"
            logger.warning("FVD failed %s: %s", segment.get("segment_id"), exc)

    # Mouth FID (primary `fid` column)
    try:
        mouth = ensure_mouth_crops(crop)
        if mouth_ref_pool_dir and Path(mouth_ref_pool_dir).is_dir() and list_crop_paths(
            Path(mouth_ref_pool_dir)
        ):
            out["fid"] = f"{compute_fid(mouth, Path(mouth_ref_pool_dir)):.6f}"
        else:
            out["quality_notes"] += "fid_skipped_no_mouth_ref;"
    except Exception as exc:
        out["quality_notes"] += f"fid_failed:{exc};"
        logger.warning("mouth FID failed %s: %s", segment.get("segment_id"), exc)

    # Optional full-face FID
    if ref_pool_dir and Path(ref_pool_dir).is_dir() and list_crop_paths(Path(ref_pool_dir)):
        try:
            out["fid_face"] = f"{compute_fid(crop, Path(ref_pool_dir)):.6f}"
        except Exception as exc:
            out["quality_notes"] += f"fid_face_failed:{exc};"

    return out


def build_reference_pool(
    segments: Sequence[dict],
    out_dir: Path,
    real_face_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Build full-face FID reference from non_speaking_face (+ optional REAL_FACE_REF_DIR)."""
    _assert_no_gt_reference()
    dirs = _gather_face_ref_dirs(segments, real_face_dir or config.REAL_FACE_REF_DIR)
    if not dirs:
        logger.warning("No non_speaking_face / REAL_FACE_REF_DIR for face FID reference")
        return None
    n = _pool_images_to_dir(list(dirs), Path(out_dir))
    if n < 10:
        logger.warning("face FID reference pool too small (%d images)", n)
        return None
    logger.info("face FID reference pool: %d images → %s", n, out_dir)
    return Path(out_dir)


def build_mouth_reference_pool(
    segments: Sequence[dict],
    out_dir: Path,
    real_mouth_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Build mouth/lower-face FID reference pool."""
    _assert_no_gt_reference()
    dirs = _gather_mouth_ref_dirs(segments, real_mouth_dir=real_mouth_dir)
    if not dirs:
        logger.warning("No mouth reference dirs for FID")
        return None
    n = _pool_images_to_dir(list(dirs), Path(out_dir))
    if n < 10:
        logger.warning("mouth FID reference pool too small (%d images)", n)
        return None
    logger.info("mouth FID reference pool: %d images → %s", n, out_dir)
    return Path(out_dir)
