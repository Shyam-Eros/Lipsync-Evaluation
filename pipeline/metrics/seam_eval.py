"""Seam / blend boundary score on mouth crops (lower is better)."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from pipeline.metrics.shared_face_crop import ensure_mouth_crops, list_crop_paths, lower_face_roi

logger = logging.getLogger(__name__)


def compute_seam_score(crop_dir: Path, max_frames: int = 80) -> float:
    paths = list_crop_paths(crop_dir)
    if len(paths) < 3:
        raise RuntimeError("too few crops for seam score")
    if len(paths) > max_frames:
        step = len(paths) // max_frames
        paths = paths[:: max(1, step)][:max_frames]
    scores = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        mouth = lower_face_roi(img, out_size=112)
        gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        h = gray.shape[0]
        band = max(3, h // 10)
        top = float(np.mean(np.abs(lap[:band, :])))
        bot = float(np.mean(np.abs(lap[-band:, :])))
        mid = float(np.mean(np.abs(lap[band : h - band, :])))
        if mid < 1e-6:
            continue
        scores.append((top + bot) / (2.0 * mid))
    if not scores:
        raise RuntimeError("no seam scores")
    return float(np.mean(scores))


def evaluate_segment(segment: dict) -> dict:
    out = {"seam_score": "", "seam_notes": ""}
    if segment.get("segment_type") not in ("speaking", "non_speaking_face"):
        out["seam_notes"] = "skipped_no_face"
        return out
    crop = segment.get("face_crop_dir")
    if not crop:
        out["seam_notes"] = "missing_crops"
        return out
    try:
        ensure_mouth_crops(Path(crop))
        out["seam_score"] = f"{compute_seam_score(Path(crop)):.6f}"
    except Exception as exc:
        out["seam_notes"] = f"seam_failed:{exc}"
    return out
