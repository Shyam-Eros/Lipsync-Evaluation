"""Teeth / tongue artifact heuristic on open-mouth frames (lower is better)."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from pipeline.metrics.mouth_geometry import mar_series
from pipeline.metrics.shared_face_crop import list_crop_paths, lower_face_roi

logger = logging.getLogger(__name__)


def compute_oral_artifact_score(crop_dir: Path) -> float:
    paths = list_crop_paths(crop_dir)
    mars = mar_series(crop_dir)
    if not mars:
        raise RuntimeError("no MAR series")
    thresh = float(np.percentile(mars, 60))
    scores = []
    for p, mar in zip(paths[: len(mars)], mars):
        if mar < thresh:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        roi = lower_face_roi(img, out_size=96)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # oral cavity: high saturation + value variance + HF texture
        sat = hsv[:, :, 1].astype(np.float32) / 255.0
        val = hsv[:, :, 2].astype(np.float32) / 255.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        hf = cv2.Laplacian(gray, cv2.CV_32F)
        pink = float(np.mean((sat > 0.35) & (val > 0.4)))
        texture = float(np.mean(np.abs(hf)))
        scores.append(0.5 * pink + 0.5 * min(texture / 0.15, 1.0))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def evaluate_segment(segment: dict) -> dict:
    out = {"oral_artifact_score": "", "oral_notes": ""}
    if segment.get("segment_type") not in ("speaking", "non_speaking_face"):
        out["oral_notes"] = "skipped_no_face"
        return out
    crop = segment.get("face_crop_dir")
    if not crop:
        out["oral_notes"] = "missing_crops"
        return out
    try:
        out["oral_artifact_score"] = f"{compute_oral_artifact_score(Path(crop)):.6f}"
    except Exception as exc:
        out["oral_notes"] = f"oral_failed:{exc}"
    return out
