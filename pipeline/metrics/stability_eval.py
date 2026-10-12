"""Frame-to-frame flicker (jitter) and warping / blend instability (lower is better)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from pipeline import config
from pipeline.metrics.shared_face_crop import ensure_mouth_crops, list_crop_paths, lower_face_roi

logger = logging.getLogger(__name__)

_MP_MESH = None


def _get_face_mesh():
    global _MP_MESH
    if _MP_MESH is not None:
        return _MP_MESH
    try:
        import mediapipe as mp

        _MP_MESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        return _MP_MESH
    except Exception as exc:
        logger.warning("MediaPipe unavailable (%s); using optical-flow jitter", exc)
        _MP_MESH = False
        return None


# Lip landmark indices (MediaPipe Face Mesh)
_LIP_IDX = list(range(61, 88)) + list(range(291, 308))


def _landmarks_from_crop(image_bgr: np.ndarray, mesh) -> Optional[np.ndarray]:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    res = mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    if config.JITTER_LANDMARK_SUBSET == "lips":
        pts = np.array([[lm[i].x, lm[i].y] for i in _LIP_IDX if i < len(lm)], dtype=np.float32)
    else:
        pts = np.array([[p.x, p.y] for p in lm], dtype=np.float32)
    return pts


def _flow_jitter(paths: List[Path]) -> float:
    prev = None
    mags = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (112, 112))
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev, img, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            mags.append(float(np.mean(mag)))
        prev = img
    if len(mags) < 2:
        raise RuntimeError("too few frames for flow jitter")
    arr = np.asarray(mags)
    k = min(5, len(arr) // 2 * 2 + 1)
    if k >= 3:
        kernel = np.ones(k) / k
        trend = np.convolve(arr, kernel, mode="same")
        resid = arr - trend
    else:
        resid = arr - arr.mean()
    return float(np.std(resid))


def compute_jitter(crop_dir: Path) -> float:
    paths = list_crop_paths(crop_dir)
    if len(paths) < 3:
        raise RuntimeError("need ≥3 crops for jitter")
    mesh = _get_face_mesh()
    if not mesh:
        return _flow_jitter(paths)

    series = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        pts = _landmarks_from_crop(img, mesh)
        if pts is None:
            continue
        series.append(pts.reshape(-1))
    if len(series) < 3:
        return _flow_jitter(paths)
    X = np.stack(series)
    deltas = np.linalg.norm(np.diff(X, axis=0), axis=1)
    k = min(5, len(deltas) // 2 * 2 + 1)
    if k >= 3:
        trend = np.convolve(deltas, np.ones(k) / k, mode="same")
        resid = deltas - trend
    else:
        resid = deltas - deltas.mean()
    return float(np.std(resid))


def compute_warping_error(crop_dir: Path, *, max_pairs: int = 64) -> float:
    """
    Optical-flow warp t → t+1 on lower-face ROI; mean photometric residual.

    High values = unstable blending / geometric jitter between frames.
    """
    face_paths = list_crop_paths(crop_dir)
    if len(face_paths) < 3:
        # Try mouth dir directly
        try:
            face_paths = list_crop_paths(ensure_mouth_crops(crop_dir))
        except Exception:
            pass
    if len(face_paths) < 3:
        raise RuntimeError("need ≥3 crops for warping error")

    # Subsample evenly if long
    if len(face_paths) > max_pairs + 1:
        idx = np.linspace(0, len(face_paths) - 1, max_pairs + 1).astype(int)
        face_paths = [face_paths[i] for i in idx]

    residuals = []
    prev_gray = None
    prev_roi = None
    for p in face_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        roi = lower_face_roi(img, out_size=112)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None and prev_roi is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            h, w = gray.shape
            grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (grid_x + flow[..., 0]).astype(np.float32)
            map_y = (grid_y + flow[..., 1]).astype(np.float32)
            warped = cv2.remap(
                prev_roi,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            # Photometric residual in [0, 1]
            err = np.mean(np.abs(warped.astype(np.float32) - roi.astype(np.float32))) / 255.0
            residuals.append(float(err))
        prev_gray = gray
        prev_roi = roi

    if len(residuals) < 2:
        raise RuntimeError("too few pairs for warping error")
    return float(np.mean(residuals))


def evaluate_segment(segment: dict) -> dict:
    out = {"jitter_score": "", "warping_error": "", "stability_notes": ""}
    if segment.get("segment_type") not in ("speaking", "non_speaking_face"):
        out["stability_notes"] = "skipped_no_face"
        return out
    crop = segment.get("face_crop_dir")
    if not crop:
        out["stability_notes"] = "missing_crops"
        return out
    crop_path = Path(crop)
    try:
        out["jitter_score"] = f"{compute_jitter(crop_path):.6f}"
    except Exception as exc:
        out["stability_notes"] += f"jitter_failed:{exc};"
    try:
        out["warping_error"] = f"{compute_warping_error(crop_path):.6f}"
    except Exception as exc:
        out["stability_notes"] += f"warp_failed:{exc};"
    return out
