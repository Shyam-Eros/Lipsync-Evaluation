"""Mouth openness / aspect ratio from face crops (MediaPipe or gradient fallback)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from pipeline.metrics.shared_face_crop import list_crop_paths, lower_face_roi

_MP_MESH = None
_LIP_IDX = list(range(61, 88)) + list(range(291, 308))


def _get_mesh():
    global _MP_MESH
    if _MP_MESH is not None:
        return _MP_MESH if _MP_MESH is not False else None
    try:
        import mediapipe as mp

        _MP_MESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        return _MP_MESH
    except Exception:
        _MP_MESH = False
        return None


def mouth_aspect_ratio(image_bgr: np.ndarray) -> Optional[float]:
    """Higher = more open mouth."""
    mesh = _get_mesh()
    if mesh:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = mesh.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            pts = np.array([[lm[i].x, lm[i].y] for i in _LIP_IDX if i < len(lm)], dtype=np.float32)
            if len(pts) >= 6:
                y_span = float(pts[:, 1].max() - pts[:, 1].min())
                x_span = float(pts[:, 0].max() - pts[:, 0].min()) + 1e-6
                return y_span / x_span
    roi = lower_face_roi(image_bgr, out_size=64)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Vertical gradient spread as openness proxy
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.std(gy) / 64.0)


def mar_series(crop_dir: Path, max_frames: int = 500) -> List[float]:
    vals = []
    for p in list_crop_paths(crop_dir)[:max_frames]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        m = mouth_aspect_ratio(img)
        if m is not None:
            vals.append(m)
    return vals


def frame_index_for_time(crop_dir: Path, t_sec: float, fps: float = 25.0) -> int:
    paths = list_crop_paths(crop_dir)
    if not paths:
        return 0
    idx = int(round(t_sec * fps))
    return max(0, min(idx, len(paths) - 1))


def mar_at_crop_index(crop_dir: Path, idx: int) -> Optional[float]:
    paths = list_crop_paths(crop_dir)
    if idx < 0 or idx >= len(paths):
        return None
    img = cv2.imread(str(paths[idx]))
    if img is None:
        return None
    return mouth_aspect_ratio(img)
