"""Upper-face identity retention vs in-clip non-speaking ref (reference-free)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from pipeline.metrics.shared_face_crop import list_crop_paths
from pipeline.segmentation.identity_clustering import _arcface_from_crop, _l2, embed_crop_dir, get_face_app

logger = logging.getLogger(__name__)


def _upper_crop(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    roi = image_bgr[: int(h * 0.55), :]
    return cv2.resize(roi, (112, 112), interpolation=cv2.INTER_LINEAR)


def _embed_upper_dir(crop_dir: Path, max_n: int = 8) -> Optional[np.ndarray]:
    app = get_face_app()
    embs = []
    for p in list_crop_paths(crop_dir)[:max_n]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        up = _upper_crop(img)
        if app is not None:
            e = _arcface_from_crop(up, app)
        else:
            continue
        if e is not None:
            embs.append(e)
    if not embs:
        return None
    return _l2(np.mean(np.stack(embs), axis=0))


def build_upper_face_refs(segments: List[dict]) -> Dict[str, np.ndarray]:
    by_id: Dict[str, List[dict]] = {}
    for s in segments:
        iid = str(s.get("identity_id", ""))
        if not iid or not s.get("face_crop_dir"):
            continue
        if s.get("segment_type") in ("speaking", "non_speaking_face"):
            by_id.setdefault(iid, []).append(s)
    refs = {}
    for iid, rows in by_id.items():
        preferred = [r for r in rows if r.get("segment_type") == "non_speaking_face"] or rows
        emb = _embed_upper_dir(Path(preferred[0]["face_crop_dir"]))
        if emb is not None:
            refs[iid] = emb
    return refs


def evaluate_segment(segment: dict, upper_refs: Dict[str, np.ndarray], **kwargs) -> dict:
    out = {"upper_face_csim": "", "expr_drift": "", "pose_notes": ""}
    if segment.get("segment_type") not in ("speaking", "non_speaking_face"):
        out["pose_notes"] = "skipped_no_face"
        return out
    iid = str(segment.get("identity_id", ""))
    if iid not in upper_refs or not segment.get("face_crop_dir"):
        out["pose_notes"] = "no_upper_ref"
        return out
    emb = _embed_upper_dir(Path(segment["face_crop_dir"]))
    ref = upper_refs[iid]
    if emb is None:
        out["pose_notes"] = "upper_embed_failed"
        return out
    csim = float(np.dot(emb, ref))
    out["upper_face_csim"] = f"{csim:.6f}"
    # Drift: 1 - csim (higher = worse expression/pose change)
    out["expr_drift"] = f"{max(0.0, 1.0 - csim):.6f}"
    return out
