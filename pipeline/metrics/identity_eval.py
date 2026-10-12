"""
CSIM — identity self-consistency across shots (not vs any GT video).

Reference embedding = clearest crop for each identity_id within the same clip.
Also reports csim_late on the last CSIM_LATE_FRAC of frames (identity drift).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from pipeline import config
from pipeline.metrics.shared_face_crop import list_crop_paths
from pipeline.segmentation.identity_clustering import (
    _arcface_from_crop,
    _hist_embedding,
    _l2,
    embed_crop_dir,
    get_face_app,
)

logger = logging.getLogger(__name__)


def _assert_no_gt(**kwargs):
    for k in ("gt_video", "source_video", "ground_truth", "original_video"):
        if kwargs.get(k) is not None:
            raise RuntimeError(f"identity_eval refused banned arg '{k}'")


def _sharpness(path: Path) -> float:
    img = cv2.imread(str(path))
    if img is None:
        return -1.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _embed_paths(paths: Sequence[Path]) -> Optional[np.ndarray]:
    """Mean L2-normalized embedding over readable crops (same family as embed_crop_dir)."""
    if not paths:
        return None
    app = get_face_app()
    embs = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if app is not None:
            emb = _arcface_from_crop(img, app)
        else:
            emb = _hist_embedding(img)
        if emb is not None:
            embs.append(np.asarray(emb, dtype=np.float32))
    if not embs:
        return None
    return _l2(np.mean(np.stack(embs), axis=0))


def build_identity_references(segments: Sequence[dict]) -> Dict[str, np.ndarray]:
    """Pick best crop dir per identity_id (prefer non_speaking_face, then largest/sharpest)."""
    by_id: Dict[str, List[dict]] = {}
    for s in segments:
        iid = str(s.get("identity_id", ""))
        if iid == "" or not s.get("face_crop_dir"):
            continue
        if s.get("segment_type") in ("speaking", "non_speaking_face"):
            by_id.setdefault(iid, []).append(s)

    refs: Dict[str, np.ndarray] = {}
    for iid, rows in by_id.items():
        preferred = [r for r in rows if r.get("segment_type") == "non_speaking_face"] or rows

        def score(r):
            crops = list_crop_paths(Path(r["face_crop_dir"]))
            if not crops:
                return -1.0
            return float(np.mean([_sharpness(p) for p in crops[:5]]))

        best = max(preferred, key=score)
        emb = embed_crop_dir(Path(best["face_crop_dir"]))
        if emb is not None:
            refs[iid] = emb
    return refs


def evaluate_segment(segment: dict, identity_refs: Dict[str, np.ndarray], **kwargs) -> dict:
    _assert_no_gt(**kwargs)
    out = {"csim": "", "csim_late": "", "identity_notes": ""}
    if segment.get("segment_type") not in ("speaking", "non_speaking_face"):
        out["identity_notes"] = "skipped_no_face"
        return out
    iid = str(segment.get("identity_id", ""))
    if iid not in identity_refs:
        out["identity_notes"] = "no_identity_ref"
        return out
    if not segment.get("face_crop_dir"):
        out["identity_notes"] = "missing_crops"
        return out
    crop_dir = Path(segment["face_crop_dir"])
    emb = embed_crop_dir(crop_dir)
    if emb is None:
        out["identity_notes"] = "embed_failed"
        return out
    ref = identity_refs[iid]
    if emb.shape != ref.shape:
        out["identity_notes"] = (
            f"dim_mismatch_emb{emb.shape[0]}_ref{ref.shape[0]}"
        )
        return out
    csim = float(np.dot(emb, ref))
    out["csim"] = f"{csim:.6f}"

    # Late-clip identity: last CSIM_LATE_FRAC of frames
    paths = list_crop_paths(crop_dir)
    if len(paths) >= 4:
        n_late = max(1, int(round(len(paths) * float(config.CSIM_LATE_FRAC))))
        late_paths = paths[-n_late:]
        late_emb = _embed_paths(late_paths)
        if late_emb is not None and late_emb.shape == ref.shape:
            out["csim_late"] = f"{float(np.dot(late_emb, ref)):.6f}"
        else:
            out["identity_notes"] += "csim_late_failed;"
    else:
        out["identity_notes"] += "csim_late_too_short;"
    return out
