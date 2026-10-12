"""Layer 1 stage 3 — cross-shot identity clustering via ArcFace embeddings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from pipeline import config
from pipeline.metrics.shared_face_crop import list_crop_paths

logger = logging.getLogger(__name__)

_FACE_APP = None
_ARCFACE_DIM = 512
_HIST_DIM = 256  # 16x16 HSV hist


def get_face_app():
    """Lazy InsightFace FaceAnalysis (ArcFace). Falls back to None if unavailable."""
    global _FACE_APP
    if _FACE_APP is not None:
        return _FACE_APP if _FACE_APP is not False else None
    try:
        from insightface.app import FaceAnalysis
        import torch

        # Prefer CPU for ORT when CUDA EP is missing (avoids noisy warnings)
        providers = ["CPUExecutionProvider"]
        if torch.cuda.is_available():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        root = str(config.INSIGHTFACE_ROOT)
        Path(root).mkdir(parents=True, exist_ok=True)
        app = FaceAnalysis(name=config.INSIGHTFACE_MODEL, root=root, providers=providers)
        ctx = 0 if torch.cuda.is_available() else -1
        try:
            app.prepare(ctx_id=ctx, det_size=(640, 640))
        except Exception:
            app.prepare(ctx_id=-1, det_size=(640, 640))
        _FACE_APP = app
        logger.info("InsightFace %s ready", config.INSIGHTFACE_MODEL)
        return _FACE_APP
    except Exception as exc:
        logger.warning(
            "InsightFace unavailable (%s) — using histogram fallback embeddings",
            exc,
        )
        _FACE_APP = False
        return None


def _l2(emb: np.ndarray) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    return emb / (np.linalg.norm(emb) + 1e-8)


def _hist_embedding(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256]).flatten()
    return _l2(hist.astype(np.float32))


def _arcface_from_crop(image_bgr: np.ndarray, app) -> Optional[np.ndarray]:
    """
    Embed an already-cropped face.

    Prefer recognition.get_feat on a 112×112 resize (crops are SyncNet 224 faces,
    so detector often fails on them). Fall back to app.get() if needed.
    """
    # 1) Direct recognition on resized crop (no re-detection)
    rec = None
    if hasattr(app, "models"):
        rec = app.models.get("recognition") or app.models.get("recognition")
    if rec is not None:
        try:
            face112 = cv2.resize(image_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
            feat = rec.get_feat(face112)
            if feat is not None:
                return _l2(np.asarray(feat, dtype=np.float32))
        except Exception:
            pass

    # 2) Full FaceAnalysis on crop (may still detect)
    try:
        faces = app.get(image_bgr)
        if faces:
            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )
            if getattr(face, "embedding", None) is not None:
                return _l2(np.asarray(face.embedding, dtype=np.float32))
            if getattr(face, "normed_embedding", None) is not None:
                return _l2(np.asarray(face.normed_embedding, dtype=np.float32))
    except Exception:
        pass
    return None


def embed_crop_dir(crop_dir: Path, top_n: int | None = None) -> Optional[np.ndarray]:
    """
    Median embedding over sharpest top-N crops.

    Uses a *single* embedding family per call:
    - If InsightFace is up: ArcFace only (skip crops that fail) → always 512-d
    - Else: histogram only → always 256-d
    Never mixes 512-d and 256-d in one vector (that caused the crash).
    """
    top_n = int(top_n if top_n is not None else config.IDENTITY_TOP_N_CROPS)
    paths = list_crop_paths(crop_dir)
    if not paths:
        return None

    scored = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        scored.append((sharp, img))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    scored = scored[:top_n]

    app = get_face_app()
    embs: List[np.ndarray] = []

    if app:
        for _, img in scored:
            e = _arcface_from_crop(img, app)
            if e is not None and e.shape[0] >= 128:
                embs.append(_l2(e))
        if not embs:
            # All ArcFace failed — consistent hist fallback for this tracklet only
            logger.debug("ArcFace failed on all crops in %s; hist fallback", crop_dir)
            for _, img in scored:
                embs.append(_hist_embedding(img))
    else:
        for _, img in scored:
            embs.append(_hist_embedding(img))

    if not embs:
        return None

    # Enforce one dimension: keep majority dim, drop mismatched (safety)
    dims = [e.shape[0] for e in embs]
    majority_dim = max(set(dims), key=dims.count)
    embs = [e for e in embs if e.shape[0] == majority_dim]
    if not embs:
        return None

    med = np.median(np.stack(embs, axis=0), axis=0).astype(np.float32)
    return _l2(med)


def _align_embedding_dict(embeddings: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """
    Ensure all tracklet embeddings share one dimension before clustering.

    Prefer 512-d (ArcFace) cluster; tracklets stuck on 256-d hist get unique
    identities instead of being dotted against ArcFace centers.
    """
    if not embeddings:
        return {}
    by_dim: Dict[int, Dict[int, np.ndarray]] = {}
    for tid, emb in embeddings.items():
        by_dim.setdefault(int(emb.shape[0]), {})[tid] = emb

    if len(by_dim) == 1:
        return embeddings

    # Prefer ArcFace-sized pool
    preferred = _ARCFACE_DIM if _ARCFACE_DIM in by_dim else max(by_dim.keys())
    logger.warning(
        "Mixed embedding dims %s — clustering on dim=%d only; "
        "other tracklets get unique identity ids",
        sorted(by_dim.keys()),
        preferred,
    )
    return by_dim[preferred]


def cluster_identities(
    tracklets: List[Dict],
    *,
    sim_thresh: float | None = None,
) -> Dict[int, int]:
    """
    Greedy threshold clustering on cosine similarity.
    Returns map tracklet_id -> identity_id.
    """
    sim_thresh = float(
        sim_thresh if sim_thresh is not None else config.IDENTITY_SIM_THRESH
    )
    embeddings: Dict[int, np.ndarray] = {}
    for t in tracklets:
        tid = int(t["tracklet_id"])
        emb = embed_crop_dir(Path(t["crop_dir"]))
        if emb is not None:
            embeddings[tid] = emb

    clusterable = _align_embedding_dict(embeddings)

    centers: List[np.ndarray] = []
    assign: Dict[int, int] = {}
    for tid, emb in clusterable.items():
        best_i, best_s = -1, -1.0
        for i, c in enumerate(centers):
            if c.shape != emb.shape:
                continue
            s = float(np.dot(emb, c))
            if s > best_s:
                best_s, best_i = s, i
        if best_s >= sim_thresh and best_i >= 0:
            assign[tid] = best_i
            updated = _l2(centers[best_i] + emb)
            centers[best_i] = updated
        else:
            assign[tid] = len(centers)
            centers.append(emb.copy())

    # Tracklets without a clusterable embedding (or wrong dim) → unique ids
    next_id = len(centers)
    for t in tracklets:
        tid = int(t["tracklet_id"])
        if tid not in assign:
            assign[tid] = next_id
            next_id += 1

    n_ids = len(set(assign.values()))
    logger.info(
        "identity clustering: %d tracklets → %d identities (clustered=%d)",
        len(tracklets),
        n_ids,
        len(clusterable),
    )
    return assign


def run_identity_clustering(
    tracklets: List[Dict], video_id: str, model_name: str
) -> Dict[int, int]:
    mapping = cluster_identities(tracklets)
    print(
        f"{model_name}/{video_id} — identity clustering: "
        f"{len(set(mapping.values()))} identities from {len(tracklets)} tracklets",
        flush=True,
    )
    return mapping
