"""
Shared face crop — the ONLY face-cropping API for all Layer 2 metrics.

Uses S3FD boxes + SyncNet-style enlarged square crop so LSE / FID / CSIM / jitter
see consistent geometry. Do not implement per-metric detectors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy import signal

from pipeline import config

logger = logging.getLogger(__name__)

_S3FD = None


def get_s3fd(device: Optional[str] = None):
    global _S3FD
    if _S3FD is not None:
        return _S3FD
    import torch
    from syncnet.detectors.s3fd import S3FD

    weights = Path(config.S3FD_WEIGHTS_PATH)
    if not weights.is_file():
        raise FileNotFoundError(
            f"S3FD weights missing: {weights}\nRun: bash syncnet/download_weights.sh"
        )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            logger.warning("CUDA not available — S3FD on CPU")
    _S3FD = S3FD(device=device, weights_path=str(weights))
    return _S3FD


def detect_faces_bgr(
    image_bgr: np.ndarray,
    detector=None,
) -> np.ndarray:
    """Return Nx5 array [x1,y1,x2,y2,conf] on BGR image."""
    detector = detector or get_s3fd()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    bboxes = detector.detect_faces(
        rgb,
        conf_th=config.FACE_DETECT_CONF,
        scales=[config.FACE_DETECT_SCALE],
    )
    return bboxes if bboxes is not None else np.empty((0, 5))


def crop_face_from_bbox(
    image_bgr: np.ndarray,
    bbox: Sequence[float],
    *,
    crop_scale: float | None = None,
    out_size: int | None = None,
) -> Optional[np.ndarray]:
    """SyncNet-style enlarged square crop → out_size² BGR."""
    crop_scale = float(crop_scale if crop_scale is not None else config.FACE_CROP_SCALE)
    out_size = int(out_size if out_size is not None else config.FACE_CROP_SIZE)
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bs = max(x2 - x1, y2 - y1) / 2.0
    if bs < 1:
        return None
    cs = crop_scale
    bsi = int(bs * (1 + 2 * cs))
    padded = np.pad(
        image_bgr,
        ((bsi, bsi), (bsi, bsi), (0, 0)),
        "constant",
        constant_values=(110, 110),
    )
    my, mx = cy + bsi, cx + bsi
    face = padded[
        int(my - bs) : int(my + bs * (1 + 2 * cs)),
        int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs)),
    ]
    if face.size == 0:
        return None
    return cv2.resize(face, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def smooth_centers(bboxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return smoothed (x, y, s) sequences for track bboxes [T,4]."""
    xs, ys, ss = [], [], []
    for det in bboxes:
        ss.append(max((det[3] - det[1]), (det[2] - det[0])) / 2.0)
        ys.append((det[1] + det[3]) / 2.0)
        xs.append((det[0] + det[2]) / 2.0)
    k = 13 if len(ss) >= 13 else (len(ss) // 2 * 2 + 1)
    if k >= 3:
        ss = signal.medfilt(ss, kernel_size=k).tolist()
        xs = signal.medfilt(xs, kernel_size=k).tolist()
        ys = signal.medfilt(ys, kernel_size=k).tolist()
    return np.asarray(xs), np.asarray(ys), np.asarray(ss)


def write_track_crops(
    frames_dir: Path,
    frame_idxs: Sequence[int],
    bboxes: np.ndarray,
    out_dir: Path,
) -> int:
    """Write smoothed SyncNet crops for a tracklet. Returns count written."""
    frames_dir = Path(frames_dir)
    flist = sorted(frames_dir.glob("*.jpg"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xs, ys, ss = smooth_centers(np.asarray(bboxes, dtype=np.float64))
    cs = config.FACE_CROP_SCALE
    out_size = config.FACE_CROP_SIZE
    written = 0
    for i, fidx in enumerate(frame_idxs):
        fidx = int(fidx)
        if fidx < 0 or fidx >= len(flist):
            continue
        image = cv2.imread(str(flist[fidx]))
        if image is None:
            continue
        bs = float(ss[i])
        bsi = int(bs * (1 + 2 * cs))
        padded = np.pad(
            image,
            ((bsi, bsi), (bsi, bsi), (0, 0)),
            "constant",
            constant_values=(110, 110),
        )
        my, mx = float(ys[i]) + bsi, float(xs[i]) + bsi
        face = padded[
            int(my - bs) : int(my + bs * (1 + 2 * cs)),
            int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs)),
        ]
        if face.size == 0:
            continue
        face = cv2.resize(face, (out_size, out_size))
        cv2.imwrite(str(out_dir / f"{written + 1:06d}.jpg"), face)
        written += 1
    return written


def list_crop_paths(crop_dir: Path) -> List[Path]:
    return sorted(Path(crop_dir).glob("*.jpg"))


def lower_face_roi(
    face_bgr: np.ndarray,
    *,
    lower_frac: float | None = None,
    out_size: int | None = None,
) -> np.ndarray:
    """Crop bottom portion of a face crop as mouth / lower-face ROI."""
    lower_frac = float(lower_frac if lower_frac is not None else config.MOUTH_LOWER_FRAC)
    out_size = int(out_size if out_size is not None else config.MOUTH_OUT_SIZE)
    h, w = face_bgr.shape[:2]
    y0 = int(h * (1.0 - lower_frac))
    roi = face_bgr[y0:h, :]
    if roi.size == 0:
        roi = face_bgr
    return cv2.resize(roi, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def mouth_crop_dir_for(face_crop_dir: Path) -> Path:
    """Sibling directory for mouth crops: .../tracklet_0001_crops → .../tracklet_0001_mouth."""
    face_crop_dir = Path(face_crop_dir)
    name = face_crop_dir.name
    if name.endswith("_crops"):
        mouth_name = name[: -len("_crops")] + "_mouth"
    else:
        mouth_name = name + "_mouth"
    return face_crop_dir.parent / mouth_name


def ensure_mouth_crops(
    face_crop_dir: Path,
    *,
    force: bool = False,
) -> Path:
    """Materialize mouth/lower-face JPGs from face crops. Returns mouth dir."""
    face_crop_dir = Path(face_crop_dir)
    mouth_dir = mouth_crop_dir_for(face_crop_dir)
    face_paths = list_crop_paths(face_crop_dir)
    if not face_paths:
        raise RuntimeError(f"no face crops in {face_crop_dir}")
    if (
        not force
        and mouth_dir.is_dir()
        and len(list_crop_paths(mouth_dir)) >= len(face_paths)
    ):
        return mouth_dir
    mouth_dir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(face_paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        mouth = lower_face_roi(img)
        cv2.imwrite(str(mouth_dir / f"{i + 1:06d}.jpg"), mouth)
    if not list_crop_paths(mouth_dir):
        raise RuntimeError(f"failed to write mouth crops for {face_crop_dir}")
    return mouth_dir


def load_mouth_clip_tensor(
    face_crop_dir: Path,
    *,
    num_frames: int | None = None,
    size: int | None = None,
) -> np.ndarray:
    """
    Load a (T, H, W, 3) uint8 mouth-crop clip for FVD.
    Samples a centered window of num_frames.
    """
    num_frames = int(num_frames if num_frames is not None else config.FVD_NUM_FRAMES)
    size = int(size if size is not None else config.FVD_CLIP_SIZE)
    mouth_dir = ensure_mouth_crops(face_crop_dir)
    paths = list_crop_paths(mouth_dir)
    if len(paths) < config.FVD_MIN_FRAMES:
        raise RuntimeError(f"too few mouth frames ({len(paths)}) for FVD")
    if len(paths) > num_frames:
        start = (len(paths) - num_frames) // 2
        paths = paths[start : start + num_frames]
    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(img)
    if len(frames) < config.FVD_MIN_FRAMES:
        raise RuntimeError(f"insufficient readable mouth frames ({len(frames)})")
    # Pad by repeating last frame if short of exact num_frames
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return np.stack(frames[:num_frames], axis=0)
