"""LSE-D / LSE-C — SyncNet vs the segment's OWN dubbed audio (never another video).

Optionally applies Tamil-clip calibration stats from
results/tamil_syncnet_calibration.json (see calibrate_syncnet_tamil.py).
"""

from __future__ import annotations

import glob
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import python_speech_features
import torch
from scipy.io import wavfile

from pipeline import config

logger = logging.getLogger(__name__)

_INSTANCE = None
_DEVICE = None
_CALIB: Optional[Dict[str, Any]] = None
_CALIB_LOADED = False


def calc_pdist(feat1, feat2, vshift=10):
    win_size = vshift * 2 + 1
    feat2p = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))
    dists = []
    for i in range(len(feat1)):
        dists.append(
            torch.nn.functional.pairwise_distance(
                feat1[[i], :].repeat(win_size, 1),
                feat2p[i : i + win_size, :],
            )
        )
    return dists


def _device() -> str:
    global _DEVICE
    if _DEVICE is None:
        if torch.cuda.is_available():
            _DEVICE = "cuda"
        else:
            _DEVICE = "cpu"
            print("WARNING: CUDA not available — SyncNet on CPU", flush=True)
    return _DEVICE


def load_syncnet(weights_path: Optional[Path] = None):
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE, _device()
    from syncnet.SyncNetModel import S

    weights_path = Path(weights_path or config.SYNCNET_WEIGHTS_PATH)
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"SyncNet weights missing: {weights_path}\n"
            "Run: bash syncnet/download_weights.sh"
        )
    device = _device()
    model = S(num_layers_in_fc_layers=1024).to(device)
    try:
        state = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(str(weights_path), map_location="cpu")
    if isinstance(state, torch.nn.Module):
        state = state.state_dict()
    model.load_state_dict(state)
    model.eval()
    _INSTANCE = model
    return _INSTANCE, device


def load_tamil_calibration(
    path: Optional[Path] = None,
    *,
    force: bool = False,
) -> Optional[Dict[str, Any]]:
    """Load Tamil SyncNet calibration JSON (cached)."""
    global _CALIB, _CALIB_LOADED
    if _CALIB_LOADED and not force:
        return _CALIB
    cal_path = Path(path or config.TAMIL_CALIBRATION_PATH)
    _CALIB_LOADED = True
    if not cal_path.is_file():
        logger.info("No Tamil SyncNet calibration at %s — raw LSE only", cal_path)
        _CALIB = None
        return None
    with cal_path.open(encoding="utf-8") as f:
        _CALIB = json.load(f)
    logger.info("Loaded Tamil SyncNet calibration from %s", cal_path)
    return _CALIB


def apply_calibration(
    lse_d: float,
    lse_c: float,
    offset: float,
    calib: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Return calibrated column strings; empty if no calib."""
    out = {"lse_d_z": "", "lse_c_z": "", "lse_offset_corrected": ""}
    if calib is None:
        calib = load_tamil_calibration()
    if not calib:
        return out
    mean_d = float(calib.get("lse_d_mean", 0.0))
    std_d = float(calib.get("lse_d_std", 1.0)) or 1.0
    mean_c = float(calib.get("lse_c_mean", 0.0))
    std_c = float(calib.get("lse_c_std", 1.0)) or 1.0
    offset_bias = float(calib.get("offset_median", 0.0))
    out["lse_d_z"] = f"{(lse_d - mean_d) / std_d:.6f}"
    out["lse_c_z"] = f"{(lse_c - mean_c) / std_c:.6f}"
    out["lse_offset_corrected"] = f"{offset - offset_bias:.3f}"
    return out


def _load_crops(crop_dir: Path):
    flist = sorted(glob.glob(str(Path(crop_dir) / "*.jpg")))
    images = []
    for f in flist:
        img = cv2.imread(f)
        if img is None:
            continue
        if img.shape[0] != config.FACE_CROP_SIZE:
            img = cv2.resize(img, (config.FACE_CROP_SIZE, config.FACE_CROP_SIZE))
        images.append(img)
    if len(images) < 6:
        raise RuntimeError(f"Need ≥6 crops, got {len(images)}")
    im = np.stack(images, axis=3)
    im = np.expand_dims(im, 0)
    im = np.transpose(im, (0, 3, 4, 1, 2))
    return torch.from_numpy(im.astype(np.float32)), len(images)


@torch.no_grad()
def evaluate_sync_track(
    crop_dir: Path,
    audio_path: Path,
) -> Tuple[float, float, float]:
    """(lse_d, lse_c, offset) — audio must be THIS segment's driving dub track."""
    model, device = load_syncnet()
    imtv, n_images = _load_crops(Path(crop_dir))
    sr, audio = wavfile.read(str(audio_path))
    if sr != config.AUDIO_SAMPLE_RATE:
        raise RuntimeError(f"Expected {config.AUDIO_SAMPLE_RATE} Hz, got {sr}")
    mfcc = list(zip(*python_speech_features.mfcc(audio, sr)))
    mfcc = np.stack([np.array(i) for i in mfcc])
    cct = torch.from_numpy(
        np.expand_dims(np.expand_dims(mfcc, 0), 0).astype(np.float32)
    )

    min_length = min(n_images, math.floor(len(audio) / 640))
    lastframe = min_length - 5
    if lastframe <= 0:
        raise RuntimeError("segment too short for SyncNet")

    vshift = int(config.SYNCNET_OFFSET_RANGE)
    bs = int(config.SYNCNET_BATCH_SIZE)
    im_feat, cc_feat = [], []
    for i in range(0, lastframe, bs):
        im_batch = [
            imtv[:, :, v : v + 5, :, :]
            for v in range(i, min(lastframe, i + bs))
        ]
        im_feat.append(model.forward_lip(torch.cat(im_batch, 0).to(device)).cpu())
        cc_batch = [
            cct[:, :, :, v * 4 : v * 4 + 20]
            for v in range(i, min(lastframe, i + bs))
        ]
        cc_feat.append(model.forward_aud(torch.cat(cc_batch, 0).to(device)).cpu())

    im_feat = torch.cat(im_feat, 0)
    cc_feat = torch.cat(cc_feat, 0)
    dists = calc_pdist(im_feat, cc_feat, vshift=vshift)
    mdist = torch.mean(torch.stack(dists, 1), 1)
    minval, minidx = torch.min(mdist, 0)
    offset = float(vshift - int(minidx.item()))
    lse_d = float(minval.item())
    lse_c = float((torch.median(mdist) - minval).item())
    return lse_d, lse_c, offset


def evaluate_segment(segment: dict) -> dict:
    """Layer 2 entry: speaking segments only."""
    empty = {
        "lse_d": "",
        "lse_c": "",
        "lse_offset": "",
        "lse_d_z": "",
        "lse_c_z": "",
        "lse_offset_corrected": "",
        "sync_notes": "skipped_non_speaking",
    }
    if segment.get("segment_type") != "speaking":
        return empty
    crop = segment.get("face_crop_dir")
    audio = segment.get("audio_slice_path")
    if not crop or not audio:
        empty["sync_notes"] = "missing_paths"
        return empty
    lse_d, lse_c, offset = evaluate_sync_track(Path(crop), Path(audio))
    out = {
        "lse_d": f"{lse_d:.6f}",
        "lse_c": f"{lse_c:.6f}",
        "lse_offset": f"{offset:.3f}",
        "sync_notes": "",
    }
    out.update(apply_calibration(lse_d, lse_c, offset))
    return out
