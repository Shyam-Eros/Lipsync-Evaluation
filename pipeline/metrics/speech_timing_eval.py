"""Audio vs visual speech timing stress (reference-free)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from pipeline import config
from pipeline.metrics.mouth_geometry import mar_series

logger = logging.getLogger(__name__)


def _audio_active_duration(wav_path: Path) -> float:
    sr, audio = wavfile.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32)
    if audio.max() > 1.5:
        audio = audio / 32768.0
    frame = int(sr * config.VAD_FRAME_MS / 1000)
    frame = max(frame, 160)
    thresh = 10 ** (config.VAD_ENERGY_FALLBACK_DB / 20.0)
    active = 0
    total = 0
    for i in range(0, len(audio) - frame, frame):
        rms = float(np.sqrt(np.mean(audio[i : i + frame] ** 2)))
        total += frame
        if rms >= thresh:
            active += frame
    return active / sr if sr else 0.0


def _visual_active_duration(crop_dir: Path, fps: float) -> float:
    mars = mar_series(crop_dir)
    if len(mars) < 4:
        return 0.0
    med = float(np.median(mars))
    active_frames = sum(1 for m in mars if m > med * 1.05)
    return active_frames / fps


def evaluate_segment(segment: dict) -> dict:
    out = {
        "av_timing_ratio": "",
        "timing_stress_flag": "",
        "timing_notes": "",
    }
    if segment.get("segment_type") != "speaking":
        out["timing_notes"] = "skipped_non_speaking"
        return out
    audio = segment.get("audio_slice_path")
    crop = segment.get("face_crop_dir")
    if not audio or not crop:
        out["timing_notes"] = "missing_paths"
        return out
    try:
        ad = _audio_active_duration(Path(audio))
        vd = _visual_active_duration(Path(crop), float(config.FRAME_EXTRACTION_FPS))
        if vd < 1e-3:
            out["timing_notes"] = "no_visual_speech"
            return out
        ratio = ad / vd
        out["av_timing_ratio"] = f"{ratio:.6f}"
        lo = float(getattr(config, "AV_TIMING_RATIO_MIN", 0.55))
        hi = float(getattr(config, "AV_TIMING_RATIO_MAX", 1.85))
        flag = 1 if ratio < lo or ratio > hi else 0
        out["timing_stress_flag"] = str(flag)
    except Exception as exc:
        out["timing_notes"] = f"timing_failed:{exc}"
    return out
