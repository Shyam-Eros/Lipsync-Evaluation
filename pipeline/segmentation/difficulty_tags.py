"""Auto difficulty tags for stress-test slicing (reference-free)."""

from __future__ import annotations

import csv
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import cv2
import numpy as np
from scipy.io import wavfile

from pipeline import config

logger = logging.getLogger(__name__)


def _shot_scale_tag(scale: str) -> str:
    if scale == "close-up":
        return "close_up"
    if scale == "medium":
        return "medium"
    if scale == "wide":
        return "wide"
    return scale.replace("-", "_") if scale else "unknown"


def _load_manifest() -> Dict[str, Set[str]]:
    path = getattr(config, "DIFFICULTY_MANIFEST_CSV", None)
    out: Dict[str, Set[str]] = {}
    if not path or not Path(path).is_file():
        return out
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = row.get("video_id", "").strip()
            tags = row.get("tags", "").strip()
            if vid and tags:
                out.setdefault(vid, set()).update(t.strip() for t in tags.split(",") if t.strip())
    return out


def _tracklet_motion_and_light(
    bbox_pickle: Path,
    frames_dir: Path,
    sample_n: int = 40,
) -> tuple[float, float, float]:
    """mean_luma, bbox_center_std, bbox_scale_std."""
    with open(bbox_pickle, "rb") as f:
        data = pickle.load(f)
    frames = np.asarray(data["frame"], dtype=np.int64)
    bboxes = np.asarray(data["bbox"], dtype=np.float64)
    if len(frames) == 0:
        return 128.0, 0.0, 0.0
    idxs = np.linspace(0, len(frames) - 1, min(sample_n, len(frames))).astype(int)
    lumas = []
    cx, cy, sc = [], [], []
    flist = sorted(Path(frames_dir).glob("*.jpg"))
    for i in idxs:
        fi = int(frames[i])
        if fi < 0 or fi >= len(flist):
            continue
        img = cv2.imread(str(flist[fi]))
        if img is None:
            continue
        x1, y1, x2, y2 = bboxes[i]
        lumas.append(float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))))
        cx.append((x1 + x2) / 2.0)
        cy.append((y1 + y2) / 2.0)
        sc.append(max(x2 - x1, y2 - y1))
    if not lumas:
        return 128.0, 0.0, 0.0
    return float(np.mean(lumas)), float(np.std(cx) + np.std(cy)), float(np.std(sc))


def _speech_rate_proxy(wav_path: Path, seg_start: float, seg_end: float) -> float:
    try:
        sr, audio = wavfile.read(str(wav_path))
        if audio.ndim > 1:
            audio = audio[:, 0]
        a0 = int(max(0, seg_start) * sr)
        a1 = int(min(len(audio), seg_end * sr))
        chunk = audio[a0:a1].astype(np.float32)
        if len(chunk) < sr // 10:
            return 0.0
        frame = int(sr * 0.02)
        peaks = 0
        prev = 0.0
        for i in range(0, len(chunk) - frame, frame):
            rms = float(np.sqrt(np.mean(chunk[i : i + frame] ** 2)))
            if rms > prev * 1.4 and rms > 1e-3:
                peaks += 1
            prev = rms
        dur = max(0.1, (a1 - a0) / sr)
        return peaks / dur
    except Exception:
        return 0.0


def tag_tracklet(
    tracklet: dict,
    shot_scale: str,
    frames_dir: Path,
    full_wav: Optional[Path],
    manifest_extra: Set[str],
) -> str:
    tags: Set[str] = {_shot_scale_tag(shot_scale)}
    tags.update(manifest_extra)
    bbox_pkl = tracklet.get("bbox_pickle")
    if bbox_pkl and Path(bbox_pkl).is_file():
        luma, motion, scale_std = _tracklet_motion_and_light(Path(bbox_pkl), frames_dir)
        if luma < float(getattr(config, "LOW_LIGHT_LUMA_MAX", 70.0)):
            tags.add("low_light")
        if motion > float(getattr(config, "FAST_MOTION_STD_MIN", 25.0)):
            tags.add("fast_motion")
        if scale_std > float(getattr(config, "FAST_MOTION_SCALE_STD_MIN", 18.0)):
            tags.add("fast_motion")
        try:
            with open(bbox_pkl, "rb") as f:
                data = pickle.load(f)
            bboxes = np.asarray(data["bbox"])
            w = bboxes[:, 2] - bboxes[:, 0]
            h = bboxes[:, 3] - bboxes[:, 1]
            aspect = float(np.median(w / (h + 1e-6)))
            if aspect < 0.72 or aspect > 1.35:
                tags.add("profile")
        except Exception:
            pass
    if full_wav and full_wav.is_file():
        t0 = float(tracklet.get("start_frame", 0)) / config.FRAME_EXTRACTION_FPS
        t1 = (float(tracklet.get("end_frame", 0)) + 1) / config.FRAME_EXTRACTION_FPS
        rate = _speech_rate_proxy(full_wav, t0, t1)
        if rate > float(getattr(config, "HIGH_SPEECH_RATE_MIN", 8.0)):
            tags.add("high_speech_rate")
    return ",".join(sorted(tags))


def enrich_segments(
    segments: List[dict],
    tracklets: Sequence[dict],
    frames_dir: Path,
    full_wav: Path,
    video_id: str,
) -> List[dict]:
    manifest = _load_manifest()
    extra = manifest.get(video_id, set())
    tid_to_track = {int(t["tracklet_id"]): t for t in tracklets}
    shot_scale_by_tid = {}
    for s in segments:
        shot_scale_by_tid[int(s.get("tracklet_id", -1))] = s.get("shot_scale", "medium")

    for seg in segments:
        tid = int(seg.get("tracklet_id", -1))
        tr = tid_to_track.get(tid)
        scale = seg.get("shot_scale") or shot_scale_by_tid.get(tid, "medium")
        if tr:
            seg["difficulty_tags"] = tag_tracklet(tr, scale, frames_dir, full_wav, extra)
        else:
            base = {_shot_scale_tag(str(scale))}
            base.update(extra)
            seg["difficulty_tags"] = ",".join(sorted(base))
    return segments
