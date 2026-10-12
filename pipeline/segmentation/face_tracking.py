"""Layer 1 stage 2 — per-shot S3FD detection + IoU face tracklets (no cross-cut)."""

from __future__ import annotations

import logging
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.interpolate import interp1d

from pipeline import config
from pipeline.io_utils import append_csv
from pipeline.metrics.shared_face_crop import detect_faces_bgr, get_s3fd, write_track_crops

logger = logging.getLogger(__name__)

TRACKLET_FIELDS = [
    "tracklet_id",
    "shot_id",
    "video_id",
    "model_name",
    "start_frame",
    "end_frame",
    "n_frames",
    "mean_face_area_ratio",
    "bbox_pickle",
    "crop_dir",
    "has_face",
]


def bb_iou(a, b) -> float:
    x_a, y_a = max(a[0], b[0]), max(a[1], b[1])
    x_b, y_b = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x_b - x_a) * max(0, y_b - y_a)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    denom = float(aa + bb - inter)
    return inter / denom if denom > 0 else 0.0


def detect_all_frames(frames_dir: Path, detector=None) -> List[List[Dict[str, Any]]]:
    detector = detector or get_s3fd()
    dets: List[List[Dict[str, Any]]] = []
    for fidx, fname in enumerate(sorted(Path(frames_dir).glob("*.jpg"))):
        image = cv2.imread(str(fname))
        frame_dets: List[Dict[str, Any]] = []
        if image is not None:
            bboxes = detect_faces_bgr(image, detector=detector)
            for bbox in bboxes:
                frame_dets.append(
                    {
                        "frame": fidx,
                        "bbox": bbox[:-1].tolist(),
                        "conf": float(bbox[-1]),
                    }
                )
        dets.append(frame_dets)
    return dets


def track_shot(scenefaces: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """IoU-link faces within one shot (joonson-style)."""
    faces = deepcopy(scenefaces)
    min_track = int(config.MIN_TRACK_FRAMES)
    num_failed = int(config.NUM_FAILED_DET)
    min_face = float(config.MIN_FACE_SIZE)
    iou_th = float(config.IOU_TRACK_THRESH)
    tracks: List[Dict[str, Any]] = []

    while True:
        track: List[Dict[str, Any]] = []
        for framefaces in faces:
            for face in list(framefaces):
                if not track:
                    track.append(face)
                    framefaces.remove(face)
                elif face["frame"] - track[-1]["frame"] <= num_failed:
                    if bb_iou(face["bbox"], track[-1]["bbox"]) > iou_th:
                        track.append(face)
                        framefaces.remove(face)
                        continue
                else:
                    break
        if not track:
            break
        if len(track) <= min_track:
            continue
        framenum = np.array([f["frame"] for f in track])
        bboxes = np.array([np.array(f["bbox"]) for f in track])
        frame_i = np.arange(framenum[0], framenum[-1] + 1)
        bboxes_i = np.stack(
            [interp1d(framenum, bboxes[:, ij])(frame_i) for ij in range(4)], axis=1
        )
        mean_w = float(np.mean(bboxes_i[:, 2] - bboxes_i[:, 0]))
        mean_h = float(np.mean(bboxes_i[:, 3] - bboxes_i[:, 1]))
        if max(mean_w, mean_h) > min_face:
            tracks.append({"frame": frame_i, "bbox": bboxes_i})
    return tracks


def classify_shot_scale(mean_area_ratio: float) -> str:
    if mean_area_ratio >= config.SHOT_SCALE_CLOSEUP:
        return "close-up"
    if mean_area_ratio >= config.SHOT_SCALE_MEDIUM:
        return "medium"
    return "wide"


def run_face_tracking(
    frames_dir: Path,
    shots: List[Dict],
    work_dir: Path,
    *,
    video_id: str,
    model_name: str,
    frame_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[List[Dict], Dict[int, str]]:
    """
    Build tracklets per shot. Returns (tracklet_rows, shot_id->shot_scale).
    Shots with no faces are noted via empty tracklets; caller marks no_face.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    detector = get_s3fd()
    all_dets = detect_all_frames(frames_dir, detector=detector)
    with open(work_dir / "faces.pckl", "wb") as f:
        pickle.dump(all_dets, f)

    if frame_hw is None:
        sample = next(iter(sorted(Path(frames_dir).glob("*.jpg"))), None)
        if sample is None:
            raise RuntimeError("no frames")
        im = cv2.imread(str(sample))
        fh, fw = im.shape[:2]
    else:
        fh, fw = frame_hw
    frame_area = float(fh * fw)

    tracklet_id = 0
    rows: List[Dict] = []
    shot_scales: Dict[int, str] = {}
    shot_face_ratios: Dict[int, List[float]] = {int(s["shot_id"]): [] for s in shots}

    for shot in shots:
        sid = int(shot["shot_id"])
        start, end = int(shot["start_frame"]), int(shot["end_frame"])
        start, end = max(0, start), min(len(all_dets), end)
        scenefaces = [list(all_dets[i]) for i in range(start, end)]
        any_face = any(len(x) > 0 for x in scenefaces)
        if not any_face:
            shot_scales[sid] = "wide"
            logger.info("shot %d — no faces (valid)", sid)
            continue

        tracks = track_shot(scenefaces) if (end - start) > config.MIN_TRACK_FRAMES else []
        if not tracks and any_face:
            # Still record that faces existed but no stable tracklet
            shot_scales[sid] = "medium"
            continue

        for t in tracks:
            frames = np.asarray(t["frame"], dtype=np.int64)
            bboxes = np.asarray(t["bbox"], dtype=np.float64)
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            mean_ratio = float(np.mean(areas) / frame_area) if frame_area else 0.0
            shot_face_ratios[sid].append(mean_ratio)

            bbox_path = work_dir / f"tracklet_{tracklet_id:04d}_bbox.pkl"
            with open(bbox_path, "wb") as f:
                pickle.dump({"frame": frames, "bbox": bboxes}, f)
            crop_dir = work_dir / f"tracklet_{tracklet_id:04d}_crops"
            n_crops = write_track_crops(frames_dir, frames, bboxes, crop_dir)

            row = {
                "tracklet_id": tracklet_id,
                "shot_id": sid,
                "video_id": video_id,
                "model_name": model_name,
                "start_frame": int(frames[0]),
                "end_frame": int(frames[-1]),
                "n_frames": int(len(frames)),
                "mean_face_area_ratio": f"{mean_ratio:.6f}",
                "bbox_pickle": str(bbox_path),
                "crop_dir": str(crop_dir),
                "has_face": 1 if n_crops > 0 else 0,
            }
            rows.append(row)
            append_csv(config.TRACKLETS_CSV, [row], TRACKLET_FIELDS)
            tracklet_id += 1

    for sid, ratios in shot_face_ratios.items():
        if ratios:
            shot_scales[sid] = classify_shot_scale(float(np.mean(ratios)))
        elif sid not in shot_scales:
            shot_scales[sid] = "wide"

    print(
        f"{model_name}/{video_id} — face tracking: {len(rows)} tracklets",
        flush=True,
    )
    return rows, shot_scales
