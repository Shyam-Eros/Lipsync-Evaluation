"""
I3D Fréchet Video Distance (true FVD).

Adapted from LatentSync eval/fvd.py (universome/fvd-comparison).
Weights: HuggingFace ByteDance/LatentSync-1.5 auxiliary/i3d_torchscript.pt
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg
import torch

from pipeline import config

logger = logging.getLogger(__name__)

_I3D = None
_I3D_DEVICE = None


def ensure_i3d_weights(path: Optional[Path] = None) -> Path:
    path = Path(path or config.I3D_WEIGHTS_PATH)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    # huggingface-cli download ByteDance/LatentSync-1.5 auxiliary/i3d_torchscript.pt --local-dir <models>
    local_dir = path.parent
    rel = config.I3D_HF_FILENAME  # auxiliary/i3d_torchscript.pt
    cmd = [
        "huggingface-cli",
        "download",
        config.I3D_HF_REPO,
        rel,
        "--local-dir",
        str(local_dir),
    ]
    logger.info("Downloading I3D weights: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        # Fallback: try huggingface_hub Python API
        try:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(
                repo_id=config.I3D_HF_REPO,
                filename=rel,
                local_dir=str(local_dir),
            )
            downloaded = Path(downloaded)
            if downloaded != path and downloaded.is_file():
                # May land at models/auxiliary/i3d_torchscript.pt
                if not path.is_file():
                    import shutil

                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(downloaded, path)
        except Exception as hub_exc:
            raise FileNotFoundError(
                f"I3D weights missing at {path}. Install huggingface_hub / "
                f"huggingface-cli and download {config.I3D_HF_REPO}/{rel}. "
                f"Errors: cli={exc}; hub={hub_exc}"
            ) from hub_exc

    # huggingface may nest under auxiliary/
    if not path.is_file():
        nested = local_dir / "auxiliary" / path.name
        if nested.is_file():
            import shutil

            shutil.copy2(nested, path)
    if not path.is_file():
        raise FileNotFoundError(f"I3D weights still missing after download: {path}")
    return path


def load_i3d(device: Optional[str] = None):
    global _I3D, _I3D_DEVICE
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if _I3D is not None and _I3D_DEVICE == device:
        return _I3D, device
    weights = ensure_i3d_weights()
    with open(weights, "rb") as f:
        model = torch.jit.load(f).eval().to(device)
    _I3D = model
    _I3D_DEVICE = device
    return _I3D, device


def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def frechet_distance(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)
    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)
    return float(np.real(m + np.trace(sigma_gen + sigma_real - s * 2)))


@torch.no_grad()
def embed_videos(
    videos: torch.Tensor,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    videos: (B, T, H, W, C) float in [0, 1]
    returns: (B, D) I3D features
    """
    model, device = load_i3d(device)
    # I3D torchscript expects (B, C, T, H, W)
    x = videos.permute(0, 4, 1, 2, 3).to(device)
    feats = model(x, rescale=False, resize=False, return_features=True)
    return feats.detach().cpu().numpy()


def clips_from_mouth_dirs(
    face_crop_dirs: Sequence[Path],
    *,
    max_clips: int = 64,
    windows_per_dir: int = 4,
) -> torch.Tensor:
    """Stack mouth-crop windows → (B, T, H, W, C) float [0,1].

    Long tracks contribute multiple non-overlapping windows so a single
    speaking segment can still yield a usable Frechet batch.
    """
    from pipeline.metrics.shared_face_crop import (
        ensure_mouth_crops,
        list_crop_paths,
        load_mouth_clip_tensor,
    )

    clips: List[torch.Tensor] = []
    n_frames = int(config.FVD_NUM_FRAMES)
    size = int(config.FVD_CLIP_SIZE)

    for d in face_crop_dirs:
        if len(clips) >= max_clips:
            break
        try:
            mouth_dir = ensure_mouth_crops(Path(d))
            paths = list_crop_paths(mouth_dir)
            if len(paths) < config.FVD_MIN_FRAMES:
                continue
            # How many non-overlapping windows fit?
            n_win = max(1, min(windows_per_dir, len(paths) // n_frames))
            if n_win == 1:
                arr = load_mouth_clip_tensor(Path(d))
                clips.append(torch.from_numpy(arr))
            else:
                import cv2

                for w in range(n_win):
                    if len(clips) >= max_clips:
                        break
                    start = w * n_frames
                    chunk = paths[start : start + n_frames]
                    if len(chunk) < config.FVD_MIN_FRAMES:
                        continue
                    frames = []
                    for p in chunk:
                        img = cv2.imread(str(p))
                        if img is None:
                            continue
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
                        frames.append(img)
                    while len(frames) < n_frames and frames:
                        frames.append(frames[-1])
                    if len(frames) >= config.FVD_MIN_FRAMES:
                        clips.append(torch.from_numpy(np.stack(frames[:n_frames], axis=0)))
        except Exception as exc:
            logger.debug("skip FVD clip %s: %s", d, exc)
    if len(clips) < 2:
        raise RuntimeError(f"need ≥2 mouth clips for FVD, got {len(clips)}")
    stacked = torch.stack(clips, dim=0).float() / 255.0
    return stacked


def compute_fvd_i3d(
    fake_face_dirs: Sequence[Path],
    real_face_dirs: Sequence[Path],
    *,
    device: Optional[str] = None,
) -> float:
    fake = clips_from_mouth_dirs(fake_face_dirs)
    real = clips_from_mouth_dirs(real_face_dirs)
    feats_fake = embed_videos(fake, device=device)
    feats_real = embed_videos(real, device=device)
    return frechet_distance(feats_fake, feats_real)


def load_real_mouth_dirs_from_calib() -> List[Path]:
    """Resolve real mouth crop dirs from calibration JSON / REAL_MOUTH_REF_DIR."""
    dirs: List[Path] = []
    ref = getattr(config, "REAL_MOUTH_REF_DIR", None)
    if ref and Path(ref).is_dir():
        dirs.append(Path(ref))
    calib_path = Path(config.TAMIL_CALIBRATION_PATH)
    if calib_path.is_file():
        import json

        with calib_path.open(encoding="utf-8") as f:
            calib = json.load(f)
        mref = calib.get("mouth_ref_dir")
        if mref and Path(mref).is_dir() and Path(mref) not in dirs:
            dirs.append(Path(mref))
    return dirs
