# SyncNet evaluation assets (vendored from joonson/syncnet_python)

## Weights (required before running evaluation)

From this directory:

```bash
bash download_weights.sh
```

This places:

| File | Path |
|------|------|
| SyncNet v2 | `data/syncnet_v2.model` |
| S3FD face detector | `detectors/s3fd/weights/sfd_face.pth` |

Source URLs (VGG / Oxford):

- http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model
- https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth

Point `SYNCNET_WEIGHTS_PATH` and `S3FD_WEIGHTS_PATH` in `../config.py` at these files (defaults already match).

## Code origin

- `SyncNetModel.py` — joonson/syncnet_python
- `detectors/s3fd/` — joonson/syncnet_python S3FD detector
- Scoring (LSE-D / LSE-C) — Wav2Lip `SyncNetInstance_calc_scores` protocol, reimplemented in `../sync_eval.py`
