# Reference-Free Dubbed-Movie Evaluation Workflow

**Project:** `/local-workspace/gowtham.s/lip-sync_inference/Evalution`  
**Venv:** `.venv`  
**Entry:** `python run_pipeline.py`  
**Calibration:** `python calibrate_syncnet_tamil.py`

---

## Why no ground-truth video

Dubbing: Hollywood video + **Tamil audio** → lip-sync → output.

Original lips match **English**. Comparing frames to that original would punish correct Tamil mouth shapes. This pipeline is **reference-free**:

| Metric | Reference |
|--------|-----------|
| LSE-D / LSE-C | Output mouth vs **its own** Tamil audio |
| Calibrated LSE | Z-score vs SyncNet scores on **real Tamil clips** |
| Mouth FID | Non-talking / Tamil real **mouth** crops |
| FVD | I3D features vs Tamil real mouth clip windows |
| CSIM / csim_late | Same actor’s face vs **their own** best embedding in the clip |
| Jitter / warping | Frame-to-frame stability within a shot |

---

## Architecture

```text
Tamil real clips ──► calibrate_syncnet_tamil.py ──► tamil_syncnet_calibration.json
                                                      + tamil_mouth_ref/

inferred model mp4
  Layer 1: shots → face tracklets → identity clusters → active speaker
           → results/segments.csv
  Layer 1b: difficulty_tags on segments (shot scale, motion, light, speech rate, …)
  Layer 2 Tier 1: LSE(+calib) | mouth FID | I3D FVD | CSIM(+late) | jitter+warp
  Layer 2 Tier 2: seam | oral artifacts | Whisper bilabial | upper-face retention | AV timing
  Post: qc_flags.csv + metrics_by_shot_scale + metrics_by_difficulty + human_eval pack
           → metrics_raw.csv
```

---

## 1. Activate

```bash
cd /local-workspace/gowtham.s/lip-sync_inference/Evalution
source .venv/bin/activate
pip install -r requirements.txt   # if needed
bash syncnet/download_weights.sh  # SyncNet + S3FD once
```

---

## 2. Tamil SyncNet calibration

```bash
python calibrate_syncnet_tamil.py --data-root /YOUR/tamil/clip/dataset
# optional: --max-clips 50 --mouth-ref-dir results/tamil_mouth_ref
```

Set `TAMIL_CALIB_DATA_ROOT` in `pipeline/config.py` to skip repeating `--data-root`.

Without calibration, raw `lse_d` / `lse_c` / `lse_offset` still run; calibrated columns stay empty and FVD falls back to a temporal proxy unless `fvd_real_dirs` exist.

---

## 3. Paths for inferred videos

**CLI (preferred):**

```bash
python run_pipeline.py \
  --video /path/to/out.mp4 --model my_model \
  --video-id clip_01
```

**Or** folder mode / `EVAL_ITEMS` in `pipeline/config.py` (any model key allowed via CLI).

Optional: `REAL_MOUTH_REF_DIR` / `REAL_FACE_REF_DIR` for curated refs (never English-synced mouths).

---

## 4. Run

```bash
python run_pipeline.py --layer1-only
CUDA_VISIBLE_DEVICES=0 python run_pipeline.py --video out.mp4 --model xdub
```

---

## 5. Interpreting scores

- **LSE-D ↓ / LSE-C ↑** — sync to Tamil audio; compare `lse_*_z` after calibration (≈0 = like real Tamil)
- **lse_offset_corrected** — AV lag after subtracting Tamil median SyncNet bias
- **FID ↓** — mouth/lower-face realism
- **FVD ↓** — mouth motion realism (I3D)
- **CSIM ↑ / csim_late ↑** — identity; watch late drop on long clips
- **jitter ↓ / warping_error ↓** — flicker and unstable blending
- Compare models on the **same `video_id`**; check **shot_scale** breakdown

---

## Quick checklist

1. `source .venv/bin/activate`
2. Weights downloaded
3. `calibrate_syncnet_tamil.py --data-root …` (once per Tamil corpus)
4. `python run_pipeline.py --video … --model …`
5. Compare models in console summary + `results/metrics_raw.csv`
