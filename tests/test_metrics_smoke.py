"""Lightweight smoke tests for metric helpers (no full pipeline)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.metrics.mouth_geometry import mar_at_crop_index
from pipeline.metrics.sync_eval import apply_calibration


class MetricsSmokeTest(unittest.TestCase):
    def test_apply_calibration_empty_without_json(self):
        out = apply_calibration(1.0, 5.0, 2.0, calib={})
        self.assertEqual(out["lse_d_z"], "")
        self.assertEqual(out["lse_c_z"], "")

    def test_apply_calibration_z_scores(self):
        calib = {
            "lse_d_mean": 2.0,
            "lse_d_std": 1.0,
            "lse_c_mean": 4.0,
            "lse_c_std": 2.0,
            "offset_median": 1.0,
        }
        out = apply_calibration(3.0, 6.0, 2.5, calib=calib)
        self.assertAlmostEqual(float(out["lse_d_z"]), 1.0)
        self.assertAlmostEqual(float(out["lse_c_z"]), 1.0)
        self.assertAlmostEqual(float(out["lse_offset_corrected"]), 1.5)

    def test_mar_on_missing_crop_dir(self):
        self.assertIsNone(mar_at_crop_index(Path("/nonexistent/crops"), 0))

    def test_seam_eval_skips_non_speaking(self):
        from pipeline.metrics.seam_eval import evaluate_segment

        r = evaluate_segment({"segment_type": "non_speaking_face", "face_crop_dir": ""})
        self.assertTrue(r.get("seam_score") == "" or r.get("seam_notes"))


if __name__ == "__main__":
    unittest.main()
