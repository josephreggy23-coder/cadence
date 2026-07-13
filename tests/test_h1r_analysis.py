"""Tests for the hierarchy-aware H1R astrocyte secondary analysis."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from analyze_h1r_astrocytes import (  # noqa: E402
    EXPECTED_SOURCES,
    FLUORESCENCE_OFFSET,
    SAMPLING_RATE_HZ,
    SCHEMA_VERSION,
    SOURCE_DOI,
    aggregate_roi_metrics_by_slice,
    analyze_ne_responses,
    build_summary,
    build_specification_grid,
    common_scale_paired_values,
    contextual_dff,
    paired_slice_values,
    pre_post_delta_auc,
    robust_response_metrics,
    validate_export,
)


class ContextualDffTests(unittest.TestCase):
    def test_default_baseline_matches_authors_window(self):
        trace = np.linspace(-50.0, 50.0, 140)
        onset = 90
        dff, f0, method = contextual_dff(trace, onset)

        start = onset - int(70 * SAMPLING_RATE_HZ)
        end = onset - int(10 * SAMPLING_RATE_HZ)
        expected_f0 = np.median(trace[start:end] + FLUORESCENCE_OFFSET)
        np.testing.assert_allclose(f0, expected_f0)
        np.testing.assert_allclose(dff, (trace + FLUORESCENCE_OFFSET - expected_f0) / expected_f0)
        self.assertEqual(method, "onset_minus_70s_to_minus_10s")

    def test_documented_30_second_fallback(self):
        trace = np.full(100, -30_000.0)
        _, f0, method = contextual_dff(trace, onset_index=35)
        self.assertEqual(method, "fallback_onset_minus_40s_to_minus_10s")
        self.assertEqual(f0, 70_000.0)

    def test_2025_notebook_uses_20_second_fallback(self):
        trace = np.full(100, -30_000.0)
        _, f0, method = contextual_dff(
            trace, onset_index=30, fallback_start_seconds=30
        )
        self.assertEqual(method, "fallback_onset_minus_30s_to_minus_10s")
        self.assertEqual(f0, 70_000.0)

    def test_too_short_baseline_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "insufficient pre-onset"):
            contextual_dff(np.ones(80), onset_index=20)


class DeltaAucTests(unittest.TestCase):
    def test_pre_post_windows_match_source_notebook(self):
        onset = 50
        samples = int(30 * SAMPLING_RATE_HZ)
        dff = np.zeros(110)
        dff[onset : onset + samples] = 0.1
        pre, post, delta, observed_samples = pre_post_delta_auc(dff, onset)

        self.assertEqual(observed_samples, 21)
        self.assertAlmostEqual(pre, 0.0)
        expected = 0.1 * (samples - 1) / SAMPLING_RATE_HZ
        self.assertAlmostEqual(post, expected)
        self.assertAlmostEqual(delta, expected)


class OrthogonalResponseTests(unittest.TestCase):
    def test_robust_z_is_affine_invariant(self):
        rng = np.random.default_rng(4)
        trace = rng.normal(100.0, 2.0, 150)
        onset = 90
        trace[onset : onset + int(30 * SAMPLING_RATE_HZ)] += 8.0
        raw, scale, robust_z = robust_response_metrics(trace, onset)
        transformed = 3.5 * trace - 8_000.0
        raw_t, scale_t, robust_z_t = robust_response_metrics(transformed, onset)
        self.assertAlmostEqual(raw_t, 3.5 * raw)
        self.assertAlmostEqual(scale_t, 3.5 * scale)
        self.assertAlmostEqual(robust_z_t, robust_z)

    def test_flat_baseline_is_rejected_for_noise_standardization(self):
        trace = np.ones(150)
        with self.assertRaisesRegex(ValueError, "baseline MAD scale"):
            robust_response_metrics(trace, onset_index=90)


class HierarchyTests(unittest.TestCase):
    def test_rois_are_aggregated_before_slice_pairing(self):
        roi_metrics = pd.DataFrame(
            [
                {"cohort": "c", "slice_id": "s1", "genotype": "WT", "delta_auc": 0.0},
                {"cohort": "c", "slice_id": "s1", "genotype": "WT", "delta_auc": 2.0},
                {"cohort": "c", "slice_id": "s1", "genotype": "KO", "delta_auc": 3.0},
                {"cohort": "c", "slice_id": "s2", "genotype": "WT", "delta_auc": 10.0},
                {"cohort": "c", "slice_id": "s2", "genotype": "KO", "delta_auc": 0.0},
                {"cohort": "c", "slice_id": "s2", "genotype": "KO", "delta_auc": 0.0},
                {"cohort": "c", "slice_id": "s2", "genotype": "KO", "delta_auc": 0.0},
                {"cohort": "c", "slice_id": "s2", "genotype": "KO", "delta_auc": 0.0},
            ]
        )
        slices = aggregate_roi_metrics_by_slice(roi_metrics)
        paired = paired_slice_values(slices, "c")

        self.assertEqual(slices.loc[(slices.slice_id == "s1") & (slices.genotype == "WT"), "n_rois"].item(), 2)
        self.assertAlmostEqual(paired.loc["s1", "WT"], 1.0)
        self.assertAlmostEqual(paired.loc["s1", "ko_minus_wt"], 2.0)
        self.assertAlmostEqual(paired.loc["s2", "ko_minus_wt"], -10.0)


def _synthetic_export() -> pd.DataFrame:
    blocks = []
    n_frames = 125
    onset = 75
    for file_id, source in EXPECTED_SOURCES.items():
        for genotype, response in (("WT", 150.0), ("KO", 300.0)):
            raw = np.full(n_frames, -30_000.0)
            raw[onset:] += response
            blocks.append(
                pd.DataFrame(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "source_doi": SOURCE_DOI,
                        "source_file_id": file_id,
                        "source_file": source["file_name"],
                        "source_sha256": source["sha256"],
                        "cohort": source["cohort"],
                        "slice_id": f"{source['cohort']}_slice1",
                        "stimulus": "NE",
                        "genotype": genotype,
                        "roi_id": f"{genotype}1",
                        "area_pixels": 100.0,
                        "onset_index": onset,
                        "sampling_rate_hz": SAMPLING_RATE_HZ,
                        "frame_index": np.arange(n_frames),
                        "raw_fluorescence": raw,
                    }
                )
            )
    return pd.concat(blocks, ignore_index=True)


class ExportAndEndToEndTests(unittest.TestCase):
    def test_provenance_validation_rejects_wrong_checksum(self):
        data = _synthetic_export()
        validate_export(data)
        data.loc[data.index[0], "source_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            validate_export(data)

    def test_synthetic_export_runs_without_raw_matlab_files(self):
        roi, slices, traces, exclusions = analyze_ne_responses(_synthetic_export())
        self.assertEqual(len(roi), 4)
        self.assertEqual(len(slices), 4)
        self.assertFalse(traces.empty)
        self.assertEqual(exclusions, [])
        for cohort in EXPECTED_SOURCES.values():
            paired = paired_slice_values(slices, cohort["cohort"])
            self.assertEqual(len(paired), 1)
            self.assertGreater(paired.iloc[0]["ko_minus_wt"], 0)

    def test_versioned_export_reproduces_documented_counts(self):
        export_path = os.path.join(
            ROOT, "data", "processed", "h1r_astrocytes_v1.csv.gz"
        )
        self.assertTrue(os.path.exists(export_path), "versioned H1R export is missing")
        digest = hashlib.sha256(Path(export_path).read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "4172cdb2eb5d3fa3b7b53648e6676a02b0314b47afcc73b81f03936ff5c2a7f6",
        )
        data = pd.read_csv(export_path)
        roi, slices, _, exclusions = analyze_ne_responses(data)

        self.assertEqual(len(data), 67_595)
        self.assertEqual(len(roi), 147)
        self.assertEqual(roi["slice_id"].nunique(), 13)
        self.assertEqual(exclusions, [])
        ne_only = paired_slice_values(slices, "ne_only_2023")
        post_ha = paired_slice_values(slices, "post_low_histamine_2025")
        self.assertEqual(len(ne_only), 5)
        self.assertEqual(int((ne_only["ko_minus_wt"] > 0).sum()), 5)
        self.assertEqual(len(post_ha), 7)
        self.assertEqual(int((post_ha["ko_minus_wt"] > 0).sum()), 6)

        common_ne = common_scale_paired_values(roi, "ne_only_2023")
        common_post = common_scale_paired_values(
            roi, "post_low_histamine_2025"
        )
        self.assertEqual(int((common_ne["ko_minus_wt"] > 0).sum()), 5)
        self.assertEqual(int((common_post["ko_minus_wt"] > 0).sum()), 6)
        self.assertAlmostEqual(common_ne["ko_minus_wt"].mean(), 11.66299704, places=6)
        self.assertAlmostEqual(common_post["ko_minus_wt"].mean(), 3.55874630, places=6)

        strict_ne = paired_slice_values(
            slices, "ne_only_2023", "median_robust_z_shift"
        )
        strict_post = paired_slice_values(
            slices, "post_low_histamine_2025", "median_robust_z_shift"
        )
        self.assertEqual(int((strict_ne["ko_minus_wt"] > 0).sum()), 3)
        self.assertEqual(int((strict_post["ko_minus_wt"] > 0).sum()), 5)
        self.assertAlmostEqual(strict_ne["ko_minus_wt"].mean(), 3.901764716, places=6)
        self.assertAlmostEqual(strict_post["ko_minus_wt"].mean(), 5.458939036, places=6)

        grid = build_specification_grid(data)
        self.assertEqual(len(grid), 72)
        for cohort in ("ne_only_2023", "post_low_histamine_2025"):
            cohort_grid = grid[grid["cohort"] == cohort]
            self.assertEqual(len(cohort_grid), 36)
            self.assertTrue((cohort_grid["common_scale_mean_effect"] > 0).all())
            self.assertTrue((cohort_grid["within_roi_z_mean_effect"] > 0).all())

        reproduced = build_summary(
            Path(export_path), data, roi, slices, exclusions,
            specification_grid=grid,
        )
        reproduced["derived_input"]["path"] = (
            "data/processed/h1r_astrocytes_v1.csv.gz"
        )
        summary_path = os.path.join(ROOT, "results", "h1r_astrocyte_summary.json")
        with open(summary_path, encoding="utf-8") as handle:
            committed = json.load(handle)
        self.assertEqual(reproduced, committed)
        self.assertEqual(
            [flag["code"] for flag in reproduced["qc_flags"]],
            ["raw_fluorescence_scale_outlier"],
        )


if __name__ == "__main__":
    unittest.main()
