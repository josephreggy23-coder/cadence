import tempfile
import unittest
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_data import (
    DffRecording,
    FLUORESCENCE_PATH,
    RealCalciumRecording,
    SEGMENTATION_PATH,
    compute_dff,
    load_zebrafish_recording,
    summarize_recording,
)


class RealDataTests(unittest.TestCase):
    def test_loader_reads_fluorescence_and_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.nwb"
            with h5py.File(path, "w") as file:
                series = file.create_group(FLUORESCENCE_PATH)
                series.create_dataset("data", data=np.arange(12).reshape(4, 3))
                starting_time = series.create_dataset("starting_time", data=0.0)
                starting_time.attrs["rate"] = 2.0
                series.create_dataset("rois", data=[2, 0, 1])
                segmentation = file.create_group(SEGMENTATION_PATH)
                segmentation.create_dataset("id", data=[10, 20, 30])
                segmentation.create_dataset("Accepted", data=[1, 0, 1])
                file.create_dataset("general/subject/species", data=np.bytes_("Danio rerio"))

            recording = load_zebrafish_recording(path, max_rois=2)

        self.assertEqual(recording.fluorescence.shape, (4, 2))
        np.testing.assert_array_equal(recording.roi_ids, [30, 10])
        np.testing.assert_allclose(recording.time_s, [0.0, 0.5, 1.0, 1.5])
        self.assertEqual(recording.species, "Danio rerio")
        summary = summarize_recording(recording)
        self.assertEqual(summary["n_rois"], 2)
        self.assertEqual(summary["duration_s"], 1.5)

    def test_roi_limit_applies_after_quality_filtering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.nwb"
            with h5py.File(path, "w") as file:
                series = file.create_group(FLUORESCENCE_PATH)
                series.create_dataset("data", data=np.arange(12).reshape(4, 3))
                starting_time = series.create_dataset("starting_time", data=0.0)
                starting_time.attrs["rate"] = 2.0
                series.create_dataset("rois", data=[0, 1, 2])
                segmentation = file.create_group(SEGMENTATION_PATH)
                segmentation.create_dataset("id", data=[10, 20, 30])
                segmentation.create_dataset("Accepted", data=[0, 1, 1])
                file.create_dataset("general/subject/species", data=np.bytes_("Danio rerio"))

            recording = load_zebrafish_recording(path, max_rois=1)

        np.testing.assert_array_equal(recording.roi_ids, [20])

    def test_dff_uses_a_per_roi_percentile_baseline(self):
        recording = RealCalciumRecording(
            fluorescence=np.array([[2.0, 4.0], [4.0, 8.0], [6.0, 12.0]]),
            time_s=np.array([0.0, 1.0, 2.0]),
            roi_ids=np.array([4, 9]),
            species="Danio rerio",
        )
        normalized = compute_dff(recording, baseline_percentile=50)
        self.assertIsInstance(normalized, DffRecording)
        np.testing.assert_allclose(normalized.baseline_fluorescence, [4.0, 8.0])
        np.testing.assert_allclose(normalized.dff, [[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5]])
        np.testing.assert_array_equal(normalized.roi_ids, recording.roi_ids)

    def test_dff_rejects_nonfinite_fluorescence(self):
        recording = RealCalciumRecording(
            fluorescence=np.array([[1.0], [np.nan]]),
            time_s=np.array([0.0, 1.0]),
            roi_ids=np.array([1]),
            species="Danio rerio",
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            compute_dff(recording)
