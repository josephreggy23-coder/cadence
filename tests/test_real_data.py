import tempfile
import unittest
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_data import FLUORESCENCE_PATH, load_zebrafish_recording, summarize_recording


class RealDataTests(unittest.TestCase):
    def test_loader_reads_fluorescence_and_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.nwb"
            with h5py.File(path, "w") as file:
                series = file.create_group(FLUORESCENCE_PATH)
                series.create_dataset("data", data=np.arange(12).reshape(4, 3))
                starting_time = series.create_dataset("starting_time", data=0.0)
                starting_time.attrs["rate"] = 2.0
                file.create_dataset("general/subject/species", data=np.bytes_("Danio rerio"))

            recording = load_zebrafish_recording(path, max_rois=2)

        self.assertEqual(recording.fluorescence.shape, (4, 2))
        np.testing.assert_allclose(recording.time_s, [0.0, 0.5, 1.0, 1.5])
        self.assertEqual(recording.species, "Danio rerio")
        summary = summarize_recording(recording)
        self.assertEqual(summary["n_rois"], 2)
        self.assertEqual(summary["duration_s"], 1.5)
