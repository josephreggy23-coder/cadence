"""Load selected real zebrafish calcium-imaging recordings from NWB/HDF5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


ZEBRAFISH_DANDISET = "001076"
ZEBRAFISH_ASSET_ID = "9676d05a-7655-41ec-aebd-271c334c0649"
ZEBRAFISH_ASSET_PATH = "sub-nan/sub-nan_ses-20230123T192927_obj-17bhudf_ophys.nwb"
FLUORESCENCE_PATH = "processing/ophys/Fluorescence/RoiResponseSeries"


@dataclass(frozen=True)
class RealCalciumRecording:
    """Fluorescence traces extracted from one real calcium-imaging recording."""

    fluorescence: np.ndarray
    time_s: np.ndarray
    species: str
    source_dandiset: str = ZEBRAFISH_DANDISET
    source_asset_id: str = ZEBRAFISH_ASSET_ID


def summarize_recording(recording: RealCalciumRecording) -> dict[str, float | int | str]:
    """Return transparent, label-free quality-control metrics for a recording."""
    fluorescence = recording.fluorescence
    return {
        "species": recording.species,
        "n_frames": int(fluorescence.shape[0]),
        "n_rois": int(fluorescence.shape[1]),
        "duration_s": float(recording.time_s[-1]),
        "mean_fluorescence": float(fluorescence.mean()),
        "median_roi_std": float(np.median(fluorescence.std(axis=0))),
    }


def load_zebrafish_recording(path: str | Path, *, max_rois: int | None = None) -> RealCalciumRecording:
    """Load fluorescence traces from the selected DANDI zebrafish NWB asset.

    The DANDI asset uses NWB 2.6.0-alpha metadata that is incompatible with the
    installed PyNWB reader. This focused HDF5 reader accesses only the standard
    RoiResponseSeries data and timing fields required by CADENCE.
    """
    if max_rois is not None and max_rois < 1:
        raise ValueError("max_rois must be at least one or None")

    with h5py.File(path, "r") as file:
        series = file[FLUORESCENCE_PATH]
        fluorescence = np.asarray(series["data"], dtype=np.float32)
        if fluorescence.ndim != 2:
            raise ValueError("fluorescence data must have shape frames x ROIs")
        if max_rois is not None:
            fluorescence = fluorescence[:, :max_rois]
        rate = float(series["starting_time"].attrs["rate"])
        if rate <= 0:
            raise ValueError("fluorescence sampling rate must be positive")
        species = file["general/subject/species"][()].decode("utf-8")

    time_s = np.arange(fluorescence.shape[0], dtype=np.float32) / rate
    return RealCalciumRecording(fluorescence=fluorescence, time_s=time_s, species=species)
