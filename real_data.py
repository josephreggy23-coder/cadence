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
SEGMENTATION_PATH = "processing/ophys/ImageSegmentation/PlaneSegmentation"


@dataclass(frozen=True)
class RealCalciumRecording:
    """Fluorescence traces extracted from one real calcium-imaging recording."""

    fluorescence: np.ndarray
    time_s: np.ndarray
    roi_ids: np.ndarray
    species: str
    source_dandiset: str = ZEBRAFISH_DANDISET
    source_asset_id: str = ZEBRAFISH_ASSET_ID


@dataclass(frozen=True)
class DffRecording:
    """Per-ROI fluorescence normalized to a documented baseline."""

    dff: np.ndarray
    baseline_fluorescence: np.ndarray
    time_s: np.ndarray
    roi_ids: np.ndarray
    species: str


def compute_dff(
    recording: RealCalciumRecording,
    *,
    baseline_percentile: float = 20.0,
) -> DffRecording:
    """Normalize each ROI as (F - F0) / F0 using a fixed percentile F0."""
    if not 0.0 < baseline_percentile < 100.0:
        raise ValueError("baseline_percentile must be between 0 and 100")
    fluorescence = np.asarray(recording.fluorescence, dtype=np.float64)
    if fluorescence.ndim != 2 or fluorescence.size == 0:
        raise ValueError("fluorescence must be a non-empty frames x ROIs array")
    if not np.isfinite(fluorescence).all():
        raise ValueError("fluorescence must contain only finite values")
    baseline = np.percentile(fluorescence, baseline_percentile, axis=0)
    if np.any(baseline <= 0):
        raise ValueError("each ROI must have a strictly positive fluorescence baseline")
    dff = ((fluorescence - baseline) / baseline).astype(np.float32)
    return DffRecording(
        dff=dff,
        baseline_fluorescence=baseline.astype(np.float32),
        time_s=recording.time_s,
        roi_ids=recording.roi_ids,
        species=recording.species,
    )


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


def load_zebrafish_recording(
    path: str | Path,
    *,
    accepted_only: bool = True,
    max_rois: int | None = None,
) -> RealCalciumRecording:
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

        segmentation = file[SEGMENTATION_PATH]
        roi_rows = np.asarray(series["rois"], dtype=int)
        roi_ids = np.asarray(segmentation["id"])[roi_rows]
        if fluorescence.shape[1] != roi_rows.size:
            raise ValueError("ROI references must match fluorescence columns")
        if accepted_only:
            accepted = np.asarray(segmentation["Accepted"], dtype=bool)[roi_rows]
            fluorescence = fluorescence[:, accepted]
            roi_ids = roi_ids[accepted]
        if max_rois is not None:
            fluorescence = fluorescence[:, :max_rois]
            roi_ids = roi_ids[:max_rois]
        rate = float(series["starting_time"].attrs["rate"])
        if rate <= 0:
            raise ValueError("fluorescence sampling rate must be positive")
        species = file["general/subject/species"][()].decode("utf-8")

    time_s = np.arange(fluorescence.shape[0], dtype=np.float32) / rate
    return RealCalciumRecording(
        fluorescence=fluorescence,
        time_s=time_s,
        roi_ids=roi_ids,
        species=species,
    )
