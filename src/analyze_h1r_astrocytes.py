"""Hierarchical secondary analysis of public H1R astrocyte calcium data.

This module analyzes the compact export produced by
``scripts/export_h1r_astrocytes.m``.  It reproduces the contextual dF/F and
nominal 30-second pre/post AUC calculation in the dataset authors' accompanying
notebook, then *first* collapses ROI measurements within each slice.  The
primary WT/KO display is paired by slice; ROIs are never presented as
independent biological replicates.

The Dryad deposit does not provide animal IDs.  Results are therefore
descriptive at the slice level and cannot support animal-level inference.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SAMPLING_RATE_HZ = 0.71
FLUORESCENCE_OFFSET = 100_000.0
SCHEMA_VERSION = "h1r-astrocytes-v1"
SOURCE_DOI = "10.5061/dryad.2280gb64x"

EXPECTED_SOURCES = {
    4_342_955: {
        "file_name": "Fig5_H1RKO_NE.mat",
        "sha256": "45c7836bcafda6369a8addf436c2838c04ab61c53d3881fe52416821fe5e8116",
        "cohort": "ne_only_2023",
    },
    4_342_954: {
        "file_name": "Fig5_H1RKO_NE_postLowHA.mat",
        "sha256": "d39c8e8453996d5f3abdc42eed0b4ea4b6485b5e142719d4128f9683972621eb",
        "cohort": "post_low_histamine_2025",
    },
}

REQUIRED_COLUMNS = {
    "schema_version",
    "source_doi",
    "source_file_id",
    "source_file",
    "source_sha256",
    "cohort",
    "slice_id",
    "stimulus",
    "genotype",
    "roi_id",
    "onset_index",
    "sampling_rate_hz",
    "frame_index",
    "raw_fluorescence",
}

COHORT_LABELS = {
    "ne_only_2023": "NE only (2023, jRGECO)",
    "post_low_histamine_2025": "NE after low HA (2025, GCaMP)",
}
GENOTYPE_COLORS = {"WT": "#303642", "KO": "#16A085"}
ROBUST_SCALE_FACTOR = 1.4826


def contextual_dff(
    trace: Iterable[float],
    onset_index: int,
    sampling_rate_hz: float = SAMPLING_RATE_HZ,
    offset: float = FLUORESCENCE_OFFSET,
    fallback_start_seconds: float = 40.0,
) -> tuple[np.ndarray, float, str]:
    """Compute the authors' contextual dF/F transform.

    The default baseline is the median from 70 s to 10 s before onset.  Frame
    counts are floored, matching the accompanying notebook.  If 70 s of
    history are unavailable, a source-notebook-specific shorter window is used:
    40 s to 10 s before onset for the 2023 file and 30 s to 10 s for the 2025
    file. A trace without the requested fallback history is rejected.
    """

    values = np.asarray(trace, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("trace must be a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("trace contains non-finite values")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    if fallback_start_seconds <= 10:
        raise ValueError("fallback_start_seconds must be greater than 10")

    onset_index = int(onset_index)
    pre_end = onset_index - int(10 * sampling_rate_hz)
    pre_start = onset_index - int(70 * sampling_rate_hz)
    baseline_method = "onset_minus_70s_to_minus_10s"
    if pre_start < 0:
        pre_start = onset_index - int(fallback_start_seconds * sampling_rate_hz)
        fallback_label = f"{fallback_start_seconds:g}"
        baseline_method = (
            f"fallback_onset_minus_{fallback_label}s_to_minus_10s"
        )
    if pre_start < 0 or pre_end <= pre_start:
        raise ValueError("insufficient pre-onset samples for contextual baseline")

    shifted = values + float(offset)
    f0 = float(np.median(shifted[pre_start:pre_end]))
    if not np.isfinite(f0) or np.isclose(f0, 0.0):
        raise ValueError("contextual baseline is zero or non-finite")
    return (shifted - f0) / f0, f0, baseline_method


def pre_post_delta_auc(
    dff: Iterable[float],
    onset_index: int,
    sampling_rate_hz: float = SAMPLING_RATE_HZ,
    window_seconds: float = 30.0,
) -> tuple[float, float, float, int]:
    """Return pre AUC, post AUC, post-minus-pre AUC, and samples per window.

    ``int(30 * 0.71) == 21`` samples are used on each side, matching the source
    notebook.  Because trapezoidal integration spans intervals between samples,
    the effective integrated duration is 20 / 0.71 = 28.17 s per nominal
    30-second window.  The onset sample is included in the post window and
    excluded from the pre window.
    """

    values = np.asarray(dff, dtype=float)
    onset_index = int(onset_index)
    samples = int(window_seconds * sampling_rate_hz)
    if samples < 2:
        raise ValueError("AUC window must contain at least two samples")
    if onset_index - samples < 0 or onset_index + samples > values.size:
        raise ValueError("trace is too short for symmetric pre/post AUC windows")

    pre = values[onset_index - samples : onset_index]
    post = values[onset_index : onset_index + samples]
    if not np.isfinite(pre).all() or not np.isfinite(post).all():
        raise ValueError("AUC window contains non-finite values")
    dx = 1.0 / sampling_rate_hz
    # numpy.trapezoid was introduced after the project's minimum NumPy version.
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    pre_auc = float(trapezoid(pre, dx=dx))
    post_auc = float(trapezoid(post, dx=dx))
    return pre_auc, post_auc, post_auc - pre_auc, samples


def robust_response_metrics(
    trace: Iterable[float],
    onset_index: int,
    sampling_rate_hz: float = SAMPLING_RATE_HZ,
    baseline_start_seconds: float = 70.0,
    response_window_seconds: float = 30.0,
    temporal_summary: str = "median",
) -> tuple[float, float, float]:
    """Return an offset-free response, robust baseline scale, and response z.

    The response is post-minus-pre raw fluorescence. The scale is 1.4826 times
    the MAD over the baseline ending 10 seconds before onset. Unlike contextual
    dF/F, these values are invariant to additive fluorescence offsets; the z
    value is also invariant to positive multiplicative gain.
    """

    values = np.asarray(trace, dtype=float)
    onset_index = int(onset_index)
    if temporal_summary not in {"mean", "median"}:
        raise ValueError("temporal_summary must be 'mean' or 'median'")
    if baseline_start_seconds <= 10:
        raise ValueError("baseline_start_seconds must be greater than 10")
    samples = int(response_window_seconds * sampling_rate_hz)
    baseline_start = onset_index - int(baseline_start_seconds * sampling_rate_hz)
    baseline_end = onset_index - int(10 * sampling_rate_hz)
    if (
        samples < 1
        or onset_index - samples < 0
        or onset_index + samples > values.size
        or baseline_start < 0
        or baseline_end <= baseline_start
    ):
        raise ValueError("trace is too short for orthogonal response windows")
    if not np.isfinite(values).all():
        raise ValueError("trace contains non-finite values")

    reducer = np.mean if temporal_summary == "mean" else np.median
    pre = values[onset_index - samples : onset_index]
    post = values[onset_index : onset_index + samples]
    baseline = values[baseline_start:baseline_end]
    raw_shift = float(reducer(post) - reducer(pre))
    baseline_median = float(np.median(baseline))
    robust_scale = float(
        ROBUST_SCALE_FACTOR * np.median(np.abs(baseline - baseline_median))
    )
    if not np.isfinite(robust_scale) or robust_scale <= 0:
        raise ValueError("baseline MAD scale is zero or non-finite")
    return raw_shift, robust_scale, raw_shift / robust_scale


def validate_export(data: pd.DataFrame) -> None:
    """Fail loudly if the compact export's schema or provenance has drifted."""

    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"export is missing required columns: {sorted(missing)}")
    if set(data["schema_version"].astype(str).unique()) != {SCHEMA_VERSION}:
        raise ValueError("unexpected export schema version")
    if set(data["source_doi"].astype(str).unique()) != {SOURCE_DOI}:
        raise ValueError("unexpected source DOI")
    rates = data["sampling_rate_hz"].astype(float).unique()
    if len(rates) != 1 or not np.isclose(rates[0], SAMPLING_RATE_HZ):
        raise ValueError("unexpected or inconsistent sampling rate")

    observed_ids = set(data["source_file_id"].astype(int).unique())
    if observed_ids != set(EXPECTED_SOURCES):
        raise ValueError(f"unexpected Dryad file IDs: {sorted(observed_ids)}")
    for file_id, expected in EXPECTED_SOURCES.items():
        subset = data[data["source_file_id"].astype(int) == file_id]
        checksums = set(subset["source_sha256"].astype(str).str.lower().unique())
        files = set(subset["source_file"].astype(str).unique())
        cohorts = set(subset["cohort"].astype(str).unique())
        if checksums != {expected["sha256"]}:
            raise ValueError(f"checksum mismatch in provenance for Dryad file {file_id}")
        if files != {expected["file_name"]} or cohorts != {expected["cohort"]}:
            raise ValueError(f"source metadata mismatch for Dryad file {file_id}")


def analyze_ne_responses(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    """Compute ROI metrics, slice aggregates, dF/F samples, and exclusions."""

    validate_export(data)
    ne = data[data["stimulus"].astype(str).str.upper() == "NE"].copy()
    group_columns = [
        "source_file_id",
        "source_file",
        "source_sha256",
        "cohort",
        "slice_id",
        "genotype",
        "roi_id",
    ]
    metric_records: list[dict[str, object]] = []
    trace_blocks: list[pd.DataFrame] = []
    exclusions: list[dict[str, str]] = []

    for keys, group in ne.groupby(group_columns, sort=True, observed=True):
        metadata = dict(zip(group_columns, keys))
        ordered = group.sort_values("frame_index")
        onset_values = ordered["onset_index"].astype(int).unique()
        rate_values = ordered["sampling_rate_hz"].astype(float).unique()
        if len(onset_values) != 1 or len(rate_values) != 1:
            raise ValueError(f"inconsistent onset or rate for {metadata}")
        onset_index = int(onset_values[0])
        rate = float(rate_values[0])
        frames = ordered["frame_index"].astype(int).to_numpy()
        if not np.array_equal(frames, np.arange(frames.size)):
            raise ValueError(f"non-contiguous frame indices for {metadata}")

        raw = ordered["raw_fluorescence"].astype(float).to_numpy()
        try:
            fallback_start_seconds = (
                30.0 if metadata["cohort"] == "post_low_histamine_2025" else 40.0
            )
            dff, f0, baseline_method = contextual_dff(
                raw,
                onset_index,
                rate,
                fallback_start_seconds=fallback_start_seconds,
            )
            pre_auc, post_auc, delta_auc, auc_samples = pre_post_delta_auc(
                dff, onset_index, rate
            )
        except ValueError as error:
            exclusions.append({
                "cohort": str(metadata["cohort"]),
                "slice_id": str(metadata["slice_id"]),
                "genotype": str(metadata["genotype"]),
                "roi_id": str(metadata["roi_id"]),
                "reason": str(error),
            })
            continue

        try:
            raw_median_shift, baseline_mad_scale, robust_z_shift = (
                robust_response_metrics(raw, onset_index, rate)
            )
            orthogonal_issue = ""
        except ValueError as error:
            # The authors' ΔAUC analysis can remain valid when a flat baseline
            # makes the orthogonal noise-standardized metric undefined.
            raw_median_shift = baseline_mad_scale = robust_z_shift = np.nan
            orthogonal_issue = str(error)

        metric_records.append(
            {
                **metadata,
                "onset_index": onset_index,
                "sampling_rate_hz": rate,
                "n_frames": raw.size,
                "area_pixels": float(ordered["area_pixels"].iloc[0])
                if "area_pixels" in ordered
                else np.nan,
                "fluorescence_offset": FLUORESCENCE_OFFSET,
                "baseline_f0": f0,
                "raw_baseline_median": f0 - FLUORESCENCE_OFFSET,
                "baseline_method": baseline_method,
                "auc_window_nominal_s": 30.0,
                "auc_samples_per_window": auc_samples,
                "auc_effective_duration_s": (auc_samples - 1) / rate,
                "pre_auc": pre_auc,
                "post_auc": post_auc,
                "delta_auc": delta_auc,
                "raw_median_shift": raw_median_shift,
                "baseline_mad_scale": baseline_mad_scale,
                "robust_z_shift": robust_z_shift,
                "orthogonal_issue": orthogonal_issue,
            }
        )
        trace_blocks.append(
            pd.DataFrame(
                {
                    "cohort": metadata["cohort"],
                    "slice_id": metadata["slice_id"],
                    "genotype": metadata["genotype"],
                    "roi_id": metadata["roi_id"],
                    "frame_offset": frames - onset_index,
                    "time_from_onset_s": (frames - onset_index) / rate,
                    "dff": dff,
                }
            )
        )

    roi_metrics = pd.DataFrame.from_records(metric_records)
    if roi_metrics.empty:
        raise ValueError("no NE ROI passed the pre/post analysis requirements")
    trace_data = pd.concat(trace_blocks, ignore_index=True)
    slice_metrics = aggregate_roi_metrics_by_slice(roi_metrics)
    return roi_metrics, slice_metrics, trace_data, exclusions


def aggregate_roi_metrics_by_slice(roi_metrics: pd.DataFrame) -> pd.DataFrame:
    """Collapse ROI delta-AUC values within each slice and genotype."""

    required = {"cohort", "slice_id", "genotype", "delta_auc"}
    missing = required - set(roi_metrics.columns)
    if missing:
        raise ValueError(f"ROI metrics missing columns: {sorted(missing)}")
    roi_metrics = roi_metrics.copy()
    for optional in (
        "raw_median_shift",
        "robust_z_shift",
        "baseline_mad_scale",
    ):
        if optional not in roi_metrics:
            roi_metrics[optional] = np.nan
    return (
        roi_metrics.groupby(
            ["cohort", "slice_id", "genotype"], sort=True, observed=True
        )
        .agg(
            n_rois=("delta_auc", "size"),
            mean_delta_auc=("delta_auc", "mean"),
            median_delta_auc=("delta_auc", "median"),
            median_raw_median_shift=("raw_median_shift", "median"),
            median_robust_z_shift=("robust_z_shift", "median"),
            median_baseline_mad_scale=("baseline_mad_scale", "median"),
        )
        .reset_index()
    )


def paired_slice_values(
    slice_metrics: pd.DataFrame,
    cohort: str,
    value_column: str = "mean_delta_auc",
) -> pd.DataFrame:
    """Return one WT/KO mean-delta-AUC pair per slice for a protocol cohort."""

    cohort_rows = slice_metrics[slice_metrics["cohort"] == cohort]
    if value_column not in cohort_rows:
        raise ValueError(f"slice metrics do not contain {value_column}")
    paired = cohort_rows.pivot(
        index="slice_id", columns="genotype", values=value_column
    )
    if "WT" not in paired or "KO" not in paired:
        return pd.DataFrame(columns=["WT", "KO", "ko_minus_wt"])
    paired = paired[["WT", "KO"]].dropna().sort_index()
    paired["ko_minus_wt"] = paired["KO"] - paired["WT"]
    return paired


def common_scale_paired_values(
    roi_metrics: pd.DataFrame, cohort: str
) -> pd.DataFrame:
    """Pair slice responses using one robust baseline scale per slice.

    A shared denominator preserves the within-slice KO−WT amplitude contrast,
    while removing additive offsets and common positive gain.
    """

    cohort_rows = roi_metrics[roi_metrics["cohort"] == cohort].dropna(
        subset=["raw_median_shift", "baseline_mad_scale"]
    )
    response = cohort_rows.pivot_table(
        index="slice_id",
        columns="genotype",
        values="raw_median_shift",
        aggfunc="median",
    )
    if "WT" not in response or "KO" not in response:
        return pd.DataFrame(
            columns=["WT", "KO", "ko_minus_wt", "slice_scale"]
        )
    response = response[["WT", "KO"]].dropna().sort_index()
    scales = cohort_rows.groupby("slice_id", sort=True)[
        "baseline_mad_scale"
    ].median()
    response["slice_scale"] = scales.reindex(response.index)
    response = response[response["slice_scale"] > 0].copy()
    response[["WT", "KO"]] = response[["WT", "KO"]].div(
        response["slice_scale"], axis=0
    )
    response["ko_minus_wt"] = response["KO"] - response["WT"]
    return response


def _orthogonal_roi_table(
    data: pd.DataFrame,
    baseline_start_seconds: float,
    response_window_seconds: float,
    temporal_summary: str,
) -> pd.DataFrame:
    """Compute one declared orthogonal-response specification per ROI."""

    records = []
    ne = data[data["stimulus"].astype(str).str.upper() == "NE"]
    keys = ["cohort", "slice_id", "genotype", "roi_id"]
    for values, group in ne.groupby(keys, sort=True, observed=True):
        ordered = group.sort_values("frame_index")
        raw = ordered["raw_fluorescence"].astype(float).to_numpy()
        raw_shift, scale, robust_z = robust_response_metrics(
            raw,
            int(ordered["onset_index"].iloc[0]),
            float(ordered["sampling_rate_hz"].iloc[0]),
            baseline_start_seconds=baseline_start_seconds,
            response_window_seconds=response_window_seconds,
            temporal_summary=temporal_summary,
        )
        records.append(
            {
                **dict(zip(keys, values)),
                "raw_shift": raw_shift,
                "baseline_mad_scale": scale,
                "robust_z": robust_z,
            }
        )
    return pd.DataFrame.from_records(records)


def build_specification_grid(data: pd.DataFrame) -> pd.DataFrame:
    """Run a fixed 36-cell exploratory robustness grid.

    The grid was declared during repository auditing after the primary data were
    inspected; it is a multiverse sensitivity analysis, not a preregistration.
    """

    records = []
    for baseline_start, response_window, temporal, roi_aggregation in itertools.product(
        (40.0, 60.0, 70.0),
        (15.0, 30.0, 45.0),
        ("mean", "median"),
        ("mean", "median"),
    ):
        roi = _orthogonal_roi_table(
            data, baseline_start, response_window, temporal
        )
        reducer = np.mean if roi_aggregation == "mean" else np.median
        for cohort in sorted(roi["cohort"].unique()):
            cohort_roi = roi[roi["cohort"] == cohort]
            raw_by_slice = cohort_roi.pivot_table(
                index="slice_id",
                columns="genotype",
                values="raw_shift",
                aggfunc=reducer,
            )
            z_by_slice = cohort_roi.pivot_table(
                index="slice_id",
                columns="genotype",
                values="robust_z",
                aggfunc=reducer,
            )
            paired_ids = raw_by_slice.index.intersection(z_by_slice.index)
            raw_by_slice = raw_by_slice.reindex(paired_ids)
            z_by_slice = z_by_slice.reindex(paired_ids)
            if not {"WT", "KO"}.issubset(raw_by_slice.columns):
                continue
            raw_by_slice = raw_by_slice[["WT", "KO"]].dropna()
            z_by_slice = z_by_slice[["WT", "KO"]].dropna()
            paired_ids = raw_by_slice.index.intersection(z_by_slice.index)
            raw_by_slice = raw_by_slice.reindex(paired_ids)
            z_by_slice = z_by_slice.reindex(paired_ids)
            scale = cohort_roi.groupby("slice_id", sort=True)[
                "baseline_mad_scale"
            ].median().reindex(paired_ids)
            common_effect = (
                raw_by_slice["KO"] - raw_by_slice["WT"]
            ) / scale
            within_z_effect = z_by_slice["KO"] - z_by_slice["WT"]
            records.append(
                {
                    "baseline_start_s": baseline_start,
                    "response_window_s": response_window,
                    "temporal_summary": temporal,
                    "roi_aggregation": roi_aggregation,
                    "cohort": cohort,
                    "paired_slice_count": int(len(common_effect)),
                    "common_scale_mean_effect": float(common_effect.mean()),
                    "common_scale_median_effect": float(common_effect.median()),
                    "common_scale_positive_slices": int((common_effect > 0).sum()),
                    "within_roi_z_mean_effect": float(within_z_effect.mean()),
                    "within_roi_z_median_effect": float(within_z_effect.median()),
                    "within_roi_z_positive_slices": int(
                        (within_z_effect > 0).sum()
                    ),
                }
            )
    return pd.DataFrame.from_records(records).sort_values(
        [
            "cohort",
            "baseline_start_s",
            "response_window_s",
            "temporal_summary",
            "roi_aggregation",
        ]
    ).reset_index(drop=True)


def _paired_effect_summary(paired: pd.DataFrame) -> dict[str, object]:
    effects = paired["ko_minus_wt"].astype(float).dropna()
    return {
        "paired_slice_count": int(len(effects)),
        "positive_slice_count": int((effects > 0).sum()),
        "effect": _finite_summary(effects),
        "paired_values": [
            {"slice_id": str(slice_id), "ko_minus_wt": float(value)}
            for slice_id, value in effects.items()
        ],
    }


def _finite_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0, "mean": None, "median": None, "sd": None, "q1": None, "q3": None}
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "sd": float(np.std(array, ddof=1)) if array.size > 1 else None,
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
    }


def build_summary(
    export_path: Path,
    data: pd.DataFrame,
    roi_metrics: pd.DataFrame,
    slice_metrics: pd.DataFrame,
    exclusions: list[dict[str, str]],
    specification_grid: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Build a JSON-safe, machine-readable descriptive summary."""

    digest = hashlib.sha256(export_path.read_bytes()).hexdigest()
    cohort_summaries: dict[str, object] = {}
    qc_flags: list[dict[str, object]] = []
    for cohort in sorted(roi_metrics["cohort"].unique()):
        cohort_roi = roi_metrics[roi_metrics["cohort"] == cohort]
        cohort_slice = slice_metrics[slice_metrics["cohort"] == cohort]
        paired = paired_slice_values(slice_metrics, cohort)
        slice_raw_scale = cohort_roi.groupby("slice_id", sort=True)[
            "raw_baseline_median"
        ].median()
        dominant_scale = float(np.median(np.abs(slice_raw_scale)))
        if dominant_scale > 0:
            for slice_id, raw_scale in slice_raw_scale.items():
                relative_scale = abs(float(raw_scale)) / dominant_scale
                if relative_scale < 0.1 or relative_scale > 10:
                    qc_flags.append(
                        {
                            "code": "raw_fluorescence_scale_outlier",
                            "cohort": str(cohort),
                            "slice_id": str(slice_id),
                            "slice_raw_baseline_median": float(raw_scale),
                            "cohort_median_absolute_raw_baseline": dominant_scale,
                            "relative_absolute_scale": relative_scale,
                            "action": "retained to reproduce the deposited analysis; interpret cautiously",
                        }
                    )
        by_genotype: dict[str, object] = {}
        for genotype in ("WT", "KO"):
            genotype_roi = cohort_roi[cohort_roi["genotype"] == genotype]
            genotype_slice = cohort_slice[cohort_slice["genotype"] == genotype]
            by_genotype[genotype] = {
                "roi_count": int(len(genotype_roi)),
                "slice_count": int(genotype_slice["slice_id"].nunique()),
                "slice_mean_delta_auc": _finite_summary(
                    genotype_slice["mean_delta_auc"]
                ),
            }
        cohort_summaries[cohort] = {
            "label": COHORT_LABELS.get(cohort, cohort),
            "by_genotype": by_genotype,
            "paired_slice_count": int(len(paired)),
            "paired_slice_ids": [str(value) for value in paired.index],
            "paired_ko_minus_wt": _finite_summary(paired["ko_minus_wt"]),
            "paired_values": [
                {
                    "slice_id": str(slice_id),
                    "wt_mean_delta_auc": float(row["WT"]),
                    "ko_mean_delta_auc": float(row["KO"]),
                    "ko_minus_wt": float(row["ko_minus_wt"]),
                }
                for slice_id, row in paired.iterrows()
            ],
        }

    if specification_grid is None:
        specification_grid = build_specification_grid(data)
    orthogonal_cohorts: dict[str, object] = {}
    flagged_slice_ids = {str(flag["slice_id"]) for flag in qc_flags}
    for cohort in sorted(roi_metrics["cohort"].unique()):
        raw_paired = paired_slice_values(
            slice_metrics, cohort, "median_raw_median_shift"
        )
        strict_z_paired = paired_slice_values(
            slice_metrics, cohort, "median_robust_z_shift"
        )
        common_paired = common_scale_paired_values(roi_metrics, cohort)
        loo_means = [
            float(strict_z_paired.drop(index=slice_id)["ko_minus_wt"].mean())
            for slice_id in strict_z_paired.index
            if len(strict_z_paired) > 1
        ]
        without_flagged = strict_z_paired[
            ~strict_z_paired.index.astype(str).isin(flagged_slice_ids)
        ]
        grid_rows = specification_grid[
            specification_grid["cohort"] == cohort
        ]
        orthogonal_cohorts[cohort] = {
            "offset_free_raw_median_shift": _paired_effect_summary(raw_paired),
            "paired_common_scale_effect": _paired_effect_summary(common_paired),
            "within_roi_robust_z_sensitivity": {
                **_paired_effect_summary(strict_z_paired),
                "leave_one_slice_out_mean_min": float(min(loo_means)),
                "excluding_flagged_slices": _paired_effect_summary(without_flagged),
            },
            "specification_grid": {
                "specification_count": int(len(grid_rows)),
                "common_scale_mean_effect_range": [
                    float(grid_rows["common_scale_mean_effect"].min()),
                    float(grid_rows["common_scale_mean_effect"].max()),
                ],
                "common_scale_positive_slice_count_range": [
                    int(grid_rows["common_scale_positive_slices"].min()),
                    int(grid_rows["common_scale_positive_slices"].max()),
                ],
                "within_roi_z_mean_effect_range": [
                    float(grid_rows["within_roi_z_mean_effect"].min()),
                    float(grid_rows["within_roi_z_mean_effect"].max()),
                ],
                "within_roi_z_positive_slice_count_range": [
                    int(grid_rows["within_roi_z_positive_slices"].min()),
                    int(grid_rows["within_roi_z_positive_slices"].max()),
                ],
            },
        }

    source_records = []
    for file_id, expected in sorted(EXPECTED_SOURCES.items()):
        source_records.append(
            {
                "doi": SOURCE_DOI,
                "dryad_version_id": 394734,
                "dryad_file_id": file_id,
                **expected,
            }
        )
    return {
        "analysis_schema_version": "h1r-secondary-analysis-v3",
        "derived_input": {
            "path": export_path.as_posix(),
            "sha256": digest,
            "row_count": int(len(data)),
            "source_files": source_records,
        },
        "method": {
            "sampling_rate_hz": SAMPLING_RATE_HZ,
            "fluorescence_offset": FLUORESCENCE_OFFSET,
            "baseline_default": "median from onset-70 s to onset-10 s",
            "baseline_fallback_by_cohort": {
                "ne_only_2023": "median from onset-40 s to onset-10 s",
                "post_low_histamine_2025": "median from onset-30 s to onset-10 s",
            },
            "auc": "post minus pre trapezoidal AUC; nominal 30 s on each side",
            "auc_samples_per_window": int(30 * SAMPLING_RATE_HZ),
            "auc_effective_duration_s": (int(30 * SAMPLING_RATE_HZ) - 1)
            / SAMPLING_RATE_HZ,
            "primary_unit": "slice/genotype mean after aggregating ROIs",
            "comparison": "paired WT versus KO within slices that contain both genotypes",
        },
        "counts": {
            "analyzed_roi_count": int(len(roi_metrics)),
            "analyzed_slice_count": int(roi_metrics["slice_id"].nunique()),
            "excluded_roi_count": int(len(exclusions)),
        },
        "cohorts": cohort_summaries,
        "offset_free_robustness": {
            "status": (
                "alternative metric on the same recordings; exploratory sensitivity "
                "analysis declared during repository audit; not preregistered or an "
                "independent replication"
            ),
            "default_windows": {
                "baseline": "onset-70 s to onset-10 s",
                "response": "median post 0-30 s minus median pre 30-0 s",
                "robust_scale": "1.4826 times baseline MAD",
            },
            "common_scale_definition": (
                "paired median-ROI KO-WT raw response divided by the median "
                "baseline MAD scale pooled across both genotypes in that slice"
            ),
            "specification_grid": {
                "baseline_start_s": [40, 60, 70],
                "response_window_s": [15, 30, 45],
                "temporal_summary": ["mean", "median"],
                "roi_to_slice_aggregation": ["mean", "median"],
                "total_specifications_per_cohort": 36,
            },
            "cohorts": orthogonal_cohorts,
        },
        "exclusions": exclusions,
        "qc_flags": qc_flags,
        "limitations": [
            "Animal IDs are absent from the deposited tables, so slices cannot be mapped to animals.",
            "All comparisons are descriptive secondary analyses; slices and ROIs are not treated as animals.",
            "The two cohorts use different sensors/protocol contexts and are summarized separately, not pooled.",
            "The nominal 30-second trapezoidal AUC uses 21 samples and spans 28.17 seconds at 0.71 Hz, matching the source notebook.",
            "A deposited slice with a markedly different raw fluorescence scale is flagged and retained rather than silently excluded.",
            "The stricter within-ROI noise-standardized sensitivity is less slice-consistent than the common-scale amplitude result.",
        ],
    }


def make_figure(
    trace_data: pd.DataFrame,
    slice_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render a hierarchy-aware trace and paired-slice summary figure."""

    cohorts = [cohort for cohort in COHORT_LABELS if cohort in set(trace_data["cohort"])]
    fig, axes = plt.subplots(
        len(cohorts), 2, figsize=(11.5, 4.1 * len(cohorts)), squeeze=False
    )
    fig.patch.set_facecolor("#F7F9FC")

    for row_index, cohort in enumerate(cohorts):
        trace_ax, auc_ax = axes[row_index]
        for axis in (trace_ax, auc_ax):
            axis.set_facecolor("white")
            axis.spines[["top", "right"]].set_visible(False)

        cohort_trace = trace_data[
            (trace_data["cohort"] == cohort)
            & (trace_data["time_from_onset_s"] >= -45)
            & (trace_data["time_from_onset_s"] <= 60)
        ]
        slice_traces = (
            cohort_trace.groupby(
                ["slice_id", "genotype", "frame_offset", "time_from_onset_s"],
                sort=True,
                observed=True,
            )["dff"]
            .mean()
            .reset_index()
        )
        for genotype in ("WT", "KO"):
            color = GENOTYPE_COLORS[genotype]
            genotype_traces = slice_traces[slice_traces["genotype"] == genotype]
            for _, one_slice in genotype_traces.groupby("slice_id", sort=True):
                trace_ax.plot(
                    one_slice["time_from_onset_s"],
                    one_slice["dff"],
                    color=color,
                    alpha=0.19,
                    linewidth=0.8,
                )
            center = (
                genotype_traces.groupby("time_from_onset_s", sort=True)["dff"]
                .median()
                .reset_index()
            )
            n_slices = genotype_traces["slice_id"].nunique()
            if not center.empty:
                trace_ax.plot(
                    center["time_from_onset_s"],
                    center["dff"],
                    color=color,
                    linewidth=2.4,
                    label=f"{genotype} median ({n_slices} slices)",
                )
        trace_ax.axvline(0, color="#D1495B", linestyle="--", linewidth=1.25)
        trace_ax.axhline(0, color="#AAB2BF", linewidth=0.7)
        trace_ax.set_title(f"{COHORT_LABELS[cohort]} — slice-mean traces", loc="left", weight="bold")
        trace_ax.set_xlabel("Time from NE onset (s)")
        trace_ax.set_ylabel("Contextual ΔF/F₀")
        trace_ax.legend(frameon=False, fontsize=9)

        paired = paired_slice_values(slice_metrics, cohort)
        for _, values in paired.iterrows():
            auc_ax.plot([0, 1], [values["WT"], values["KO"]], color="#AAB2BF", alpha=0.7, linewidth=1)
            auc_ax.scatter(0, values["WT"], color=GENOTYPE_COLORS["WT"], s=34, zorder=3)
            auc_ax.scatter(1, values["KO"], color=GENOTYPE_COLORS["KO"], s=34, zorder=3)
        if not paired.empty:
            medians = [paired["WT"].median(), paired["KO"].median()]
            auc_ax.plot([0, 1], medians, color="#111827", linewidth=3.2, zorder=4)
            auc_ax.scatter([0, 1], medians, color="#111827", s=58, zorder=5)
            difference = paired["ko_minus_wt"].median()
            auc_ax.text(
                0.02,
                0.98,
                f"{len(paired)} paired slices\nmedian KO−WT = {difference:.3g}",
                transform=auc_ax.transAxes,
                va="top",
                fontsize=9,
                bbox={"facecolor": "#F3F6FA", "edgecolor": "none", "pad": 6},
            )
        else:
            auc_ax.text(0.5, 0.5, "No slices contain both genotypes", ha="center", va="center")
        auc_ax.set_xticks([0, 1], ["WT", "KO"])
        auc_ax.set_xlim(-0.35, 1.35)
        auc_ax.set_ylabel("NE ΔAUC (post − pre)\nΔF/F₀ · s")
        auc_ax.set_title("Paired slice means (ROIs averaged first)", loc="left", weight="bold")
        auc_ax.grid(axis="y", color="#E7EBF0", linewidth=0.8)

    fig.suptitle(
        "H1R astrocyte calcium responses: hierarchy-aware secondary analysis",
        x=0.07,
        ha="left",
        fontsize=16,
        weight="bold",
        color="#172033",
    )
    fig.text(
        0.07,
        0.015,
        "Thin traces / paired lines = slices after averaging ROIs  •  0.71 Hz  •  "
        "No animal IDs in deposit: descriptive only, no animal-level inference",
        fontsize=9,
        color="#5B6472",
    )
    fig.tight_layout(rect=(0.04, 0.045, 0.99, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def make_orthogonal_figure(
    roi_metrics: pd.DataFrame,
    specification_grid: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot offset-free paired effects and the exploratory multiverse."""

    cohorts = [
        cohort for cohort in COHORT_LABELS
        if cohort in set(roi_metrics["cohort"])
    ]
    fig, axes = plt.subplots(
        len(cohorts), 2, figsize=(12.5, 4.2 * len(cohorts)), squeeze=False
    )
    fig.patch.set_facecolor("#F7F9FC")
    for row, cohort in enumerate(cohorts):
        paired_ax, grid_ax = axes[row]
        for axis in (paired_ax, grid_ax):
            axis.set_facecolor("white")
            axis.spines[["top", "right"]].set_visible(False)

        paired = common_scale_paired_values(roi_metrics, cohort)
        for _, values in paired.iterrows():
            paired_ax.plot(
                [0, 1], [values["WT"], values["KO"]],
                color="#AAB2BF", alpha=0.75, linewidth=1,
            )
            paired_ax.scatter(0, values["WT"], color=GENOTYPE_COLORS["WT"], s=34)
            paired_ax.scatter(1, values["KO"], color=GENOTYPE_COLORS["KO"], s=34)
        paired_ax.axhline(0, color="#D5DAE1", linewidth=0.8)
        paired_ax.set_xticks([0, 1], ["WT", "KO"])
        paired_ax.set_xlim(-0.35, 1.35)
        paired_ax.set_ylabel("Raw median response /\npooled slice baseline MAD")
        paired_ax.set_title(
            f"{COHORT_LABELS[cohort]}\nOffset-free paired amplitude",
            loc="left", weight="bold",
        )
        paired_ax.text(
            0.02, 0.98,
            f"{int((paired['ko_minus_wt'] > 0).sum())}/{len(paired)} positive slices\n"
            f"median KO−WT = {paired['ko_minus_wt'].median():.3g}",
            transform=paired_ax.transAxes, va="top", fontsize=9,
            bbox={"facecolor": "#F3F6FA", "edgecolor": "none", "pad": 6},
        )

        rows = specification_grid[specification_grid["cohort"] == cohort]
        jitter = np.linspace(-0.10, 0.10, len(rows))
        grid_ax.scatter(
            np.zeros(len(rows)) + jitter,
            rows["common_scale_mean_effect"],
            color="#386CB0", alpha=0.65, s=25,
        )
        grid_ax.scatter(
            np.ones(len(rows)) + jitter,
            rows["within_roi_z_mean_effect"],
            color="#E67E22", alpha=0.65, s=25,
        )
        grid_ax.axhline(0, color="#B42318", linestyle="--", linewidth=1.1)
        grid_ax.set_xticks(
            [0, 1], ["Common-scale\namplitude", "Within-ROI\nrobust z"]
        )
        grid_ax.set_xlim(-0.35, 1.35)
        grid_ax.set_ylabel("Mean paired KO−WT effect\nacross slices")
        grid_ax.set_title(
            "Sensitivity grid (36 specifications)", loc="left", weight="bold"
        )
        grid_ax.grid(axis="y", color="#E7EBF0", linewidth=0.8)

    fig.suptitle(
        "H1R offset-free robustness: alternative response metric and specification sensitivity",
        x=0.055, ha="left", fontsize=15, weight="bold", color="#172033",
    )
    fig.text(
        0.055, 0.012,
        "Each dot at right is one audit-defined specification; this is exploratory, "
        "not preregistered or animal-level inference.",
        fontsize=9, color="#5B6472",
    )
    fig.tight_layout(rect=(0.03, 0.04, 0.99, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/h1r_astrocytes_v1.csv.gz"),
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("results/h1r_astrocyte_summary.json")
    )
    parser.add_argument(
        "--roi-metrics", type=Path, default=Path("results/h1r_roi_delta_auc.csv")
    )
    parser.add_argument(
        "--slice-metrics", type=Path, default=Path("results/h1r_slice_delta_auc.csv")
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("figures/h1r_astrocyte_secondary_analysis.png"),
    )
    parser.add_argument(
        "--orthogonal-figure",
        type=Path,
        default=Path("figures/h1r_orthogonal_validation.png"),
    )
    parser.add_argument(
        "--specification-results",
        type=Path,
        default=Path("results/h1r_orthogonal_specifications.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.input)
    roi_metrics, slice_metrics, trace_data, exclusions = analyze_ne_responses(data)
    specification_grid = build_specification_grid(data)
    summary = build_summary(
        args.input,
        data,
        roi_metrics,
        slice_metrics,
        exclusions,
        specification_grid=specification_grid,
    )

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.roi_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.slice_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.specification_results.parent.mkdir(parents=True, exist_ok=True)
    roi_metrics.to_csv(args.roi_metrics, index=False, float_format="%.10g")
    slice_metrics.to_csv(args.slice_metrics, index=False, float_format="%.10g")
    specification_grid.to_csv(
        args.specification_results, index=False, float_format="%.10g"
    )
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    make_figure(trace_data, slice_metrics, args.figure)
    make_orthogonal_figure(
        roi_metrics, specification_grid, args.orthogonal_figure
    )

    print(f"Analyzed {len(roi_metrics)} NE ROIs across {roi_metrics['slice_id'].nunique()} slices")
    for cohort in sorted(roi_metrics["cohort"].unique()):
        paired = paired_slice_values(slice_metrics, cohort)
        median = paired["ko_minus_wt"].median() if not paired.empty else float("nan")
        print(f"  {cohort}: {len(paired)} paired slices; median KO-WT delta AUC={median:.6g}")
    print(
        f"Wrote {args.summary}, {args.roi_metrics}, {args.slice_metrics}, "
        f"{args.specification_results}, {args.figure}, and {args.orthogonal_figure}"
    )


if __name__ == "__main__":
    main()
