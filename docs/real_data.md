# Real-data methods and provenance

CADENCE contains two deliberately separate real-data paths:

1. a reproducible secondary analysis of public mouse cortical astrocyte H1R
   recordings; and
2. an out-of-domain NWB ingestion/QC check on a public zebrafish neuronal
   recording.

Neither path validates the synthetic controller.

## H1R mouse astrocyte dataset

The source is the Dryad deposit accompanying Taylor et al., *PLOS Biology*
(2025):

- [article DOI 10.1371/journal.pbio.3003376](https://doi.org/10.1371/journal.pbio.3003376)
- [dataset DOI 10.5061/dryad.2280gb64x](https://doi.org/10.5061/dryad.2280gb64x)
- [authors' analysis code, Zenodo 16809849](https://doi.org/10.5281/zenodo.16809849)

CADENCE uses two small deposited MATLAB files from Dryad version `394734`:

| Dryad file ID | File | SHA-256 | Role |
| --- | --- | --- | --- |
| `4342955` | `Fig5_H1RKO_NE.mat` | `45c7836bcafda6369a8addf436c2838c04ab61c53d3881fe52416821fe5e8116` | NE-only cohort, 5 paired slices |
| `4342954` | `Fig5_H1RKO_NE_postLowHA.mat` | `d39c8e8453996d5f3abdc42eed0b4ea4b6485b5e142719d4128f9683972621eb` | NE after low histamine, 7 paired slices |

The raw files are ignored by Git. The exporter verifies both hashes before
loading them, so a wrong same-named file cannot acquire valid-looking
provenance.

## Versioned derived data

MATLAB v7.3 tables are converted into a compact long-form file:

- `data/processed/h1r_astrocytes_v1.csv.gz`
- 67,595 samples
- derived SHA-256
  `4172cdb2eb5d3fa3b7b53648e6676a02b0314b47afcc73b81f03936ff5c2a7f6`
- explicit source DOI, Dryad version/file IDs, source hashes, cohort, stimulus,
  slice, genotype, ROI, onset, sampling rate, and raw fluorescence

The gzip stream is deterministic, and CI opens the committed export, verifies
its hash, reruns the analysis, and compares the complete summary with the
versioned result.

## Re-export from the sources

Download the two exact files above from the [Dryad record](https://doi.org/10.5061/dryad.2280gb64x)
into `data/real/`, then run from the repository root:

```powershell
Get-FileHash data\real\Fig5_H1RKO_NE.mat -Algorithm SHA256
Get-FileHash data\real\Fig5_H1RKO_NE_postLowHA.mat -Algorithm SHA256

matlab -batch "addpath('scripts'); export_h1r_astrocytes"
python src/analyze_h1r_astrocytes.py
```

The exporter uses MATLAB because the source files contain MATLAB table objects
whose v7.3 HDF5 representation is not reliably decoded by generic Python HDF5
readers.

## Response analysis

The analysis follows the relevant operations in the authors' notebook:

1. sample at 0.71 Hz;
2. add a fluorescence offset of 100,000;
3. use the median from 70 to 10 seconds before NE onset as `F0`;
4. when the full baseline is unavailable, use the source-specific fallback:
   40 to 10 seconds before onset for the 2023 file and 30 to 10 seconds for the
   2025 file;
5. compute contextual `(F - F0) / F0`;
6. compute post-minus-pre trapezoidal AUC with 21 samples on each side of onset.

At 0.71 Hz, 21 samples span 20 intervals, or 28.17 seconds. The repository calls
these the authors' nominal 30-second windows and records the effective duration
explicitly.

ROIs are first averaged within each slice and genotype. Wild-type and knockout
slice means are then paired only where both occur in the same deposited slice.
The two protocol/sensor cohorts remain separate.

## Current descriptive result

The analysis includes 147 NE ROIs across 13 slices and excludes none:

| Cohort | Paired slices | Positive KO−WT differences | Median KO−WT ΔAUC |
| --- | ---: | ---: | ---: |
| NE only (2023, jRGECO) | 5 | 5/5 | +0.349 |
| NE after low HA (2025, GCaMP) | 7 | 6/7 | +0.324 |

One 2025 slice has a raw fluorescence scale far below the cohort median. It is
flagged and retained to reproduce the deposit rather than excluded after seeing
the result.

## Offset-free robustness analysis

The primary ΔAUC follows the source notebook and therefore inherits its
`+100,000` fluorescence offset. CADENCE also runs an analysis that does not use
that offset, contextual ΔF/F₀, or AUC. It reuses the same recordings and is an
alternative measurement-scale check, not an independent replication or assay:

1. compute each ROI's raw median response from nominal 30-second pre/post
   windows;
2. aggregate ROIs to each slice/genotype with the median;
3. form the within-slice KO−WT contrast; and
4. divide by one robust baseline scale—`1.4826 × MAD`—pooled across both
   genotypes in that slice.

| Cohort | Mean common-scale KO−WT effect | Positive slices |
| --- | ---: | ---: |
| NE only | +11.663 baseline-MAD units | 5/5 |
| NE after low HA | +3.559 | 6/7 |

An audit-defined grid varies baseline start (40/60/70 s), response window
(15/30/45 s), temporal mean/median, and ROI-to-slice mean/median: 36
specifications per cohort. The mean common-scale effect remains positive in all
36 for both cohorts. This grid was defined after inspection and is explicitly
exploratory, not preregistered.

A stricter sensitivity standardizes every ROI by its own baseline MAD before
slice aggregation. Its mean paired effect is positive in both cohorts, including
under leave-one-slice-out checks, but individual-slice directions weaken to 3/5
and 5/7. The defensible conclusion is a larger response amplitude on average,
not universally improved signal-to-noise.

The full grid is versioned in
[`results/h1r_orthogonal_specifications.csv`](../results/h1r_orthogonal_specifications.csv)
and visualized in
[`figures/h1r_orthogonal_validation.png`](../figures/h1r_orthogonal_validation.png).

## Limits on interpretation

- Animal IDs are absent from the deposited tables, so slices cannot be mapped
  to animals and no animal-level p-value is justified.
- ROIs are subsamples, not independent animals.
- The cohorts use different indicators and protocol contexts and are not pooled.
- ΔAUC magnitude depends on the authors' additive fluorescence offset; the
  offset-free analysis supports the average amplitude direction but the stricter
  per-ROI signal-to-noise result is less slice-consistent.
- This is a secondary analysis of an already published experiment, not a
  prospective validation of CADENCE's synthetic states or control equation.

Machine-readable details are in
[`results/h1r_astrocyte_summary.json`](../results/h1r_astrocyte_summary.json).

## Zebrafish NWB ingestion check

`real_data.py` can load one asset from
[DANDI:001076](https://dandiarchive.org/dandiset/001076). The recording contains
pan-neuronal larval zebrafish calcium imaging, not glia. CADENCE uses it only to
exercise NWB discovery, segmentation-QC filtering, timestamps, and per-ROI
normalization. It is not evidence for the H1R result or controller.

Raw source files remain outside version control. Reuse must follow the source
records' terms, and any report must cite the article, dataset, and analysis code.
