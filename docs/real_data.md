# Real glial calcium recordings: candidate datasets and ingestion plan

**Status: nothing has been downloaded or ingested.** Every number in the README
comes from synthetic data with known ground truth. This document records the
survey so the wet-lab step starts from evidence rather than a search engine.

---

## What CADENCE actually needs

The pipeline is deliberately undemanding about format. It needs a long-form table:

| column | meaning |
|---|---|
| `condition` | experimental group (the intact/blocked analogue) |
| `trace_id` | one identifier per cell/ROI — the independent unit for bootstrapping |
| `time_s` | seconds, uniformly sampled |
| `calcium` | dF/F or z-scored fluorescence |
| `true_state` | **optional**; only used for scoring, never for fitting |

Two properties matter far more than the file format:

1. **A perturbation that removes the feedback pathway.** Without a blocked-style
   condition there is no kill-shot, and the central claim becomes untestable.
2. **Sampling fast relative to the sensor.** The kinetic model estimates `tau` in
   frames; the REFRACTORY precision ceiling is set by how much the indicator
   blurs the switch. At 2 Hz with GCaMP this is already the binding constraint.

---

## Candidates, best fit first

### 1. Dryad — cortical astrocyte H1R across sleep/wake ⭐ best scientific match

`doi:10.5061/dryad.2280gb64x`

**Why it fits:** compares **wild-type vs H1R-knockout** mice. That is a genuine
real-world analogue of the intact/blocked design — a genetic removal of a
signalling pathway — which is exactly what the kill-shot needs. Contains ex vivo
two-photon Ca²⁺ imaging plus fibre photometry.

**The practical problem:** ~134 GB total. The calcium-bearing files are large:

| file | size | notes |
|---|---|---|
| `Fig2_H1Rpharm_data.mat` | 13.3 GB | pharmacological manipulation |
| `Fig6_KO_jRGECO.pkl` | 7.3 GB | KO calcium |
| `Fig6_Fig7_WT_grabAd.pkl` | 6.8 GB | WT |
| `Fig4_H1RKO_NE_data.mat` | 2.0 GB | KO + norepinephrine |
| `Fig5_H1RKO_NE.mat` | **0.6 MB** | smallest calcium-bearing file |
| `Fig3_*.csv` | ~6 KB | histology quantification, **not traces** |

**Cheapest useful entry point:** `Fig5_H1RKO_NE.mat` (620 KB) — small enough to
inspect structure immediately, but **KO-only**, so it cannot supply the WT
comparison on its own. A genuine WT-vs-KO contrast means pulling GB-scale files.

### 2. AQuA — ex vivo and in vivo GCaMP astrocyte datasets

<https://github.com/yu-lab-vt/AQuA> — Wang et al., *Nature Neuroscience* 2019.

Ships downloadable ex vivo and in vivo GCaMP astrocyte recordings plus synthetic
sets with ground truth. **Easiest starting point** for validating that the
ingestion path works on real fluorescence. No built-in pathway-blockade contrast,
so it exercises Steps 1b–3 but not the kill-shot.

### 3. aqua-py-analysis

<https://github.com/Achilleas/aqua-py-analysis> — MIT licensed, three astrocyte
datasets (behaviour, vibrissa stimulation), distributed as ZIPs.

### 4. DANDI Archive

<https://registry.opendata.aws/dandiarchive/> — NWB-format neurophysiology.
Worth a targeted search for astrocyte GCaMP with a pharmacological arm.

---

## Recommended sequence

1. **AQuA first, for plumbing.** Confirm the kinetic model fits a real GCaMP trace
   and that `tau` estimation returns something physically sensible. Low download
   cost, no scientific claim attached.
2. **Then Dryad WT-vs-KO, for the science.** Only this one can test the kill-shot
   on real tissue. Requires a deliberate decision about the GB-scale download.

## Expected difficulties (better to name them now)

- **`tau` will differ per indicator and per prep.** It is already estimated by
  profile likelihood rather than assumed, so this should be handled — but it is
  the first thing to sanity-check on real data.
- **Real traces have motion artefacts, bleaching and drift.** The synthetic
  generator has none. Detrending will likely be needed *before* the kinetic model,
  and detrending can itself distort the decay tail the model relies on.
- **There is no `true_state` on real data.** Steps 1b–2 lose their scoring
  target, so validation has to shift onto the Step 3 contrast (WT vs KO `b1`) and
  the transition-matrix structure.
- **ROI definition is a judgement call.** Astrocyte calcium is famously
  compartmentalised; whole-cell ROIs may average away the very transitions the
  model is trying to detect. AQuA exists precisely because of this.

## Licensing

Check per dataset before redistributing anything. aqua-py-analysis is MIT. The
Dryad entry is under Dryad's reuse terms. AQuA datasets should be cited to
Wang et al. 2019. **No dataset should be committed into this repo** — add a
loader plus a download script, keep `data/` gitignored as it already is.
