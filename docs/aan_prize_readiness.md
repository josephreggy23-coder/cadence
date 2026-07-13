# AAN Neuroscience Research Prize readiness

Checked against the public AAN materials in July 2026. This is a planning
document, not an application and not a promise of eligibility or an award.

## What the AAN actually evaluates

The [official prize page](https://www.aan.com/research/neuroscience-research-prize)
says that applicants must be enrolled in grades 9–12 at a U.S. secondary school.
Projects must be individual, original research and the written application must
be the applicant's original work. A formal laboratory is not required.

The first round evaluates only an abstract of at most 300 words for clarity,
clear methods, and succinct writing. Finalists are judged on:

1. direct relevance to neuroscience;
2. creativity and original problem-solving;
3. feasible scope, interpretation, significance, and treatment of pitfalls;
4. an organized report with readable, labelled figures and tables.

The application also requires a research report, bibliography, and electronic
signatures from a parent or guardian, a teacher, and a mentor. The public page
still shows the previous cycle as closed; dates for the next cycle were not
posted when this file was checked. Confirm dates and any updated rules directly
with AAN before preparing a submission.

## What recent winning work suggests

This is an evidence-based pattern, not an additional rule.

- The 2025 polymersome project combined machine-learning design with in-vitro
  verification ([official abstract](https://www.aan.com/msa/public/events/abstractdetails/61458)).
- The 2025 PCSK9 project combined a 2,992-compound computational screen with
  ELISA, uptake, flow-cytometry, and viability experiments, while still naming
  the need for in-vivo validation
  ([official abstract](https://www.aan.com/msa/Public/Events/AbstractDetails/61459)).
- AAN's 2021 announcement praised computational projects for real-world
  implications, technical depth, clear writing, and properly discussed
  significance ([official announcement](https://www.aan.com/PressRoom/Home/PressRelease/4882)).

The practical lesson for CADENCE is that polished software is not the scientific
result. A competitive entry needs a narrow neuroscience question, independent
biological evidence, uncertainty at the correct experimental level, and a clear
boundary between observation and assumption.

## Current evidence map

| Component | What it establishes | What it does not establish |
| --- | --- | --- |
| Synthetic generator | A known four-state plant with a known dwell-dependent exit law | That astrocytes have four such states or use this mechanism |
| Sensor-aware estimator | Whether an estimator can recover synthetic states despite indicator memory | Biological state identity in an unlabeled recording |
| Synthetic controller | Behaviour of a policy under the assumed plant and intervention equation | Safety, efficacy, or mechanism in tissue, animals, or people |
| DANDI 001076 loader | NWB ingestion and QC on a real neuronal zebrafish recording | Glial biology or the controller hypothesis |
| Dryad H1R analysis | Exploratory secondary analysis plus an offset-free alternative metric on the same real mouse astrocyte recordings | Independent replication, prospective validation of CADENCE, or animal-level inference when IDs are unavailable |

The most defensible project question at this stage is:

> Can a sensor-aware model distinguish simulated state dynamics from
> calcium-indicator memory, and under what assumptions does model-based control
> fail?

## Prize-blocking questions to resolve

Before applying, the student should ask `science@aan.com` for written guidance on:

- whether secondary analysis of an open dataset is eligible;
- how AAN interprets an "individual project" when the source dataset has its own
  authors and the code has received AI assistance;
- what AI-assisted coding is permitted and how it should be disclosed;
- the next deadline and current report/file limits;
- whether prior publication or public GitHub development affects eligibility.

## Scientific work still needed

1. Obtain animal identifiers for the H1R data or a dataset with enough animals
   for animal-level inference. ROIs are subsamples, not independent animals.
2. Freeze a prospective analysis plan before the next confirmatory dataset. The
   current H1R work is exploratory and cannot be called preregistered afterward.
3. Compare dwell history with a predictor computed from measured fluorescence.
   A positive slope on a state-derived accumulator currently shows increasing
   exit hazard, not a molecular calcium-load mechanism.
4. Add model-mismatch simulations, multiple seeds, heterogeneous traces,
   bleaching/drift, kinetic-grid sensitivity, and strictly held-out scoring.
5. Compare policies on an efficacy-cost frontier against tuned open-loop and
   threshold baselines. Intervention cost is currently in arbitrary units.
6. Treat the blocked-feedback result as a structural sanity check: the same
   coefficient disables both feedback and intervention in the simulated plant,
   so failure is guaranteed by the equation rather than independently observed.

## Student-owned submission checklist

- [ ] I can derive every equation and explain every code path used for a result.
- [ ] I reran the analysis myself from a fresh environment.
- [ ] I made and recorded the scientific choices, including exclusions.
- [ ] My mentor reviewed the biological interpretation and experimental unit.
- [ ] I wrote the application abstract and report in my own words.
- [ ] Every number in the abstract points to a script, result file, and figure.
- [ ] I distinguish exploratory analyses from prospective confirmation.
- [ ] I disclose data reuse, software tools, and AI assistance as AAN directs.
- [ ] Parent/guardian, teacher, and mentor signatures are arranged.

Official recipients are listed on AAN's
[Scientific Research Award Recipients](https://www.aan.com/research/scientific-research-award-recipients)
page.
