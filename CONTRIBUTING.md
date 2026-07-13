# Contributing to CADENCE

CADENCE is a research prototype. Contributions should make the code, evidence,
or interpretation more reproducible—not merely make the repository busier.

## Before proposing a change

1. State whether the change affects synthetic benchmarking, public-data analysis,
   documentation, or infrastructure.
2. Keep synthetic results, exploratory secondary analyses, and biological claims
   clearly separated. Do not describe a simulation result as a biological
   validation.
3. Do not commit raw source recordings. The versioned H1R export includes the
   required provenance and checksums; see [real-data methods](docs/real_data.md).
4. If a change affects a reported number, regenerate the owned result files and
   figures rather than editing them by hand.
5. Run the relevant checks from the repository root:

   ```bash
   python scripts/check_abstract_length.py
   python -m unittest discover -s tests -v
   python tests/test_pipeline.py
   ```

## Reporting an issue

Include the exact command, Python version, operating system, traceback or
unexpected output, and whether the versioned derived H1R export was present.
For a scientific concern, identify the claimed result, its experimental unit,
and the proposed interpretation boundary.

## Research integrity

Do not add a result to make the project appear more active or more conclusive
than the evidence allows. Keep generated artifacts traceable to code, preserve
negative or mixed results, and describe assistance and source-data reuse as
documented in [AI_ASSISTANCE.md](AI_ASSISTANCE.md).
