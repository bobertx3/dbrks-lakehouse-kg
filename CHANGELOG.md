# Changelog

Adheres to [Semantic Versioning](https://semver.org/) and [Keep a Changelog](https://keepachangelog.com/).

## [0.2.0] — 2026-07-16

Merged the nvidia + cpu editions into one kit and de-vendored the pipeline.

### Changed
- **One repo, two compute tiers.** Merged `lakehouse-kg-starter` (nvidia) and
  `lakehouse-kg-starter-cpu` — the `lakehouse_kg/` package trees were byte-identical.
  GPU is now an optional tier via `requirements-gpu.txt` (RAPIDS cuGraph, notebook 07)
  on top of the CPU base; all three scale notebooks (06 networkx, 07 cuGraph,
  08 distributed GraphFrames) ship in the one kit.
- **De-vendored the agentic pipeline.** Deleted the local `lakehouse_kg/` package
  (agents/domains/generators/genie/orchestrator) — it duplicated the now-canonical
  standalone `agentic-triplets` package (Wave 0.5). Notebooks `%pip install`
  `agentic-triplets` and import from it. One source of truth.
- Added a parameterized Databricks Asset Bundle (`databricks.yml`) and per-tier
  `requirements.txt` / `requirements-gpu.txt`.
- LLM endpoint defaults updated to a current Foundation Model.
- gold_triplets predicate-casing comment aligned with kg-contracts (producer choice,
  not "lower snake_case").
- Doc-alignment header (Interconnectivity for Agent Context — Solution 3).

### Retired
- `lakehouse-kg-starter-cpu` repo (superseded by this merged kit's CPU tier).

## [0.1.0]

Initial starter kit (nvidia + cpu editions): 8-agent pipeline, gold_triplets contract,
Genie space + UC traversal/analytics functions, CPU/GPU/distributed batch algorithms.
