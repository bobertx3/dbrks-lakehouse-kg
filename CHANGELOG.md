# Changelog

Adheres to [Semantic Versioning](https://semver.org/) and [Keep a Changelog](https://keepachangelog.com/).

## [0.3.0] — 2026-07-29

Prepared the kit as a Databricks Solution Accelerator and switched the
distributed-CPU engine to a serverless-capable one.

### Changed
- **Notebook 08 now uses `graphframes-serverless`** instead of JVM Apache
  GraphFrames. It runs on serverless compute and on any classic cluster, needs
  no ML runtime and no cluster library install, and computes true Louvain
  communities (the JVM path used label propagation as a stand-in).
- `cluster_specs/classic_graphframes.json` moved from an ML runtime to a
  standard runtime, since GraphFrames Serverless installs with `%pip`.
- `requirements.txt` adds `graphframes-serverless` and no longer pins the
  pipeline to a private git URL.
- Rewrote `README.md` in Simplified Technical English. Added an engine-selection
  table (networkx / GraphFrames Serverless / cuGraph) and Solution Accelerator
  badges.

### Licensing
- Relicensed under the Databricks License (was Apache-2.0) and added a `NOTICE`
  file, matching Databricks Solution Accelerator conventions.
- Removed internal references (planning-doc links, internal codenames, private
  repository URLs) from all files.

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

### Retired
- `lakehouse-kg-starter-cpu` repo (superseded by this merged kit's CPU tier).

## [0.1.0]

Initial starter kit (nvidia + cpu editions): 8-agent pipeline, gold_triplets contract,
Genie space + UC traversal/analytics functions, CPU/GPU/distributed batch algorithms.
