# Release + Solution Accelerator submission checklist

This checklist covers four related repositories. One is an open-source library.
Three are Databricks Solution Accelerators. Work through them in order, because
the accelerators depend on the library being publicly installable.

```
lakehouse-kg (accelerator)
├── agentic-triplets   (accelerator)  → produces gold_triplets
├── kg-contracts       (accelerator)  → the shared schema both sides import
└── graphframes-serverless (Apache-2.0 library)  → used by notebook 08
```

## Licensing decision (agreed)

| Repository | License | Target |
|---|---|---|
| `graphframes_serverless` (SparkGraph) | Apache-2.0 | Public open-source library (GitHub + PyPI) |
| `kg-contracts` | Databricks License | Official Databricks Solution Accelerator |
| `agentic-triplets` | Databricks License | Official Databricks Solution Accelerator |
| `lakehouse-kg` | Databricks License | Official Databricks Solution Accelerator |

## What this branch already did (in-repo, done)

- **lakehouse-kg**: notebook 08 rewritten to use `graphframes-serverless` (runs on
  serverless, true Louvain); README rewritten in Simplified Technical English with
  an engine-selection table (networkx / GraphFrames Serverless / cuGraph);
  relicensed to the Databricks License; added `NOTICE`; standard-runtime cluster
  spec; internal references removed; badges added.
- **graphframes_serverless**: internal references removed; install instructions
  point to PyPI; PyPI packaging metadata (classifiers, keywords, readme) added.
  Stays Apache-2.0.
- **kg-contracts**: relicensed to the Databricks License; added `NOTICE`; internal
  references removed from source, README, tests, and the CI config; badges added.
- **agentic-triplets**: relicensed to the Databricks License; added `NOTICE`;
  internal references removed; badges added.

## Action items — YOU must do these (outside the repos)

These need your credentials or account access. I cannot do them.

### 1. Release `graphframes-serverless` (do this first — it unblocks the rest)
- [ ] Make the `graphframes_serverless` GitHub repository **public**.
- [ ] Publish the package to **public PyPI** as `graphframes-serverless`
      (`python -m build` then `twine upload dist/*`). The three accelerators and
      lakehouse-kg notebook 08 install it by this name.
- [ ] Confirm `pip install graphframes-serverless` works from a clean environment.

### 2. Confirm the dependency names the accelerators install
- [ ] `agentic-triplets` and `kg-contracts` are installed by name (no git URL).
      They must be publicly installable before an accelerator that pip-installs
      them will run for an external user. Decide for each: publish to PyPI, or
      submit into `databricks-industry-solutions` first and reference from there.

### 3. Regenerate the architecture diagram
- [ ] `docs/architecture.drawio` label was updated to "GraphFrames Serverless".
      Re-export `docs/architecture.png` from draw.io (I cannot render it here).

### 4. Official Solution Accelerator submission (Databricks-internal process)
- [ ] Confirm the current submission process for `databricks-industry-solutions`
      (it changes — check the internal contributor guide / #solution-accelerators).
- [ ] Verify the exact `LICENSE` and `NOTICE` text against a freshly cloned
      current accelerator. This branch used the text from an existing accelerator
      repo; confirm it is still current before submitting.
- [ ] Copyright is assigned to **Databricks, Inc.** in the license and notice.
      Confirm that matches the contribution agreement for the org.
- [ ] Add any org-required files not present yet (for example `CONTRIBUTING.md`,
      `SECURITY.md`) to match the org template.
- [ ] Confirm each accelerator runs end-to-end on a clean workspace with only
      synthetic data (no private dependencies, no internal endpoints).

## Notes and residual risks

- **License review.** The Databricks License forbids derivative works without a
  separate agreement. Confirm that this is the intended license for code you want
  others to build on. It is more restrictive than the Apache-2.0 you use for
  SparkGraph — that difference is deliberate here.
- **Circular naming.** `agentic-triplets` and `lakehouse-kg` reference
  `kg-contracts` and each other. Once the public homes are decided, do one pass to
  point every cross-reference at the final public URLs.
- **The submission itself is not an in-repo change.** These repos are now prepared
  to spec, but acceptance into `databricks-industry-solutions` is a review process
  that happens outside them.
