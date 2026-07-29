# Lakehouse Knowledge Graph Starter Kit

[![Databricks Solution Accelerator](https://img.shields.io/badge/Databricks-Solution%20Accelerator-FF3621?logo=databricks&logoColor=white)](https://www.databricks.com/solutions/accelerators)
![Unity Catalog](https://img.shields.io/badge/Unity%20Catalog-enabled-00A972)
![Serverless](https://img.shields.io/badge/Serverless-ready-1B3139)

Build a knowledge graph on Databricks. You do not need an external graph
database.

An agentic pipeline reads your Delta tables. It finds entities and generates the
relationships between them. This kit then makes the graph easy to query. You can
query it in three ways:

- **Genie** — ask questions in natural language.
- **Unity Catalog SQL functions** — call the graph from SQL or from an agent tool.
- **Batch graph algorithms** — compute centrality and communities.

Everything is built on one contract: a `gold_triplets` Delta table. Any team
that can produce this table gets the full query layer. The team can use the
pipeline in this kit or its own ETL.

![Architecture](docs/architecture.png)

## What is in the kit

| Layer | Files |
|---|---|
| Triplet generation | the `agentic-triplets` package (an 8-agent pipeline with pluggable domain packs), installed through `requirements.txt` |
| Data model | `sql/01_gold_triplets.sql`, `sql/02_dataset_registry.sql` |
| Genie space | `notebooks/03_derive_graph_views.py`, `notebooks/04_create_genie_space.py` |
| Graph SQL functions | `sql/03_traversal_functions.sql`, `sql/04_analytics_functions.sql`, `notebooks/05_register_uc_functions.py` |
| Batch algorithms | `notebooks/06_graph_algorithms.py`, `notebooks/07_cugraph_gpu.py`, `notebooks/08_graphframes_distributed.py` |
| Worked example | a synthetic fraud dataset (`notebooks/01_synthetic_data.py`) and a fraud domain pack |
| Diagram | `docs/architecture.drawio`, `docs/architecture.png` |

## Choose a graph algorithm engine

Notebooks 06, 07, and 08 all compute the same result. Each one writes the same
two tables (`entity_centrality` and `entity_communities`). The serving SQL
functions read either output. Pick the engine that matches your data size and
your available compute.

| Engine | Notebook | Compute | Best for | Notes |
|---|---|---|---|---|
| **networkx** (driver) | `06_graph_algorithms.py` | Serverless or any cluster | Graphs up to about 5M edges | Pure Python on the driver. The default. Includes approximate betweenness. |
| **GraphFrames Serverless** (distributed CPU) | `08_graphframes_distributed.py` | Serverless or any cluster | Large graphs, no GPU | Distributed Spark. Runs on serverless. Computes true Louvain. Does not compute betweenness. |
| **NVIDIA RAPIDS cuGraph** (GPU) | `07_cugraph_gpu.py` | GPU cluster | Very large graphs, fastest | Scales to hundreds of millions of edges. Needs a GPU cluster (`cluster_specs/gpu_cugraph.json`). |

Start with notebook 06. Move to notebook 08 or 07 when your graph is too large
for the driver. Notebook 06 stops with a clear message when the graph is above
its edge limit. It then tells you to use notebook 07 or 08.

### About the two distributed engines

- **GraphFrames Serverless** ([`graphframes-serverless`](https://pypi.org/project/graphframes-serverless/))
  is a pure-Python graph library. It has the GraphFrames API but needs no JVM
  library install. It runs on Databricks serverless compute. Notebook 08
  installs it with `%pip`.
- **NVIDIA RAPIDS cuGraph** is the GPU option. It is the fastest engine at large
  scale. Install it with `requirements-gpu.txt` on a RAPIDS GPU cluster. Skip it
  if you have no GPU. Everything else runs on CPU without change.

## Quick start

Deploy the bundle with `databricks bundle deploy`, or clone the repo into your
workspace. Then run the notebooks in order. The notebooks install their
dependencies with `%pip`. All notebooks use widgets for parameters. The bundle
sets `catalog`, `schema`, `domain`, and `llm_endpoint`. The defaults target
`main.knowledge_graph`.

1. **`01_synthetic_data`** — generate the fraud example. Skip this if you have your own tables.
2. **`02_triplet_pipeline`** — run the pipeline. Choose a domain pack (`generic` or `fraud`). This writes `gold_triplets`.
3. **`03_derive_graph_views`** — build the graph views over `gold_triplets`.
4. **`04_create_genie_space`** — create a Genie space over those views.
5. **`05_register_uc_functions`** — register 8 Unity Catalog SQL functions.
6. **`06_graph_algorithms`** — compute PageRank, degree, betweenness, communities, and components (networkx driver path).
7. **`07_cugraph_gpu`** — the GPU alternative to notebook 06 (NVIDIA RAPIDS cuGraph).
8. **`08_graphframes_distributed`** — the distributed-CPU alternative to notebook 06 (GraphFrames Serverless).

Run only one of notebooks 06, 07, or 08. They produce the same tables.

## The triplet contract

`gold_triplets` holds one row for each relationship. The first 8 columns are the
shared contract. The `properties` and `created_at` columns are an optional
addition from this kit.

| Column | Type | Meaning |
|---|---|---|
| `subject_id` / `subject_type` | STRING | The source entity and its type |
| `predicate` | STRING | The relationship type (for example, `OWNS_ACCOUNT`) |
| `object_id` / `object_type` | STRING | The target entity and its type |
| `confidence` | DOUBLE | A value from 0.0 to 1.0 |
| `source_agent` / `source_method` | STRING | The agent and the technique that produced the row |
| `properties` | STRING | JSON metadata (scores, reasoning, amounts) |
| `created_at` | TIMESTAMP | The time of generation |

`dataset_registry` holds one row for each dataset you onboard. This lets many
graphs share one deployment of the query layer.

## The agentic pipeline

The pipeline uses 8 agents. Each agent asks a different question of the data.

| Phase | Agent | Question |
|---|---|---|
| Discovery | Schema discovery | What datasets and columns exist? |
| Discovery | Entity resolution | What are the key entities? |
| Generation | Relationship (rule-based) | What direct links exist? |
| Generation | Statistical | What anomalies stand out? |
| Generation | ML clustering | What hidden clusters exist? |
| Generation | LLM semantic | What can a foundation model infer? |
| Enrichment | Graph topology | What hubs, bridges, and communities appear? |
| Validation | Validation | Are the triplets consistent and deduplicated? |

```python
from agentic_triplets import PipelineConfig, AgenticOrchestrator
from agentic_triplets.domains import get_domain_pack

config = PipelineConfig(
    catalog="main",
    schema="knowledge_graph",
    domain=get_domain_pack("generic"),   # or "fraud"
    llm_endpoint="databricks-claude-sonnet-4-6",
)
summary = AgenticOrchestrator(spark, config).run()
```

### Domain packs

All domain knowledge lives in a domain pack, not in the agents. A domain pack
holds the entity types, the predicate vocabulary, the LLM prompts, and the Genie
view definitions. Two packs ship with the pipeline:

- **`generic`** — neutral defaults that work on any schema.
- **`fraud`** — the full worked example, with fraud entity types and predicates.

To adapt the kit, build a new domain pack and pass it to `PipelineConfig`. You
do not need to change any agent code.

## Unity Catalog functions

Notebook `05_register_uc_functions.py` registers 8 SQL functions. Each one
returns a table. You can call them from SQL, from Genie, or as an agent tool.

- **Traversal** (read `gold_triplets` directly): `neighbors`, `khop`, `connection_path`, `subgraph_edges`.
- **Analytics** (read the precomputed tables): `cluster_of`, `members_of_cluster`, `top_central_entities`, `shared_community`.

The pattern is simple. Run the costly algorithms once in batch (notebook 06, 07,
or 08). Save the results to `entity_centrality` and `entity_communities`. Then
serve the results through fast function lookups.

## Notes

- The k-hop traversal uses fixed-hop join chains. It does not use `WITH
  RECURSIVE`. Always query the k-hop views with a `start_id` filter.
- Notebook 06 runs on the driver. It stops with a clear message when the graph
  is above its edge limit (5M by default). It then points you to notebook 07 or 08.
- Notebook 07 needs a GPU cluster. Do not run it on serverless or on a CPU cluster.
- Notebook 08 runs on serverless compute or any cluster.
- All data, names, and identifiers in this repo are synthetic. No real data ships here.
