# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Batch graph analytics over gold_triplets (CPU / serverless-safe)
# MAGIC
# MAGIC Reads the `gold_triplets` edge list, builds a networkx graph **on the
# MAGIC driver**, and precomputes the analytics tables that the serving-layer UC
# MAGIC functions (`sql/04_analytics_functions.sql`) read at query time:
# MAGIC
# MAGIC | Output table | Columns |
# MAGIC |---|---|
# MAGIC | `entity_centrality` | entity_id, entity_type, pagerank, degree, betweenness (nullable) |
# MAGIC | `entity_communities` | entity_id, entity_type, community_id, community_size, component_id |
# MAGIC
# MAGIC (`component_id` is an extra column beyond the function contract — the UC
# MAGIC functions reference columns by name, so extras are harmless.)
# MAGIC
# MAGIC ### Algorithms
# MAGIC | Algorithm | Notes |
# MAGIC |---|---|
# MAGIC | PageRank | directed, damping 0.85 |
# MAGIC | Degree | undirected degree per entity |
# MAGIC | Betweenness | **sampled/approximate** (`k` pivot nodes), skipped above a node-count guard — exact betweenness is O(V·E) |
# MAGIC | Louvain communities | `networkx.community.louvain_communities`, fixed seed |
# MAGIC | Connected components | undirected, persisted as `component_id` |
# MAGIC
# MAGIC ### Scale guidance
# MAGIC This notebook is intentionally **pure Python on the driver** so it runs on
# MAGIC serverless compute with no cluster-config knobs. That works comfortably up to
# MAGIC roughly **~5M edges**. Beyond that, switch to one of the two scale
# MAGIC alternatives — both write the exact same two output tables:
# MAGIC `notebooks/07_cugraph_gpu.py` (NVIDIA RAPIDS cuGraph on a GPU cluster) or
# MAGIC `notebooks/08_graphframes_distributed.py` (GraphFrames Serverless,
# MAGIC distributed CPU). The `max_driver_edges` widget enforces this with a hard
# MAGIC stop rather than a silent OOM.

# COMMAND ----------

# MAGIC %pip install networkx --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "knowledge_graph")
dbutils.widgets.text("min_confidence", "0.0", "minimum triplet confidence to include (NULL confidence is kept)")
dbutils.widgets.text("max_driver_edges", "5000000", "hard stop: refuse to collect more edges than this to the driver")
dbutils.widgets.text("betweenness_max_nodes", "50000", "skip betweenness entirely above this node count (writes NULL)")
dbutils.widgets.text("betweenness_sample_k", "256", "number of pivot nodes for approximate betweenness")
dbutils.widgets.text("exclude_source_agents", "graph_topology,ml_clustering,statistical_analysis", "source_agents whose triplets are excluded from the input graph (comma-sep)")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
MIN_CONFIDENCE = float(dbutils.widgets.get("min_confidence"))
EXCLUDE_AGENTS = [a.strip() for a in dbutils.widgets.get("exclude_source_agents").split(",") if a.strip()]
MAX_DRIVER_EDGES = int(dbutils.widgets.get("max_driver_edges"))
BETWEENNESS_MAX_NODES = int(dbutils.widgets.get("betweenness_max_nodes"))
BETWEENNESS_SAMPLE_K = int(dbutils.widgets.get("betweenness_sample_k"))

FQ = CATALOG + "." + SCHEMA
GT = FQ + ".gold_triplets"
print("Input:  " + GT)
print("Output: " + FQ + ".entity_centrality, " + FQ + ".entity_communities")

# COMMAND ----------

# MAGIC %md ## 1. Load the edge list to the driver
# MAGIC
# MAGIC By default only *structural* triplets (FK joins, shared attributes, LLM
# MAGIC semantic links) feed the algorithms. The statistical, ML, and graph-topology
# MAGIC agents emit entity→annotation edges (pattern nodes, behavioral clusters,
# MAGIC community memberships) — feeding those back in makes the synthetic
# MAGIC annotation nodes dominate centrality instead of the real entities. Clear
# MAGIC `exclude_source_agents` to analyze the full annotated graph instead.

# COMMAND ----------

from pyspark.sql import functions as F

edges_sdf = (
    spark.table(GT)
    .where((F.col("confidence").isNull()) | (F.col("confidence") >= MIN_CONFIDENCE))
    .where(~F.col("source_agent").isin(EXCLUDE_AGENTS) if EXCLUDE_AGENTS else F.lit(True))
    .select("subject_id", "subject_type", "object_id", "object_type")
    .where(F.col("subject_id").isNotNull() & F.col("object_id").isNotNull())
)

n_edges = edges_sdf.count()
print("edges after confidence filter: {:,}".format(n_edges))
if n_edges == 0:
    raise ValueError(GT + " has no rows - run the triplet-generation notebooks (01-04) first.")
if n_edges > MAX_DRIVER_EDGES:
    raise ValueError(
        "gold_triplets has {:,} edges, above the {:,} driver limit. ".format(n_edges, MAX_DRIVER_EDGES)
        + "Use notebooks/07_cugraph_gpu.py (GPU cuGraph) or notebooks/08_graphframes_distributed.py "
        + "(GraphFrames Serverless, distributed CPU) - both write the same output tables."
    )

pdf = edges_sdf.toPandas()
print("collected {:,} edges to the driver".format(len(pdf)))

# COMMAND ----------

# MAGIC %md ## 2. Build the graph
# MAGIC
# MAGIC Directed graph for PageRank; undirected view for degree, betweenness,
# MAGIC Louvain, and connected components. An entity's type is taken from the first
# MAGIC triplet that mentions it (types should be consistent per entity anyway).

# COMMAND ----------

import networkx as nx
import numpy as np
import pandas as pd

type_pdf = pd.concat([
    pdf[["subject_id", "subject_type"]].rename(columns={"subject_id": "entity_id", "subject_type": "entity_type"}),
    pdf[["object_id", "object_type"]].rename(columns={"object_id": "entity_id", "object_type": "entity_type"}),
]).drop_duplicates(subset=["entity_id"], keep="first")
type_map = dict(zip(type_pdf.entity_id, type_pdf.entity_type))

G = nx.DiGraph()
G.add_edges_from(zip(pdf.subject_id, pdf.object_id))
Gu = G.to_undirected()
print("graph: V={:,}, E={:,} (directed), E={:,} (undirected)".format(
    G.number_of_nodes(), G.number_of_edges(), Gu.number_of_edges()))

# COMMAND ----------

# MAGIC %md ## 3. PageRank + degree

# COMMAND ----------

pagerank = nx.pagerank(G, alpha=0.85)
degree = dict(Gu.degree())
print("pagerank + degree computed for {:,} nodes".format(len(pagerank)))

# COMMAND ----------

# MAGIC %md ## 4. Betweenness (sampled, guarded)
# MAGIC
# MAGIC Exact betweenness is O(V·E) — far too slow beyond small graphs. We use
# MAGIC networkx's pivot sampling (`k` source nodes) for an approximation, and skip
# MAGIC the computation entirely (NULL column) above `betweenness_max_nodes`.

# COMMAND ----------

n_nodes = Gu.number_of_nodes()
if n_nodes <= BETWEENNESS_MAX_NODES:
    k = min(BETWEENNESS_SAMPLE_K, n_nodes)
    betweenness = nx.betweenness_centrality(Gu, k=k, seed=42)
    print("approximate betweenness computed with k={} pivots".format(k))
else:
    betweenness = None
    print("skipping betweenness: {:,} nodes > guard {:,} (column will be NULL)".format(
        n_nodes, BETWEENNESS_MAX_NODES))

# COMMAND ----------

# MAGIC %md ## 5. Louvain communities + connected components

# COMMAND ----------

from networkx.algorithms.community import louvain_communities

communities = louvain_communities(Gu, seed=42)
community_of = {}
community_size = {}
for cid, members in enumerate(communities):
    community_size[cid] = len(members)
    for node in members:
        community_of[node] = cid
print("louvain: {:,} communities, largest = {:,} nodes".format(
    len(communities), max(community_size.values())))

component_of = {}
component_sizes = []
for comp_id, comp in enumerate(nx.connected_components(Gu)):
    component_sizes.append(len(comp))
    for node in comp:
        component_of[node] = comp_id
component_sizes.sort(reverse=True)
print("connected components: {:,} total, giant component = {:,} nodes ({:.1%} of graph)".format(
    len(component_sizes), component_sizes[0], component_sizes[0] / n_nodes))

# COMMAND ----------

# MAGIC %md ## 6. Write `entity_centrality` and `entity_communities`

# COMMAND ----------

nodes = list(Gu.nodes())

centrality_pdf = pd.DataFrame({
    "entity_id": nodes,
    "entity_type": [type_map.get(n) for n in nodes],
    "pagerank": [float(pagerank.get(n, 0.0)) for n in nodes],
    "degree": [int(degree.get(n, 0)) for n in nodes],
    "betweenness": [float(betweenness[n]) if betweenness is not None else np.nan for n in nodes],
})
# force StringDtype so an all-null type column still maps to STRING, not binary
centrality_pdf["entity_id"] = centrality_pdf["entity_id"].astype("string")
centrality_pdf["entity_type"] = centrality_pdf["entity_type"].astype("string")

communities_pdf = pd.DataFrame({
    "entity_id": nodes,
    "entity_type": [type_map.get(n) for n in nodes],
    "community_id": [int(community_of[n]) for n in nodes],
    "community_size": [int(community_size[community_of[n]]) for n in nodes],
    "component_id": [int(component_of[n]) for n in nodes],
})
communities_pdf["entity_id"] = communities_pdf["entity_id"].astype("string")
communities_pdf["entity_type"] = communities_pdf["entity_type"].astype("string")

CENTRALITY_SCHEMA = "entity_id string, entity_type string, pagerank double, degree bigint, betweenness double"
COMMUNITIES_SCHEMA = "entity_id string, entity_type string, community_id bigint, community_size bigint, component_id bigint"

(spark.createDataFrame(centrality_pdf, schema=CENTRALITY_SCHEMA)
      .write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(FQ + ".entity_centrality"))
(spark.createDataFrame(communities_pdf, schema=COMMUNITIES_SCHEMA)
      .write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(FQ + ".entity_communities"))

print("wrote {:,} rows to {}.entity_centrality".format(len(centrality_pdf), FQ))
print("wrote {:,} rows to {}.entity_communities".format(len(communities_pdf), FQ))

# COMMAND ----------

# MAGIC %md ## 7. Sanity check via the serving functions

# COMMAND ----------

print("--- top_central_entities(10) ---")
display(spark.sql("SELECT * FROM " + FQ + ".top_central_entities(10)"))

print("--- largest communities ---")
display(spark.sql(
    "SELECT community_id, MAX(community_size) AS community_size, COUNT(*) AS n_members "
    "FROM " + FQ + ".entity_communities GROUP BY community_id ORDER BY community_size DESC LIMIT 10"
))
