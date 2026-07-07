# Databricks notebook source
# MAGIC %md
# MAGIC # 08 — Distributed graph analytics with Apache GraphFrames (scale-out CPU alternative)
# MAGIC
# MAGIC The distributed CPU alternative to `notebooks/06_graph_algorithms.py`: when
# MAGIC `gold_triplets` grows past what the driver-side networkx path handles (~5M
# MAGIC edges) and a GPU cluster (`notebooks/07_cugraph_gpu.py`) is not an option,
# MAGIC [Apache GraphFrames](https://graphframes.github.io/graphframes/docs/_site/index.html)
# MAGIC runs the same analytics as distributed Spark jobs. It reads `gold_triplets`
# MAGIC as the edge list and writes the **same two output tables**, so the serving
# MAGIC UC functions (`sql/04_analytics_functions.sql`) work unchanged.
# MAGIC
# MAGIC | Output table | Columns |
# MAGIC |---|---|
# MAGIC | `entity_centrality` | entity_id, entity_type, pagerank, degree, betweenness (NULL) |
# MAGIC | `entity_communities` | entity_id, entity_type, community_id, community_size, component_id |
# MAGIC
# MAGIC ### Algorithms (and how they differ from notebook 06)
# MAGIC | Algorithm | Notes |
# MAGIC |---|---|
# MAGIC | PageRank | directed, damping 0.85, fixed iterations |
# MAGIC | Degree | undirected degree (mutual edges collapsed, matching notebook 06) |
# MAGIC | Betweenness | **not available in GraphFrames** — written as NULL (the serving functions tolerate it) |
# MAGIC | Label propagation | GraphFrames has no Louvain; LPA fills `community_id` instead. Communities are comparable but not identical to notebook 06's Louvain output |
# MAGIC | Connected components | persisted as `component_id` (requires a checkpoint directory) |
# MAGIC
# MAGIC ### Cluster requirement
# MAGIC Run on a **Databricks ML runtime** cluster (e.g. `16.4.x-cpu-ml-scala2.12` —
# MAGIC see `cluster_specs/classic_graphframes.json`): GraphFrames ships pre-installed
# MAGIC there, no library install needed. Standard (non-ML) runtimes and serverless
# MAGIC compute do not include it — use notebook 06 on those instead.

# COMMAND ----------

try:
    from graphframes import GraphFrame
except ImportError as e:
    raise ImportError(
        "graphframes is not available on this cluster. Attach an ML runtime cluster "
        "(cluster_specs/classic_graphframes.json) - GraphFrames ships pre-installed there. "
        "On non-ML or serverless compute, use notebooks/06_graph_algorithms.py instead."
    ) from e

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "knowledge_graph")
dbutils.widgets.text("min_confidence", "0.0", "minimum triplet confidence to include (NULL confidence is kept)")
dbutils.widgets.text("exclude_source_agents", "graph_topology,ml_clustering,statistical_analysis", "source_agents whose triplets are excluded from the input graph (comma-sep)")
dbutils.widgets.text("pagerank_max_iter", "20", "fixed PageRank iterations")
dbutils.widgets.text("lpa_max_iter", "10", "label-propagation iterations")
dbutils.widgets.text("checkpoint_dir", "dbfs:/tmp/lakehouse_kg_gf_checkpoints", "checkpoint dir (required by connectedComponents)")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
MIN_CONFIDENCE = float(dbutils.widgets.get("min_confidence"))
EXCLUDE_AGENTS = [a.strip() for a in dbutils.widgets.get("exclude_source_agents").split(",") if a.strip()]
PAGERANK_MAX_ITER = int(dbutils.widgets.get("pagerank_max_iter"))
LPA_MAX_ITER = int(dbutils.widgets.get("lpa_max_iter"))
CHECKPOINT_DIR = dbutils.widgets.get("checkpoint_dir").strip()

FQ = CATALOG + "." + SCHEMA
GT = FQ + ".gold_triplets"
print("Input:  " + GT)
print("Output: " + FQ + ".entity_centrality, " + FQ + ".entity_communities")

spark.sparkContext.setCheckpointDir(CHECKPOINT_DIR)

# COMMAND ----------

# MAGIC %md ## 1. Build the vertex and edge DataFrames
# MAGIC
# MAGIC Same input contract as notebook 06: by default only *structural* triplets
# MAGIC feed the algorithms (see the `exclude_source_agents` note there). Everything
# MAGIC stays a distributed DataFrame — nothing is collected to the driver.

# COMMAND ----------

from pyspark.sql import functions as F

triplets = (
    spark.table(GT)
    .where((F.col("confidence").isNull()) | (F.col("confidence") >= MIN_CONFIDENCE))
    .where(~F.col("source_agent").isin(EXCLUDE_AGENTS) if EXCLUDE_AGENTS else F.lit(True))
    .where(F.col("subject_id").isNotNull() & F.col("object_id").isNotNull())
    .select("subject_id", "subject_type", "object_id", "object_type")
)

edges = triplets.select(
    F.col("subject_id").alias("src"), F.col("object_id").alias("dst")
).where(F.col("src") != F.col("dst")).distinct().cache()

vertices = (
    triplets.select(F.col("subject_id").alias("id"), F.col("subject_type").alias("entity_type"))
    .unionByName(triplets.select(F.col("object_id").alias("id"), F.col("object_type").alias("entity_type")))
    .groupBy("id").agg(F.max("entity_type").alias("entity_type"))
    .cache()
)

n_edges, n_vertices = edges.count(), vertices.count()
if n_edges == 0:
    raise ValueError(GT + " has no rows - run the triplet-generation notebooks (01-04) first.")
print("graph: V={:,}, E={:,} (distinct directed edges)".format(n_vertices, n_edges))

g = GraphFrame(vertices, edges)

# COMMAND ----------

# MAGIC %md ## 2. PageRank + degree

# COMMAND ----------

pr = g.pageRank(resetProbability=0.15, maxIter=PAGERANK_MAX_ITER)
pagerank_df = pr.vertices.select("id", "pagerank")

# Undirected degree with mutual edges collapsed, matching notebook 06's semantics
und_edges = edges.select(
    F.least("src", "dst").alias("a"), F.greatest("src", "dst").alias("b")
).distinct()
degree_df = (
    und_edges.select(F.col("a").alias("id"))
    .unionByName(und_edges.select(F.col("b").alias("id")))
    .groupBy("id").agg(F.count("*").alias("degree"))
)
print("pagerank + degree computed")

# COMMAND ----------

# MAGIC %md ## 3. Label-propagation communities + connected components

# COMMAND ----------

lpa_df = g.labelPropagation(maxIter=LPA_MAX_ITER).select("id", F.col("label").alias("community_id"))
cc_df = g.connectedComponents().select("id", F.col("component").alias("component_id"))

community_sizes = lpa_df.groupBy("community_id").agg(F.count("*").alias("community_size"))
n_communities = community_sizes.count()
print("label propagation: {:,} communities".format(n_communities))

# COMMAND ----------

# MAGIC %md ## 4. Write `entity_centrality` and `entity_communities`
# MAGIC
# MAGIC Identical schemas to notebooks 06/07. Betweenness has no GraphFrames
# MAGIC implementation, so the column is written as NULL — `top_central_entities`
# MAGIC and friends read pagerank/degree and tolerate it.

# COMMAND ----------

centrality_df = (
    vertices
    .join(pagerank_df, "id", "left")
    .join(degree_df, "id", "left")
    .select(
        F.col("id").cast("string").alias("entity_id"),
        F.col("entity_type").cast("string"),
        F.coalesce(F.col("pagerank"), F.lit(0.0)).cast("double").alias("pagerank"),
        F.coalesce(F.col("degree"), F.lit(0)).cast("bigint").alias("degree"),
        F.lit(None).cast("double").alias("betweenness"),
    )
)

communities_df = (
    vertices
    .join(lpa_df, "id", "left")
    .join(community_sizes, "community_id", "left")
    .join(cc_df, "id", "left")
    .select(
        F.col("id").cast("string").alias("entity_id"),
        F.col("entity_type").cast("string"),
        F.col("community_id").cast("bigint"),
        F.col("community_size").cast("bigint"),
        F.col("component_id").cast("bigint"),
    )
)

(centrality_df.write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(FQ + ".entity_centrality"))
(communities_df.write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(FQ + ".entity_communities"))

print("wrote {:,} rows to {}.entity_centrality".format(n_vertices, FQ))
print("wrote {:,} rows to {}.entity_communities".format(n_vertices, FQ))

# COMMAND ----------

# MAGIC %md ## 5. Sanity check via the serving functions

# COMMAND ----------

print("--- top_central_entities(10) ---")
display(spark.sql("SELECT * FROM " + FQ + ".top_central_entities(10)"))

print("--- largest communities ---")
display(spark.sql(
    "SELECT community_id, MAX(community_size) AS community_size, COUNT(*) AS n_members "
    "FROM " + FQ + ".entity_communities GROUP BY community_id ORDER BY community_size DESC LIMIT 10"
))
