# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — GPU graph analytics with NVIDIA RAPIDS cuGraph (scale-up alternative)
# MAGIC
# MAGIC The scale-up alternative to `notebooks/06_graph_algorithms.py`: when
# MAGIC `gold_triplets` grows past what the driver-side networkx path handles
# MAGIC comfortably (~5M edges), cuGraph runs the same algorithms on GPU and scales
# MAGIC to hundreds of millions of edges. It reads `gold_triplets` as the edge list
# MAGIC and writes the **same two output tables** as notebook 06, so the serving UC
# MAGIC functions (`sql/04_analytics_functions.sql`) work unchanged.
# MAGIC
# MAGIC | Output table | Columns |
# MAGIC |---|---|
# MAGIC | `entity_centrality` | entity_id, entity_type, pagerank, degree, betweenness (nullable) |
# MAGIC | `entity_communities` | entity_id, entity_type, community_id, community_size, component_id |
# MAGIC
# MAGIC ### Algorithms
# MAGIC | Algorithm | Use |
# MAGIC |---|---|
# MAGIC | PageRank | persisted to `entity_centrality` |
# MAGIC | Degree | persisted to `entity_centrality` |
# MAGIC | Betweenness | sampled (`betweenness_k` pivots), optional — persisted or NULL |
# MAGIC | Louvain | persisted to `entity_communities` |
# MAGIC | Connected components | persisted as `component_id` |
# MAGIC | SSSP + BFS | interactive demo from a seed entity (displayed, not persisted) |
# MAGIC
# MAGIC ### Cluster requirement
# MAGIC Attach the GPU cluster from `cluster_specs/gpu_cugraph.json` (single-node,
# MAGIC DBR `15.4.x-gpu-ml`). Do NOT run on serverless or CPU clusters — use
# MAGIC notebook 06 there instead.

# COMMAND ----------

# MAGIC %pip install --extra-index-url=https://pypi.nvidia.com cugraph-cu12==24.10.* cudf-cu12==24.10.* --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "knowledge_graph")
dbutils.widgets.text("min_confidence", "0.0", "minimum triplet confidence to include (NULL confidence is kept)")
dbutils.widgets.text("betweenness_k", "0", "pivot nodes for sampled betweenness; 0 = skip (NULL column)")
dbutils.widgets.text("seed_entity_id", "", "(optional) seed entity for the SSSP/BFS demo; defaults to highest-degree entity")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
MIN_CONFIDENCE = float(dbutils.widgets.get("min_confidence"))
BETWEENNESS_K = int(dbutils.widgets.get("betweenness_k"))
SEED = dbutils.widgets.get("seed_entity_id").strip() or None

FQ = CATALOG + "." + SCHEMA
GT = FQ + ".gold_triplets"
print("Input:  " + GT)
print("Output: " + FQ + ".entity_centrality, " + FQ + ".entity_communities")

# COMMAND ----------

# MAGIC %md ## 1. Load gold_triplets as a src/dst edge list
# MAGIC
# MAGIC cuGraph wants integer vertex ids, so entity ids are mapped to dense int64
# MAGIC vids and mapped back before writing. `confidence` becomes the edge weight
# MAGIC (clipped away from zero so weighted algorithms stay stable).

# COMMAND ----------

import cudf, cugraph, pandas as pd, numpy as np
from pyspark.sql import functions as F

edges_pd = (
    spark.table(GT)
    .where((F.col("confidence").isNull()) | (F.col("confidence") >= MIN_CONFIDENCE))
    .where(F.col("subject_id").isNotNull() & F.col("object_id").isNotNull())
    .select("subject_id", "subject_type", "object_id", "object_type", "confidence")
    .toPandas()
)
if len(edges_pd) == 0:
    raise ValueError(GT + " has no rows - run the triplet-generation notebooks (01-04) first.")
print("loaded {:,} edges".format(len(edges_pd)))

# entity -> type map (first triplet that mentions the entity wins)
type_pd = pd.concat([
    edges_pd[["subject_id", "subject_type"]].rename(columns={"subject_id": "entity_id", "subject_type": "entity_type"}),
    edges_pd[["object_id", "object_type"]].rename(columns={"object_id": "entity_id", "object_type": "entity_type"}),
]).drop_duplicates(subset=["entity_id"], keep="first")
type_map = dict(zip(type_pd.entity_id, type_pd.entity_type))

# Integer vertex ids
nodes_pd = pd.DataFrame({"entity_id": type_pd.entity_id.values}).reset_index(drop=True)
nodes_pd["vid"] = nodes_pd.index.astype("int64")
id_to_vid = dict(zip(nodes_pd.entity_id, nodes_pd.vid))
vid_to_id = dict(zip(nodes_pd.vid, nodes_pd.entity_id))

edges_pd["src"] = edges_pd.subject_id.map(id_to_vid)
edges_pd["dst"] = edges_pd.object_id.map(id_to_vid)
edges_pd["weight"] = pd.to_numeric(edges_pd.confidence, errors="coerce").fillna(1.0).clip(lower=0.001)
edges_pd = edges_pd.dropna(subset=["src", "dst"]).astype({"src": "int64", "dst": "int64"})

edf = cudf.DataFrame.from_pandas(edges_pd[["src", "dst", "weight"]])
print("loaded {:,} edges into cuDF".format(len(edf)))

G = cugraph.Graph(directed=True)
G.from_cudf_edgelist(edf, source="src", destination="dst", edge_attr="weight")

G_und = cugraph.Graph(directed=False)
G_und.from_cudf_edgelist(edf, source="src", destination="dst", edge_attr="weight")
print("cuGraph: V={:,}, E={:,}".format(G.number_of_vertices(), G.number_of_edges()))

# COMMAND ----------

# MAGIC %md ## 2. PageRank + degree

# COMMAND ----------

pr_pd = cugraph.pagerank(G, alpha=0.85).to_pandas()  # vertex, pagerank

deg_pd = G_und.degree().to_pandas()  # vertex, degree
print("pagerank + degree computed for {:,} vertices".format(len(pr_pd)))

# COMMAND ----------

# MAGIC %md ## 3. Betweenness (sampled, optional)
# MAGIC
# MAGIC Even on GPU, betweenness is the expensive one — enable it by setting the
# MAGIC `betweenness_k` widget to a pivot count (e.g. 256). Left at 0 the column is
# MAGIC written as NULL, which the serving functions tolerate.

# COMMAND ----------

if BETWEENNESS_K > 0:
    k = min(BETWEENNESS_K, G_und.number_of_vertices())
    bt_pd = cugraph.betweenness_centrality(G_und, k=k, seed=42).to_pandas()
    bt_pd = bt_pd.rename(columns={"betweenness_centrality": "betweenness"})[["vertex", "betweenness"]]
    print("sampled betweenness computed with k={} pivots".format(k))
else:
    bt_pd = None
    print("betweenness skipped (betweenness_k=0) - column will be NULL")

# COMMAND ----------

# MAGIC %md ## 4. Louvain communities + connected components

# COMMAND ----------

parts, modularity = cugraph.louvain(G_und)
parts_pd = parts.to_pandas()  # vertex, partition
print("louvain: {:,} communities, modularity={:.4f}".format(parts_pd.partition.nunique(), modularity))

cc_pd = cugraph.connected_components(G_und).to_pandas()  # vertex, labels
print("connected components: {:,}".format(cc_pd.labels.nunique()))

# COMMAND ----------

# MAGIC %md ## 5. SSSP + BFS demo from a seed entity
# MAGIC
# MAGIC Interactive sanity check (displayed, not persisted): weighted shortest-path
# MAGIC distance and layered BFS depth counts from a seed. Point `seed_entity_id` at
# MAGIC any entity in the graph; the default is the highest-degree entity.

# COMMAND ----------

if SEED:
    if SEED not in id_to_vid:
        raise ValueError("seed_entity_id=" + repr(SEED) + " not found in " + GT)
    seed_vid = id_to_vid[SEED]
else:
    seed_vid = int(deg_pd.sort_values("degree", ascending=False).vertex.iloc[0])
    SEED = vid_to_id[seed_vid]
print("Seed entity: " + str(SEED) + " (vid=" + str(seed_vid) + ")")

sssp = cugraph.sssp(G, source=seed_vid).to_pandas()
sssp["entity_id"] = sssp.vertex.map(vid_to_id)
reachable = sssp[sssp.distance < np.inf].sort_values("distance")
print("Reachable from {}: {:,}/{:,}".format(SEED, len(reachable), G.number_of_vertices()))
display(spark.createDataFrame(reachable[["entity_id", "distance"]].head(50)))

bfs = cugraph.bfs(G, start=seed_vid).to_pandas()
depth = bfs[bfs.distance < np.inf].groupby("distance").size().reset_index(name="n_entities")
display(spark.createDataFrame(depth))

# COMMAND ----------

# MAGIC %md ## 6. Write `entity_centrality` and `entity_communities`
# MAGIC
# MAGIC Same schemas as notebook 06 — the serving UC functions read either engine's
# MAGIC output interchangeably.

# COMMAND ----------

out = nodes_pd.merge(pr_pd, left_on="vid", right_on="vertex", how="left") \
              .merge(deg_pd, left_on="vid", right_on="vertex", how="left", suffixes=("", "_d"))
if bt_pd is not None:
    out = out.merge(bt_pd, left_on="vid", right_on="vertex", how="left", suffixes=("", "_b"))
else:
    out["betweenness"] = np.nan

centrality_pdf = pd.DataFrame({
    "entity_id": out.entity_id.astype("string"),
    "entity_type": out.entity_id.map(type_map).astype("string"),
    "pagerank": out.pagerank.fillna(0.0).astype("float64"),
    "degree": out.degree.fillna(0).astype("int64"),
    "betweenness": out.betweenness.astype("float64"),
})

comm = nodes_pd.merge(parts_pd, left_on="vid", right_on="vertex", how="inner") \
               .merge(cc_pd, left_on="vid", right_on="vertex", how="left", suffixes=("", "_c"))
sizes = comm.groupby("partition")["vid"].transform("size")
communities_pdf = pd.DataFrame({
    "entity_id": comm.entity_id.astype("string"),
    "entity_type": comm.entity_id.map(type_map).astype("string"),
    "community_id": comm.partition.astype("int64"),
    "community_size": sizes.astype("int64"),
    "component_id": comm.labels.fillna(-1).astype("int64"),
})

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
