# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Derive graph views from `gold_triplets`
# MAGIC
# MAGIC This notebook turns the pipeline's `gold_triplets` table (or any node/edge
# MAGIC table pair you point it at) into a set of SQL views that make the graph
# MAGIC queryable by Genie — and by any analyst who only speaks SQL.
# MAGIC
# MAGIC | View | What it gives Genie |
# MAGIC |---|---|
# MAGIC | `nodes_derived` | One row per entity: id, type, in/out/total degree. Derived from the union of subjects and objects when you have no separate node table. Skipped if you supply your own node table. |
# MAGIC | `edges_enriched` | Every edge with the full source-node and dest-node attribute set joined in (prefixed `source_…` / `dest_…`). Answers "show me edges where the source is a Person" in a single SQL query. |
# MAGIC | `downstream_khop` | Recursive forward traversal: for every node, every node reachable in ≤ K hops, plus the node path and the predicates walked. Answers "what does X lead to". |
# MAGIC | `upstream_khop` | Same in reverse. Answers "what points at X". |
# MAGIC | `node_degree` | In/out/total degree per node. Useful for hub identification. |
# MAGIC
# MAGIC ### Input contract
# MAGIC
# MAGIC By default this reads the starter kit's central data contract:
# MAGIC `main.knowledge_graph.gold_triplets` with columns `subject_id`, `subject_type`,
# MAGIC `predicate`, `object_id`, `object_type`, `confidence`, `source_agent`,
# MAGIC `source_method`, `properties`, `created_at`. Each row is one directed edge
# MAGIC `subject_id -[predicate]-> object_id`; there is no separate node table, so the
# MAGIC node set is derived from the union of subjects and objects.
# MAGIC
# MAGIC It stays fully bring-your-own, though. The minimum required is an **edge
# MAGIC table** with source-id and dest-id columns. Everything else — label, type,
# MAGIC confidence, properties — is passed through as opaque attributes Genie can
# MAGIC filter on. If you *do* have a node table, set `node_table` and `node_id_col`
# MAGIC and it will be used instead of deriving one.

# COMMAND ----------

dbutils.widgets.text("edge_table",      "main.knowledge_graph.gold_triplets", "Edge table (FQN)")
dbutils.widgets.text("node_table",      "",                                   "Optional node table (FQN; blank = derive from edges)")
dbutils.widgets.text("output_catalog",  "main",                               "Output catalog")
dbutils.widgets.text("output_schema",   "knowledge_graph",                    "Output schema (will be created)")

dbutils.widgets.text("edge_source_col", "subject_id",   "Edge table: source-id column")
dbutils.widgets.text("edge_dest_col",   "object_id",    "Edge table: dest-id column")
dbutils.widgets.text("edge_label_col",  "predicate",    "Edge table: label column (blank if none)")
dbutils.widgets.text("src_type_col",    "subject_type", "Edge table: source-type column (used when deriving nodes; blank if none)")
dbutils.widgets.text("dst_type_col",    "object_type",  "Edge table: dest-type column (used when deriving nodes; blank if none)")
dbutils.widgets.text("node_id_col",     "id",           "Node table: id column (only used when node_table is set)")
dbutils.widgets.text("max_hops",        "3",            "Traversal depth for downstream/upstream views")

EDGE_TABLE  = dbutils.widgets.get("edge_table")
NODE_TABLE  = dbutils.widgets.get("node_table").strip()
OUT_CATALOG = dbutils.widgets.get("output_catalog")
OUT_SCHEMA  = dbutils.widgets.get("output_schema")
EDGE_SRC    = dbutils.widgets.get("edge_source_col")
EDGE_DST    = dbutils.widgets.get("edge_dest_col")
EDGE_LABEL  = dbutils.widgets.get("edge_label_col").strip()
SRC_TYPE    = dbutils.widgets.get("src_type_col").strip()
DST_TYPE    = dbutils.widgets.get("dst_type_col").strip()
NODE_ID     = dbutils.widgets.get("node_id_col")
MAX_HOPS    = int(dbutils.widgets.get("max_hops"))

DERIVE_NODES = not NODE_TABLE

OUT_NS = f"{OUT_CATALOG}.{OUT_SCHEMA}"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {OUT_NS}")

print(f"Source: edges={EDGE_TABLE}  nodes={'(derived from edges)' if DERIVE_NODES else NODE_TABLE}")
print(f"Output: {OUT_NS}.*")
print(f"Columns: edge.source={EDGE_SRC}  edge.dest={EDGE_DST}  edge.label={EDGE_LABEL or '(none)'}  max_hops={MAX_HOPS}")

# COMMAND ----------

# MAGIC %md ## 1. Validate the inputs

# COMMAND ----------

from pyspark.sql import functions as F

edge_df = spark.table(EDGE_TABLE)
edge_cols = edge_df.columns

assert EDGE_SRC in edge_cols, f"edge_source_col '{EDGE_SRC}' not in {EDGE_TABLE}: {edge_cols}"
assert EDGE_DST in edge_cols, f"edge_dest_col '{EDGE_DST}' not in {EDGE_TABLE}: {edge_cols}"

if EDGE_LABEL and EDGE_LABEL not in edge_cols:
    print(f"WARN: edge_label_col '{EDGE_LABEL}' not in {EDGE_TABLE}; k-hop views will not carry predicates.")
    EDGE_LABEL = ""
if DERIVE_NODES:
    if SRC_TYPE and SRC_TYPE not in edge_cols:
        print(f"WARN: src_type_col '{SRC_TYPE}' not in {EDGE_TABLE}; derived node type will be NULL for sources.")
        SRC_TYPE = ""
    if DST_TYPE and DST_TYPE not in edge_cols:
        print(f"WARN: dst_type_col '{DST_TYPE}' not in {EDGE_TABLE}; derived node type will be NULL for dests.")
        DST_TYPE = ""

print(f"edges: {edge_df.count():,} rows, columns: {edge_cols}")

if not DERIVE_NODES:
    node_df = spark.table(NODE_TABLE)
    assert NODE_ID in node_df.columns, f"node_id_col '{NODE_ID}' not in {NODE_TABLE}: {node_df.columns}"
    print(f"nodes: {node_df.count():,} rows, columns: {node_df.columns}")

    # Quick referential integrity check
    node_ids   = node_df.select(F.col(NODE_ID).alias("id")).distinct()
    edge_srcs  = edge_df.select(F.col(EDGE_SRC).alias("id")).distinct()
    edge_dsts  = edge_df.select(F.col(EDGE_DST).alias("id")).distinct()
    orphan_srcs = edge_srcs.join(node_ids, "id", "left_anti").count()
    orphan_dsts = edge_dsts.join(node_ids, "id", "left_anti").count()
    if orphan_srcs or orphan_dsts:
        print(f"WARN: {orphan_srcs} edge sources and {orphan_dsts} edge dests do not match any node id. Traversal will skip them.")

# COMMAND ----------

# MAGIC %md ## 2. The node source — `nodes_derived` or your own table
# MAGIC
# MAGIC When no node table is given, derive one from the union of subjects and
# MAGIC objects. Each entity gets its id, its type (taken from `subject_type` /
# MAGIC `object_type` when present), and its in/out/total degree — so the derived
# MAGIC view is immediately useful on its own ("how many Persons are in the graph?").

# COMMAND ----------

def qid(name: str) -> str:
    """Backtick-quote a column name for SQL."""
    return "`" + name.replace("`", "``") + "`"

if DERIVE_NODES:
    src_type_expr = qid(SRC_TYPE) if SRC_TYPE else "CAST(NULL AS STRING)"
    dst_type_expr = qid(DST_TYPE) if DST_TYPE else "CAST(NULL AS STRING)"

    nodes_sql = f"""
    CREATE OR REPLACE VIEW {OUT_NS}.nodes_derived AS
    WITH endpoints AS (
        SELECT {qid(EDGE_SRC)} AS id, {src_type_expr} AS type, 1 AS is_src FROM {EDGE_TABLE}
        UNION ALL
        SELECT {qid(EDGE_DST)} AS id, {dst_type_expr} AS type, 0 AS is_src FROM {EDGE_TABLE}
    )
    SELECT
      id,
      MAX(type)        AS type,
      SUM(is_src)      AS out_degree,
      SUM(1 - is_src)  AS in_degree,
      COUNT(*)         AS degree
    FROM endpoints
    WHERE id IS NOT NULL
    GROUP BY id
    """
    spark.sql(nodes_sql)
    NODE_SOURCE = f"{OUT_NS}.nodes_derived"
    NODE_SOURCE_ID = "id"
    print(f"Created {NODE_SOURCE}")
else:
    NODE_SOURCE = NODE_TABLE
    NODE_SOURCE_ID = NODE_ID
    print(f"Using provided node table {NODE_SOURCE} (nodes_derived skipped)")

node_source_cols = spark.table(NODE_SOURCE).columns

# COMMAND ----------

# MAGIC %md ## 3. `edges_enriched` — every edge with both endpoints joined in
# MAGIC
# MAGIC Every edge column (`predicate`, `confidence`, `source_method`, `properties`,
# MAGIC … — whatever the edge table carries) is passed through as-is, and all node
# MAGIC columns are aliased with `source_` / `dest_` prefixes so Genie can
# MAGIC `WHERE source_type = 'Person' AND dest_type = 'Company'` without collisions.

# COMMAND ----------

# SELECT lists for the joins. Always include every column from edges, plus prefixed copies of every node column.
edge_sel   = ", ".join(f"e.{qid(c)} AS {qid(c)}" for c in edge_cols)
source_sel = ", ".join(f"ns.{qid(c)} AS {qid('source_' + c)}" for c in node_source_cols)
dest_sel   = ", ".join(f"nd.{qid(c)} AS {qid('dest_' + c)}"   for c in node_source_cols)

enriched_sql = f"""
CREATE OR REPLACE VIEW {OUT_NS}.edges_enriched AS
SELECT
  {edge_sel},
  {source_sel},
  {dest_sel}
FROM {EDGE_TABLE} e
JOIN {NODE_SOURCE} ns ON e.{qid(EDGE_SRC)} = ns.{qid(NODE_SOURCE_ID)}
JOIN {NODE_SOURCE} nd ON e.{qid(EDGE_DST)} = nd.{qid(NODE_SOURCE_ID)}
"""
spark.sql(enriched_sql)
print(f"Created {OUT_NS}.edges_enriched")

# COMMAND ----------

# MAGIC %md ## 4. `downstream_khop` and `upstream_khop` — fixed-hop traversal
# MAGIC
# MAGIC Fixed-hop `UNION ALL` blocks (one per hop, with cycle protection via the
# MAGIC accumulated `path`) rather than `WITH RECURSIVE`: Spark's recursive-CTE
# MAGIC executor materializes the full closure before any outer `start_id` filter is
# MAGIC applied, which blows the recursion row limit on dense graphs. Plain join
# MAGIC chains get normal predicate pushdown, so `WHERE start_id = ...` stays cheap.
# MAGIC When an edge-label column is configured, each row also carries `predicates` —
# MAGIC the ordered list of relationship labels walked — so Genie can answer not just
# MAGIC "what is reachable from X" but "*how* is X connected to Y".

# COMMAND ----------

def khop_sql(view_name: str, walk_from: str, walk_to: str) -> str:
    """Build a fixed-hop k-hop view walking edges from `walk_from` to `walk_to`."""
    if EDGE_LABEL:
        h1_extra   = f",\n           ARRAY(CAST(e.{qid(EDGE_LABEL)} AS STRING)) AS predicates"
        step_extra = f", array_append(p.predicates, CAST(e.{qid(EDGE_LABEL)} AS STRING))"
    else:
        h1_extra   = ""
        step_extra = ""
    hops = [f"""
    SELECT e.{qid(walk_from)} AS start_id,
           e.{qid(walk_to)} AS reached_id,
           1 AS hop,
           ARRAY(e.{qid(walk_from)}, e.{qid(walk_to)}) AS path{h1_extra}
    FROM {EDGE_TABLE} e
    WHERE e.{qid(walk_from)} <> e.{qid(walk_to)}"""]
    for k in range(2, MAX_HOPS + 1):
        hops.append(f"""
    SELECT p.start_id,
           e.{qid(walk_to)} AS reached_id,
           {k} AS hop,
           array_append(p.path, e.{qid(walk_to)}) AS path{step_extra}
    FROM hop{k - 1} p
    JOIN {EDGE_TABLE} e ON e.{qid(walk_from)} = p.reached_id
    WHERE NOT array_contains(p.path, e.{qid(walk_to)})""")
    ctes = ",\n".join(f"hop{k} AS ({sql}\n)" for k, sql in enumerate(hops, start=1))
    union = "\nUNION ALL\n".join(f"SELECT * FROM hop{k}" for k in range(1, MAX_HOPS + 1))
    return f"""
CREATE OR REPLACE VIEW {OUT_NS}.{view_name} AS
WITH
{ctes}
{union}
"""

spark.sql(khop_sql("downstream_khop", walk_from=EDGE_SRC, walk_to=EDGE_DST))
spark.sql(khop_sql("upstream_khop",   walk_from=EDGE_DST, walk_to=EDGE_SRC))
print(f"Created {OUT_NS}.downstream_khop and {OUT_NS}.upstream_khop  (max {MAX_HOPS} hops)")

# COMMAND ----------

# MAGIC %md ## 5. `node_degree` — in/out/total degree per node

# COMMAND ----------

if DERIVE_NODES:
    # nodes_derived already computed degrees; node_degree is a thin projection
    # so downstream consumers get the same view name either way.
    degree_sql = f"""
    CREATE OR REPLACE VIEW {OUT_NS}.node_degree AS
    SELECT id, out_degree, in_degree, degree AS total_degree
    FROM {OUT_NS}.nodes_derived
    """
else:
    degree_sql = f"""
    CREATE OR REPLACE VIEW {OUT_NS}.node_degree AS
    SELECT
      n.{qid(NODE_SOURCE_ID)} AS id,
      COALESCE(out_d.out_degree, 0) AS out_degree,
      COALESCE(in_d.in_degree, 0)   AS in_degree,
      COALESCE(out_d.out_degree, 0) + COALESCE(in_d.in_degree, 0) AS total_degree
    FROM {NODE_SOURCE} n
    LEFT JOIN (
      SELECT {qid(EDGE_SRC)} AS id, COUNT(*) AS out_degree
      FROM {EDGE_TABLE} GROUP BY {qid(EDGE_SRC)}
    ) out_d ON n.{qid(NODE_SOURCE_ID)} = out_d.id
    LEFT JOIN (
      SELECT {qid(EDGE_DST)} AS id, COUNT(*) AS in_degree
      FROM {EDGE_TABLE} GROUP BY {qid(EDGE_DST)}
    ) in_d ON n.{qid(NODE_SOURCE_ID)} = in_d.id
    """
spark.sql(degree_sql)
print(f"Created {OUT_NS}.node_degree")

# COMMAND ----------

# MAGIC %md ## 6. Smoke test
# MAGIC
# MAGIC The k-hop views walk from *every* node, so an unfiltered `COUNT(*)` materializes the
# MAGIC full transitive closure — on dense graphs that exceeds Spark's recursion row limit.
# MAGIC They are meant to be queried the way Genie queries them: filtered by `start_id`.
# MAGIC The smoke test does the same, anchoring at a single sample node.

# COMMAND ----------

for view in (["nodes_derived"] if DERIVE_NODES else []) + ["edges_enriched", "node_degree"]:
    n = spark.sql(f"SELECT COUNT(*) AS n FROM {OUT_NS}.{view}").first()["n"]
    print(f"  {OUT_NS}.{view}: {n:,} rows")

sample_id = spark.sql(
    f"SELECT {qid(EDGE_SRC)} AS id FROM {EDGE_TABLE} LIMIT 1"
).first()["id"]
for view in ["downstream_khop", "upstream_khop"]:
    n = spark.sql(
        f"SELECT COUNT(*) AS n FROM (SELECT 1 FROM {OUT_NS}.{view} WHERE start_id = ? LIMIT 10000)",
        args=[sample_id],
    ).first()["n"]
    print(f"  {OUT_NS}.{view} (start_id={sample_id}): {n:,} rows (capped at 10,000)")

# COMMAND ----------

# MAGIC %md ## 7. Pass the table list downstream
# MAGIC
# MAGIC The Genie-space notebook (`04_create_genie_space`) will attach these tables.
# MAGIC Emit the FQN list as the notebook exit value so jobs can chain.

# COMMAND ----------

genie_tables = [
    EDGE_TABLE,
    NODE_SOURCE,
    f"{OUT_NS}.edges_enriched",
    f"{OUT_NS}.downstream_khop",
    f"{OUT_NS}.upstream_khop",
    f"{OUT_NS}.node_degree",
]
import json
dbutils.notebook.exit(json.dumps(genie_tables))
