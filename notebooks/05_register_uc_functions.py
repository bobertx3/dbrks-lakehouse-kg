# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Register the KG data model + UC functions
# MAGIC
# MAGIC Executes the DDL in `sql/` against the target catalog/schema:
# MAGIC
# MAGIC | File | Creates |
# MAGIC |---|---|
# MAGIC | `sql/01_gold_triplets.sql` | `gold_triplets` table (the core edge contract) |
# MAGIC | `sql/02_dataset_registry.sql` | `dataset_registry` control table |
# MAGIC | `sql/03_traversal_functions.sql` | `neighbors`, `khop`, `connection_path`, `subgraph_edges` |
# MAGIC | `sql/04_analytics_functions.sql` | `entity_centrality` + `entity_communities` tables, `cluster_of`, `members_of_cluster`, `top_central_entities`, `shared_community` |
# MAGIC
# MAGIC ### Why the notebook substitutes tokens
# MAGIC UC function DDL cannot be parameterized natively (no `IDENTIFIER()` for the
# MAGIC function's own three-level name, no bind parameters in `CREATE FUNCTION`
# MAGIC bodies), so the `.sql` files use plain `${catalog}` / `${schema}` placeholder
# MAGIC tokens. This notebook reads each file, replaces the tokens with the widget
# MAGIC values via simple string `.replace()` (no f-string templating, so `{...}` in
# MAGIC SQL text can never bite), splits on `;` statement boundaries, and runs each
# MAGIC statement with `spark.sql`.
# MAGIC
# MAGIC Assumes the repo is checked out as Workspace Files (Git folder), so the
# MAGIC `sql/` directory is reachable at `../sql` relative to this notebook.
# MAGIC
# MAGIC The traversal functions intentionally use **fixed-hop UNION ALL CTEs** rather
# MAGIC than recursive CTEs — recursive CTEs are unreliable on DBSQL and inside UC
# MAGIC function bodies.

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "knowledge_graph")
dbutils.widgets.text("grant_execute_to", "", "(optional) principal granted EXECUTE on functions + SELECT on tables, e.g. account users")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
GRANT_TO = dbutils.widgets.get("grant_execute_to").strip()

spark.sql("CREATE SCHEMA IF NOT EXISTS " + CATALOG + "." + SCHEMA)
print("Target: " + CATALOG + "." + SCHEMA)

# COMMAND ----------

# MAGIC %md ## 1. Locate the sql/ directory

# COMMAND ----------

import os

# Notebook cwd is the notebook's directory when the repo lives in Workspace Files.
SQL_DIR = os.path.normpath(os.path.join(os.getcwd(), "..", "sql"))
if not os.path.isdir(SQL_DIR):
    raise FileNotFoundError(
        "sql/ directory not found at " + SQL_DIR + ". "
        "Run this notebook from a Git folder / Workspace Files checkout of the repo."
    )

SQL_FILES = [
    "01_gold_triplets.sql",
    "02_dataset_registry.sql",
    "03_traversal_functions.sql",
    "04_analytics_functions.sql",
]
print("sql dir: " + SQL_DIR)

# COMMAND ----------

# MAGIC %md ## 2. Substitute tokens, split statements, execute

# COMMAND ----------

def split_statements(sql_text):
    """Split a .sql file on `;`. The sql/ files guarantee no semicolons inside
    comments or string literals, so a plain split is safe. Fragments that are
    only comments/whitespace (e.g. trailing header comments) are skipped."""
    statements = []
    for chunk in sql_text.split(";"):
        lines = [ln for ln in chunk.splitlines() if ln.strip()]
        if not lines or all(ln.strip().startswith("--") for ln in lines):
            continue
        statements.append(chunk.strip())
    return statements


def run_sql_file(filename):
    path = os.path.join(SQL_DIR, filename)
    with open(path, "r") as f:
        text = f.read()
    text = text.replace("${catalog}", CATALOG).replace("${schema}", SCHEMA)
    stmts = split_statements(text)
    print("== " + filename + " (" + str(len(stmts)) + " statements) ==")
    for stmt in stmts:
        first_line = next(ln.strip() for ln in stmt.splitlines() if ln.strip() and not ln.strip().startswith("--"))
        print("   " + first_line[:100])
        spark.sql(stmt)


for fname in SQL_FILES:
    run_sql_file(fname)

print("All DDL executed.")

# COMMAND ----------

# MAGIC %md ## 3. Smoke-test every function
# MAGIC
# MAGIC Harmless SELECTs with dummy IDs — an empty `gold_triplets` / analytics table
# MAGIC returns zero rows, which is fine. Each call is wrapped in try/except so one
# MAGIC failure does not hide the rest; failures are re-raised at the end.

# COMMAND ----------

FQ = CATALOG + "." + SCHEMA

SMOKE_TESTS = [
    ("neighbors",            "SELECT * FROM " + FQ + ".neighbors('__smoke_test__')"),
    ("khop",                 "SELECT * FROM " + FQ + ".khop('__smoke_test__', 2)"),
    ("connection_path",      "SELECT * FROM " + FQ + ".connection_path('__smoke_a__', '__smoke_b__')"),
    ("subgraph_edges",       "SELECT * FROM " + FQ + ".subgraph_edges('__smoke_test__', 2)"),
    ("cluster_of",           "SELECT " + FQ + ".cluster_of('__smoke_test__') AS community_id"),
    ("members_of_cluster",   "SELECT * FROM " + FQ + ".members_of_cluster(CAST(-1 AS BIGINT))"),
    ("top_central_entities", "SELECT * FROM " + FQ + ".top_central_entities(5)"),
    ("shared_community",     "SELECT * FROM " + FQ + ".shared_community('__smoke_a__', '__smoke_b__')"),
]

failures = []
for name, query in SMOKE_TESTS:
    try:
        n = spark.sql(query).count()
        print("ok   " + name + " (" + str(n) + " rows)")
    except Exception as e:
        failures.append((name, str(e)))
        print("FAIL " + name + ": " + str(e)[:200])

if failures:
    raise RuntimeError("Smoke tests failed for: " + ", ".join(name for name, _ in failures))
print("All " + str(len(SMOKE_TESTS)) + " functions smoke-tested.")

# COMMAND ----------

# MAGIC %md ## 4. Optional grants
# MAGIC
# MAGIC If `grant_execute_to` is set (e.g. `account users`, or an app/agent service
# MAGIC principal's applicationId), grant EXECUTE on the functions and SELECT on the
# MAGIC underlying tables so serving endpoints and Genie can call them.

# COMMAND ----------

FUNCTIONS = [
    "neighbors", "khop", "connection_path", "subgraph_edges",
    "cluster_of", "members_of_cluster", "top_central_entities", "shared_community",
]
TABLES = ["gold_triplets", "dataset_registry", "entity_centrality", "entity_communities"]

if GRANT_TO:
    principal = "`" + GRANT_TO + "`"
    spark.sql("GRANT USE CATALOG ON CATALOG " + CATALOG + " TO " + principal)
    spark.sql("GRANT USE SCHEMA ON SCHEMA " + FQ + " TO " + principal)
    for fn in FUNCTIONS:
        spark.sql("GRANT EXECUTE ON FUNCTION " + FQ + "." + fn + " TO " + principal)
        print("granted EXECUTE on " + fn)
    for t in TABLES:
        spark.sql("GRANT SELECT ON TABLE " + FQ + "." + t + " TO " + principal)
        print("granted SELECT on " + t)
else:
    print("grant_execute_to not set - skipping grants.")

print("Done. Functions are ready to attach as agent tools or use from Genie.")
