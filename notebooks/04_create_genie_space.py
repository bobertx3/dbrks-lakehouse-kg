# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Create the Genie space
# MAGIC
# MAGIC Creates (or updates) a Genie space pointing at the views produced by
# MAGIC `03_derive_graph_views`, making the knowledge graph chat-ready. The space is
# MAGIC seeded with:
# MAGIC
# MAGIC - **Curated instructions** explaining the triplet schema and which view
# MAGIC   answers "what is reachable from X" vs. "what connects X and Y" vs.
# MAGIC   "show edges where …".
# MAGIC - **Sample questions** so the Genie chat suggests good starter prompts.
# MAGIC - **Example SQL** so the LLM has worked patterns for k-hop graph traversal.
# MAGIC
# MAGIC ### What you need
# MAGIC - A SQL warehouse id (the space binds to one)
# MAGIC - A workspace folder path where the space will live (e.g. `/Workspace/Users/you@company.com`)
# MAGIC - The same `edge_table` / `node_table` / output schema as you used in notebook 03
# MAGIC - Optionally, extra domain views the pipeline produced (comma-separated in
# MAGIC   `extra_tables`) — they get bound to the space alongside the graph views.

# COMMAND ----------

dbutils.widgets.text("edge_table",      "main.knowledge_graph.gold_triplets")
dbutils.widgets.text("node_table",      "",                                       "Optional node table (blank = nodes_derived from notebook 03)")
dbutils.widgets.text("output_catalog",  "main")
dbutils.widgets.text("output_schema",   "knowledge_graph")
dbutils.widgets.text("warehouse_id",    "",                                       "SQL warehouse id (required)")
dbutils.widgets.text("space_title",     "Knowledge Graph Genie",                  "Title of the Genie space")
dbutils.widgets.text("space_description", "Chat with the lakehouse knowledge graph: triplets, multi-hop traversal, and hub analysis over derived graph views.",
                                                                                  "One-line description")
dbutils.widgets.text("parent_path",     "",                                       "Workspace folder for the space (defaults to current user home)")
dbutils.widgets.text("edge_source_col", "subject_id")
dbutils.widgets.text("edge_dest_col",   "object_id")
dbutils.widgets.text("edge_label_col",  "predicate")
dbutils.widgets.text("node_id_col",     "id")
dbutils.widgets.text("extra_tables",    "",                                       "Extra views to bind (comma-separated; bare names resolve to the output schema)")
dbutils.widgets.dropdown("on_conflict", "update", ["update", "skip", "create_new"], "If a space with this title exists")

EDGE_TABLE   = dbutils.widgets.get("edge_table")
NODE_TABLE   = dbutils.widgets.get("node_table").strip()
OUT_CATALOG  = dbutils.widgets.get("output_catalog")
OUT_SCHEMA   = dbutils.widgets.get("output_schema")
WAREHOUSE_ID = dbutils.widgets.get("warehouse_id").strip()
SPACE_TITLE  = dbutils.widgets.get("space_title")
SPACE_DESC   = dbutils.widgets.get("space_description")
PARENT_PATH  = dbutils.widgets.get("parent_path").strip()
EDGE_SRC     = dbutils.widgets.get("edge_source_col")
EDGE_DST     = dbutils.widgets.get("edge_dest_col")
EDGE_LABEL   = dbutils.widgets.get("edge_label_col").strip()
NODE_ID      = dbutils.widgets.get("node_id_col")
EXTRA_TABLES = dbutils.widgets.get("extra_tables").strip()
ON_CONFLICT  = dbutils.widgets.get("on_conflict")

assert WAREHOUSE_ID, "warehouse_id is required. Find one under SQL Warehouses in the workspace."

OUT_NS = f"{OUT_CATALOG}.{OUT_SCHEMA}"

# The node source mirrors notebook 03: your own table if given, else the derived view.
NODE_SOURCE    = NODE_TABLE if NODE_TABLE else f"{OUT_NS}.nodes_derived"
NODE_SOURCE_ID = NODE_ID if NODE_TABLE else "id"

# Pipeline / domain views to bind alongside the graph views.
extra_tables = []
for name in EXTRA_TABLES.split(","):
    name = name.strip()
    if not name:
        continue
    extra_tables.append(name if "." in name else f"{OUT_NS}.{name}")

# COMMAND ----------

import json, uuid, os
import requests

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
HOST = ctx.apiUrl().get()
TOKEN = ctx.apiToken().get()
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

if not PARENT_PATH:
    user = ctx.userName().get()
    PARENT_PATH = f"/Workspace/Users/{user}"
print(f"Workspace:    {HOST}")
print(f"Parent path:  {PARENT_PATH}")
print(f"Warehouse:    {WAREHOUSE_ID}")
print(f"Space title:  {SPACE_TITLE}")
print(f"Extra tables: {extra_tables or '(none)'}")

# COMMAND ----------

# MAGIC %md ## 1. Build the `serialized_space` payload

# COMMAND ----------

def newid() -> str:
    return uuid.uuid4().hex

# Pre-fetch table columns so we can write column_configs telling Genie which fields
# carry entity-matchable values (the id columns and the type/label-ish columns).
edge_columns = [f.name for f in spark.table(EDGE_TABLE).schema.fields]
node_columns = [f.name for f in spark.table(NODE_SOURCE).schema.fields]

def col_configs(columns, entity_match_cols):
    out = []
    for c in sorted(columns):
        cfg = {"column_name": c, "enable_format_assistance": True}
        if c in entity_match_cols:
            cfg["enable_entity_matching"] = True
        out.append(cfg)
    return out

# Enable entity matching on the id columns (so Genie can match user-typed entity
# ids) and on any column that smells like a type/label/name grouping.
likely_grouping = {"type", "Type", "name", "Name", "label", "Label",
                   "subject_type", "object_type"}
edge_entity_cols = {c for c in edge_columns
                    if c in {EDGE_SRC, EDGE_DST, EDGE_LABEL} | likely_grouping}
node_entity_cols = {c for c in node_columns
                    if c in {NODE_SOURCE_ID} | likely_grouping}

# data_sources.tables — sort by identifier (Genie requires this)
tables_block = []
for fqn in sorted(set([
    EDGE_TABLE,
    NODE_SOURCE,
    f"{OUT_NS}.edges_enriched",
    f"{OUT_NS}.downstream_khop",
    f"{OUT_NS}.upstream_khop",
    f"{OUT_NS}.node_degree",
] + extra_tables)):
    if fqn == EDGE_TABLE:
        cfg = col_configs(edge_columns, edge_entity_cols)
    elif fqn == NODE_SOURCE:
        cfg = col_configs(node_columns, node_entity_cols)
    else:
        cfg = []
    entry = {"identifier": fqn}
    if cfg:
        entry["column_configs"] = cfg
    tables_block.append(entry)

# Optional provenance hints — only mentioned when the edge table carries them.
provenance_lines = []
if "confidence" in edge_columns:
    provenance_lines.append("- `confidence` (0.0–1.0) scores how certain the extraction is. When users ask for \"high-confidence\" or \"reliable\" relationships, filter `confidence >= 0.8` (or the threshold they give).")
if "source_method" in edge_columns:
    provenance_lines.append("- `source_method` records how the edge was extracted (e.g. by which agent or rule). Users can filter or group by it to audit provenance.")
if "properties" in edge_columns:
    provenance_lines.append("- `properties` is a JSON string of extra edge attributes; use `get_json_object(properties, '$.key')` to reach into it when asked.")
provenance_text = ("\nPROVENANCE COLUMNS ON EDGES:\n" + "\n".join(provenance_lines) + "\n") if provenance_lines else ""

predicates_hint = ("Each k-hop row also carries `predicates`, the ordered list of relationship labels walked."
                   if EDGE_LABEL else "")
label_clause = f", labeled by `{EDGE_LABEL}`" if EDGE_LABEL else ""
label_example = f", \"show all `{EDGE_LABEL}` = 'works_for' edges\"" if EDGE_LABEL else ""
label_rule = (f"- When users name a relationship in words (owns, works for, cites, ...), match it against the `{EDGE_LABEL}` column."
              if EDGE_LABEL else "")

# Curated instructions — tell the LLM how to use the views.
instructions_text = f"""
This Genie space answers questions about a knowledge graph stored as subject–predicate–object triplets.

DATA MODEL:
- `{EDGE_TABLE}` is the edge (triplet) table. Each row is one directed relationship from `{EDGE_SRC}` to `{EDGE_DST}`{label_clause}. An "entity" is any value appearing in `{EDGE_SRC}` or `{EDGE_DST}`.
- `{NODE_SOURCE}` is the entity (node) table: one row per entity with its id (`{NODE_SOURCE_ID}`), its type, and its degree.
{provenance_text}
DERIVED VIEWS (prefer these for analyst questions):
- `{OUT_NS}.edges_enriched` is `{EDGE_TABLE}` joined to both endpoints with `source_*` and `dest_*` prefixed attribute columns. Use this for ANY question about the relationships of an entity or about edges with conditions on the endpoints ("show everything connected to X", "list all edges from a Person to a Company"{label_example}).
- `{OUT_NS}.downstream_khop` answers "what is reachable FROM entity X" — every entity reachable by following edges forward in <= K hops, with the node `path`. Filter `WHERE start_id = '<id>'` and `WHERE hop <= <N>`. {predicates_hint}
- `{OUT_NS}.upstream_khop` is the same in reverse — "what points AT entity X".
- `{OUT_NS}.node_degree` gives in/out/total degree per entity. Use for hub identification and isolated-entity lookups.

RULES:
- For multi-hop reachability questions ("what is connected to X within N hops", "what does X eventually lead to", "what depends on X"), ALWAYS use the relevant `*_khop` view, not raw `{EDGE_TABLE}`.
- For "what connects X and Y" / "is there a path between X and Y", query `downstream_khop` with `WHERE start_id = '<X>' AND reached_id = '<Y>'` and return the shortest `path` (lowest `hop`). If nothing is found, also try the reverse direction (start from Y).
- For direct relationships of a single entity, use `edges_enriched` filtered on `{EDGE_SRC}` or `{EDGE_DST}`.
- For "top hubs", "most connected", "rank by connections", use `node_degree` ordered by `total_degree`.
{label_rule}
- When users refer to an entity by name rather than id, resolve the id from `{NODE_SOURCE}` first (or search both `{EDGE_SRC}` and `{EDGE_DST}`), then plug that id into the traversal views.
- Always cite the view you used in the answer.
""".strip()

instructions_block = [{"id": newid(), "content": [instructions_text]}]

# Sample questions (shown in the empty-state of the chat)
sample_questions = [
    {"id": newid(), "question": [q]} for q in [
        "Which 10 entities are the biggest hubs in the graph?",
        "What entities are reachable from 'REPLACE_WITH_REAL_ID' within 3 hops?",
        "What connects 'ENTITY_A_ID' and 'ENTITY_B_ID'?",
        "List all relationships of entity 'REPLACE_WITH_REAL_ID', grouped by predicate.",
        "How many edges does each predicate have, and what is the average confidence per predicate?",
        "Are there any isolated entities (degree = 0)?",
    ]
]

# Example SQL — concrete patterns the LLM can crib from
example_sqls = [
    {
        "id": newid(),
        "question": ["What is reachable from a given entity within 3 hops?"],
        "sql": [
            f"SELECT reached_id, hop, path\n",
            f"FROM `{OUT_CATALOG}`.`{OUT_SCHEMA}`.`downstream_khop`\n",
            f"WHERE start_id = 'REPLACE_WITH_REAL_ID' AND hop <= 3\n",
            f"ORDER BY hop, reached_id"
        ],
    },
    {
        "id": newid(),
        "question": ["What connects entity A and entity B?"],
        "sql": [
            f"SELECT start_id, reached_id, hop, path\n",
            f"FROM `{OUT_CATALOG}`.`{OUT_SCHEMA}`.`downstream_khop`\n",
            f"WHERE start_id = 'ENTITY_A_ID' AND reached_id = 'ENTITY_B_ID'\n",
            f"ORDER BY hop\n",
            f"LIMIT 5"
        ],
    },
    {
        "id": newid(),
        "question": ["List the 10 entities with the highest total degree"],
        "sql": [
            f"SELECT id, in_degree, out_degree, total_degree\n",
            f"FROM `{OUT_CATALOG}`.`{OUT_SCHEMA}`.`node_degree`\n",
            f"ORDER BY total_degree DESC\n",
            f"LIMIT 10"
        ],
    },
]
if EDGE_LABEL and "confidence" in edge_columns:
    example_sqls.append({
        "id": newid(),
        "question": ["Show the high-confidence relationships of an entity, grouped by predicate"],
        "sql": [
            f"SELECT `{EDGE_LABEL}`, COUNT(*) AS n_edges, AVG(confidence) AS avg_confidence\n",
            f"FROM `{OUT_CATALOG}`.`{OUT_SCHEMA}`.`edges_enriched`\n",
            f"WHERE (`{EDGE_SRC}` = 'REPLACE_WITH_REAL_ID' OR `{EDGE_DST}` = 'REPLACE_WITH_REAL_ID')\n",
            f"  AND confidence >= 0.8\n",
            f"GROUP BY `{EDGE_LABEL}`\n",
            f"ORDER BY n_edges DESC"
        ],
    })

# Genie requires every id-keyed list to be sorted by id (like tables by identifier)
sample_questions.sort(key=lambda x: x["id"])
example_sqls.sort(key=lambda x: x["id"])
instructions_block.sort(key=lambda x: x["id"])

serialized_space = {
    "version": 2,
    "config": {"sample_questions": sample_questions},
    "data_sources": {"tables": tables_block},
    "instructions": {
        "text_instructions": instructions_block,
        "example_question_sqls": example_sqls,
    },
}

serialized_str = json.dumps(serialized_space)
print(f"serialized_space size: {len(serialized_str):,} chars")
print(f"tables attached: {len(tables_block)}")

# COMMAND ----------

# MAGIC %md ## 2. Look up any existing space with the same title

# COMMAND ----------

def list_spaces():
    out = []
    page_token = None
    while True:
        params = {"page_size": 100}
        if page_token:
            params["page_token"] = page_token
        r = requests.get(f"{HOST}/api/2.0/genie/spaces", headers=HEADERS, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        out.extend(data.get("spaces", []))
        page_token = data.get("next_page_token")
        if not page_token:
            break
    return out

existing = [s for s in list_spaces() if s.get("title") == SPACE_TITLE]
print(f"Found {len(existing)} existing space(s) with title '{SPACE_TITLE}'")
for s in existing:
    print(f"  - id={s.get('space_id')}  parent={s.get('parent_path')}")

# COMMAND ----------

# MAGIC %md ## 3. Create or update

# COMMAND ----------

def create_space():
    body = {
        "title": SPACE_TITLE,
        "description": SPACE_DESC,
        "parent_path": PARENT_PATH,
        "warehouse_id": WAREHOUSE_ID,
        "serialized_space": serialized_str,
    }
    r = requests.post(f"{HOST}/api/2.0/genie/spaces", headers=HEADERS, data=json.dumps(body), timeout=120)
    r.raise_for_status()
    return r.json()

def patch_space(space_id):
    body = {
        "title": SPACE_TITLE,
        "description": SPACE_DESC,
        "warehouse_id": WAREHOUSE_ID,
        "serialized_space": serialized_str,
    }
    r = requests.patch(f"{HOST}/api/2.0/genie/spaces/{space_id}", headers=HEADERS, data=json.dumps(body), timeout=120)
    r.raise_for_status()
    return r.json()

if not existing:
    result = create_space()
    print(f"Created space {result.get('space_id')}")
elif ON_CONFLICT == "create_new":
    result = create_space()
    print(f"Created additional space {result.get('space_id')} (existing left alone)")
elif ON_CONFLICT == "update":
    sid = existing[0]["space_id"]
    result = patch_space(sid)
    print(f"Updated existing space {sid}")
elif ON_CONFLICT == "skip":
    result = existing[0]
    print(f"Skipped — existing space {result.get('space_id')} unchanged")
else:
    raise ValueError(f"Unknown on_conflict: {ON_CONFLICT}")

space_id = result.get("space_id") or (existing[0]["space_id"] if existing else None)
space_url = f"{HOST.rstrip('/')}/genie/rooms/{space_id}" if space_id else None

print(f"\n  Space ID:  {space_id}")
print(f"  Space URL: {space_url}\n")
print("Open the URL above to start chatting with your knowledge graph.")

dbutils.notebook.exit(json.dumps({"space_id": space_id, "space_url": space_url}))
