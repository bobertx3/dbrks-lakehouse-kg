-- 01_gold_triplets.sql — the core triplet contract for the lakehouse knowledge graph
--
-- PARAMETER TOKENS: `${catalog}` and `${schema}` are plain placeholder tokens,
-- not native SQL parameters. Unity Catalog DDL (tables and especially functions)
-- cannot be parameterized natively, so notebooks/05_register_uc_functions.py
-- reads this file, substitutes the tokens with its widget values
-- (default: main.knowledge_graph), splits on statement boundaries, and executes
-- each statement via spark.sql. Run that notebook rather than pasting this file
-- into a SQL editor -- or find/replace the tokens yourself.
--
-- gold_triplets is the single contract every layer of the starter kit agrees on:
--   * extraction agents (lakehouse_kg / notebooks 01-04) APPEND rows here
--   * traversal functions (03_traversal_functions.sql) read it at query time
--   * batch analytics (notebooks 06/07) read it and precompute derived tables
--   * Genie and agents discover its meaning through the comments below
--
-- Keep statements semicolon-terminated with no semicolons inside comments or
-- string literals: notebook 05 splits this file on semicolons.

CREATE TABLE IF NOT EXISTS ${catalog}.${schema}.gold_triplets (
  subject_id    STRING NOT NULL COMMENT 'ID of the subject (head) entity of the triplet. Joinable to object_id of other rows to walk the graph.',
  subject_type  STRING           COMMENT 'Entity type of the subject, e.g. company, person, document, part. Free-form but should be consistent within a dataset.',
  predicate     STRING NOT NULL COMMENT 'Relationship name connecting subject to object, e.g. OWNS_ACCOUNT, located_at, supplies. Casing is the producer choice (agentic packs use UPPER_SNAKE) - keep it consistent within a dataset.',
  object_id     STRING NOT NULL COMMENT 'ID of the object (tail) entity of the triplet. Joinable to subject_id of other rows to walk the graph.',
  object_type   STRING           COMMENT 'Entity type of the object, e.g. company, address, chemical, topic. Free-form but should be consistent within a dataset.',
  confidence    DOUBLE           COMMENT 'Extraction confidence in [0.0, 1.0]. Deterministic joins are typically 1.0 and LLM-extracted relations lower. Filter with confidence >= threshold for high-precision views.',
  source_agent  STRING           COMMENT 'Name of the agent or pipeline that produced this triplet, e.g. schema_profiler, entity_extractor, human_review. Provenance for auditing.',
  source_method STRING           COMMENT 'How the triplet was derived, e.g. fk_join, llm_extraction, rule, embedding_match. Finer-grained provenance than source_agent.',
  properties    STRING           COMMENT 'Optional JSON object string with edge attributes, e.g. weights, timestamps, source row keys. NULL when the edge has no extra attributes. Parse with from_json or get_json_object.',
  created_at    TIMESTAMP        COMMENT 'When the triplet was written. Use for incremental processing and time-travel debugging.'
)
COMMENT 'Gold knowledge-graph edge list: one row per (subject, predicate, object) triplet with confidence and provenance. This is the single contract between extraction agents, traversal UC functions, graph analytics, and Genie.';
