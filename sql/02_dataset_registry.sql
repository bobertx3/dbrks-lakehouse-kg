-- 02_dataset_registry.sql — control table listing every onboarded dataset
--
-- PARAMETER TOKENS: `${catalog}` and `${schema}` are plain placeholder tokens,
-- not native SQL parameters. notebooks/05_register_uc_functions.py substitutes
-- them with its widget values (default: main.knowledge_graph) and executes each
-- statement via spark.sql. UC DDL cannot be parameterized natively.
--
-- One row per dataset onboarded onto the starter kit. Apps, agents, and jobs
-- read this table to discover which knowledge graphs exist, where their triplet
-- tables live, and how to describe them to users -- instead of hardcoding
-- catalog/schema names per dataset.
--
-- Keep statements semicolon-terminated with no semicolons inside comments or
-- string literals: notebook 05 splits this file on semicolons.

CREATE TABLE IF NOT EXISTS ${catalog}.${schema}.dataset_registry (
  dataset_key        STRING NOT NULL COMMENT 'Stable machine key for the dataset, e.g. supply_chain or corp_ownership. Unique by convention -- upsert with MERGE on this key.',
  display_name       STRING           COMMENT 'Human-friendly name shown in UIs, e.g. Supply Chain Graph.',
  triplet_table      STRING NOT NULL COMMENT 'Fully qualified name of the dataset gold_triplets table, e.g. main.knowledge_graph.gold_triplets. Consumers read edges from here.',
  description        STRING           COMMENT 'One-paragraph domain description of what the graph contains. Agents can inject this into system prompts as dataset context.',
  sample_prompts     ARRAY<STRING>    COMMENT 'Example natural-language questions this graph can answer. Surfaced as suggested prompts in chat UIs.',
  owner              STRING           COMMENT 'Email or team responsible for the dataset.',
  agent_endpoint     STRING           COMMENT 'Optional model-serving endpoint name of an agent scoped to this dataset. NULL if the dataset has no dedicated agent.',
  enabled            BOOLEAN          COMMENT 'Soft on/off switch. Consumers should ignore rows where enabled = false.',
  created_at         TIMESTAMP        COMMENT 'When the dataset was first registered.',
  updated_at         TIMESTAMP        COMMENT 'When this row was last modified. Update on every MERGE.'
)
COMMENT 'Registry of knowledge-graph datasets onboarded onto the starter kit: one row per dataset with its triplet table location, description, sample prompts, and optional agent endpoint. Query this to discover available graphs.';

-- Example upsert for onboarding a dataset (run manually after adapting values,
-- note there is intentionally no trailing semicolon inside this comment block):
--
--   MERGE INTO <catalog>.<schema>.dataset_registry AS t
--   USING (SELECT 'my_dataset' AS dataset_key) AS s
--   ON t.dataset_key = s.dataset_key
--   WHEN MATCHED THEN UPDATE SET t.updated_at = current_timestamp()
--   WHEN NOT MATCHED THEN INSERT
--     (dataset_key, display_name, triplet_table, description, sample_prompts,
--      owner, agent_endpoint, enabled, created_at, updated_at)
--   VALUES
--     ('my_dataset', 'My Dataset Graph',
--      '<catalog>.<schema>.gold_triplets',
--      'A knowledge graph of ... built from ...',
--      array('Who is connected to X?', 'Which entities are most central?'),
--      'owner@example.com', NULL, TRUE, current_timestamp(), current_timestamp())
