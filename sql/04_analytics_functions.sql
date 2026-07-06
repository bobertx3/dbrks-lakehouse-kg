-- 04_analytics_functions.sql — serving-layer UC functions over precomputed graph analytics
--
-- PARAMETER TOKENS: `${catalog}` and `${schema}` are plain placeholder tokens,
-- not native SQL parameters. UC function DDL cannot be parameterized natively,
-- so notebooks/05_register_uc_functions.py substitutes the tokens with its
-- widget values (default: main.knowledge_graph) and executes each statement
-- via spark.sql.
--
-- These functions are cheap lookups over two tables PRECOMPUTED in batch by
-- notebooks/06_graph_algorithms.py (CPU, networkx on the driver) or
-- notebooks/07_cugraph_gpu.py (GPU, NVIDIA RAPIDS cuGraph -- the scale-up
-- alternative). Both notebooks write the exact same output schemas, so the
-- functions below work regardless of which engine produced the tables.
-- The CREATE TABLE IF NOT EXISTS statements exist so the functions can be
-- registered before the first analytics run (the tables start empty). The
-- notebooks may add extra columns (e.g. component_id) -- functions reference
-- columns by name so extras are harmless.
--
-- Keep statements semicolon-terminated with no semicolons inside comments or
-- string literals: notebook 05 splits this file on semicolons.

CREATE TABLE IF NOT EXISTS ${catalog}.${schema}.entity_centrality (
  entity_id   STRING COMMENT 'Entity ID, matching subject_id / object_id in gold_triplets.',
  entity_type STRING COMMENT 'Entity type carried over from gold_triplets.',
  pagerank    DOUBLE COMMENT 'PageRank score of the entity in the full graph. Higher means more structurally important.',
  degree      BIGINT COMMENT 'Number of edges touching the entity (undirected degree).',
  betweenness DOUBLE COMMENT 'Approximate betweenness centrality (sampled). NULL when the computation was skipped for scale reasons.'
)
COMMENT 'Precomputed per-entity centrality scores over gold_triplets. Written by notebooks/06_graph_algorithms.py (CPU) or notebooks/07_cugraph_gpu.py (GPU). Refresh by re-running either notebook.';

CREATE TABLE IF NOT EXISTS ${catalog}.${schema}.entity_communities (
  entity_id      STRING COMMENT 'Entity ID, matching subject_id / object_id in gold_triplets.',
  entity_type    STRING COMMENT 'Entity type carried over from gold_triplets.',
  community_id   BIGINT COMMENT 'Louvain community label. Entities sharing a community_id are densely interconnected.',
  community_size BIGINT COMMENT 'Number of entities in this community.'
)
COMMENT 'Precomputed Louvain community assignment per entity over gold_triplets. Written by notebooks/06_graph_algorithms.py (CPU) or notebooks/07_cugraph_gpu.py (GPU). Refresh by re-running either notebook.';

-- Which community an entity belongs to (scalar lookup)
CREATE OR REPLACE FUNCTION ${catalog}.${schema}.cluster_of(
  entity_id STRING COMMENT 'Entity ID to look up'
)
RETURNS BIGINT
COMMENT 'Louvain community id an entity belongs to, or NULL if the entity is not in the precomputed communities table. Feed the result to members_of_cluster to list the rest of the community.'
RETURN (
  SELECT c.community_id
  FROM ${catalog}.${schema}.entity_communities c
  WHERE c.entity_id = cluster_of.entity_id
  LIMIT 1
);

-- All entities in a community
CREATE OR REPLACE FUNCTION ${catalog}.${schema}.members_of_cluster(
  community_id BIGINT COMMENT 'Community id, e.g. from cluster_of'
)
RETURNS TABLE (
  entity_id      STRING,
  entity_type    STRING,
  community_size BIGINT
)
COMMENT 'All entities assigned to a given Louvain community (up to 1000 rows). Use after cluster_of to answer "what else is in this cluster".'
RETURN
  SELECT c.entity_id, c.entity_type, c.community_size
  FROM ${catalog}.${schema}.entity_communities c
  WHERE c.community_id = members_of_cluster.community_id
  ORDER BY c.entity_id
  LIMIT 1000;

-- Top-n most central entities in the graph
CREATE OR REPLACE FUNCTION ${catalog}.${schema}.top_central_entities(
  n INT COMMENT 'How many entities to return, capped at 1000'
)
RETURNS TABLE (
  entity_id   STRING,
  entity_type STRING,
  pagerank    DOUBLE,
  degree      BIGINT,
  betweenness DOUBLE
)
COMMENT 'The n most central entities in the graph by PageRank (n capped at 1000), with degree and sampled betweenness. Use for "which entities matter most" questions.'
RETURN
  SELECT entity_id, entity_type, pagerank, degree, betweenness
  FROM (
    SELECT c.entity_id, c.entity_type, c.pagerank, c.degree, c.betweenness,
           row_number() OVER (ORDER BY c.pagerank DESC, c.entity_id) AS rn
    FROM ${catalog}.${schema}.entity_centrality c
  ) ranked
  WHERE rn <= LEAST(top_central_entities.n, 1000);

-- Do two entities share a community?
CREATE OR REPLACE FUNCTION ${catalog}.${schema}.shared_community(
  a STRING COMMENT 'First entity ID',
  b STRING COMMENT 'Second entity ID'
)
RETURNS TABLE (
  entity_a       STRING,
  entity_b       STRING,
  community_a    BIGINT,
  community_b    BIGINT,
  same_community BOOLEAN,
  community_size BIGINT
)
COMMENT 'Single-row comparison of the Louvain communities of two entities. same_community is true when both entities are in the same community, and community_size then gives its size. NULL community ids mean the entity was not found.'
RETURN
  SELECT shared_community.a AS entity_a,
         shared_community.b AS entity_b,
         ca.community_id AS community_a,
         cb.community_id AS community_b,
         (ca.community_id IS NOT NULL AND ca.community_id = cb.community_id) AS same_community,
         CASE WHEN ca.community_id = cb.community_id THEN ca.community_size END AS community_size
  FROM (SELECT 1 AS one) anchor
  LEFT JOIN ${catalog}.${schema}.entity_communities ca ON ca.entity_id = shared_community.a
  LEFT JOIN ${catalog}.${schema}.entity_communities cb ON cb.entity_id = shared_community.b;
