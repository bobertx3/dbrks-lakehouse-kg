-- 03_traversal_functions.sql — on-demand bounded traversal UC functions over gold_triplets
--
-- PARAMETER TOKENS: `${catalog}` and `${schema}` are plain placeholder tokens,
-- not native SQL parameters. UC function DDL cannot be parameterized natively,
-- so notebooks/05_register_uc_functions.py substitutes the tokens with its
-- widget values (default: main.knowledge_graph) and executes each statement
-- via spark.sql.
--
-- Design notes:
--   * All multi-hop walks are FIXED-HOP UNION ALL CTEs, not recursive CTEs.
--     Recursive CTEs require UNION ALL and are unreliable on DBSQL and inside
--     UC function bodies -- fixed-hop unions are the dependable pattern.
--   * Hops are capped (khop at 3, connection_path at 4, subgraph_edges at 2)
--     because each extra hop is another self-join of the edge list.
--   * Every function has row LIMITs sized for use as an agent/Genie tool.
--   * The graph is treated as undirected for reachability (edges are unioned
--     in both directions) so traversal works regardless of predicate direction.
--
-- Keep statements semicolon-terminated with no semicolons inside comments or
-- string literals: notebook 05 splits this file on semicolons.

-- 1-hop typed neighbors of an entity, both edge directions
CREATE OR REPLACE FUNCTION ${catalog}.${schema}.neighbors(
  entity_id STRING COMMENT 'Entity ID to look up, matched against subject_id and object_id'
)
RETURNS TABLE (
  predicate     STRING,
  neighbor_id   STRING,
  neighbor_type STRING,
  direction     STRING,
  confidence    DOUBLE
)
COMMENT 'Direct (1-hop) neighbors of an entity in the knowledge graph, in both edge directions. direction is out when the entity is the subject and in when it is the object. Use this first when exploring an unfamiliar entity.'
RETURN
  SELECT * FROM (
    SELECT predicate, object_id AS neighbor_id, object_type AS neighbor_type,
           'out' AS direction, confidence
    FROM ${catalog}.${schema}.gold_triplets
    WHERE subject_id = neighbors.entity_id
    UNION ALL
    SELECT predicate, subject_id AS neighbor_id, subject_type AS neighbor_type,
           'in' AS direction, confidence
    FROM ${catalog}.${schema}.gold_triplets
    WHERE object_id = neighbors.entity_id
  ) n
  ORDER BY confidence DESC
  LIMIT 500;

-- k-hop neighborhood (k capped at 3), undirected reachability
CREATE OR REPLACE FUNCTION ${catalog}.${schema}.khop(
  entity_id STRING COMMENT 'Entity ID to expand from',
  k         INT    COMMENT 'Number of hops to expand, 1 to 3. Values above 3 are capped at 3.'
)
RETURNS TABLE (
  neighbor_id STRING,
  hops        INT
)
COMMENT 'All entities within k hops of an entity (k capped at 3), with the minimum hop distance to each. Treats edges as undirected. Use for "what is near X" questions.'
RETURN
  WITH e AS (
    SELECT subject_id AS src, object_id AS dst FROM ${catalog}.${schema}.gold_triplets
    UNION
    SELECT object_id AS src, subject_id AS dst FROM ${catalog}.${schema}.gold_triplets
  ),
  lv AS (
    SELECT e1.dst AS neighbor_id, 1 AS hops
    FROM e e1 WHERE e1.src = khop.entity_id
    UNION ALL
    SELECT e2.dst, 2
    FROM e e1 JOIN e e2 ON e1.dst = e2.src
    WHERE e1.src = khop.entity_id
    UNION ALL
    SELECT e3.dst, 3
    FROM e e1 JOIN e e2 ON e1.dst = e2.src JOIN e e3 ON e2.dst = e3.src
    WHERE e1.src = khop.entity_id
  )
  SELECT neighbor_id, MIN(hops) AS hops
  FROM lv
  WHERE neighbor_id <> khop.entity_id AND hops <= LEAST(khop.k, 3)
  GROUP BY neighbor_id
  ORDER BY hops, neighbor_id
  LIMIT 5000;

-- Connection paths between two entities, up to 4 hops, undirected
CREATE OR REPLACE FUNCTION ${catalog}.${schema}.connection_path(
  a STRING COMMENT 'First entity ID',
  b STRING COMMENT 'Second entity ID'
)
RETURNS TABLE (
  hops INT,
  path STRING
)
COMMENT 'Simple paths connecting two entities up to 4 hops apart, formatted as id -> id -> id, shortest first. Treats edges as undirected. Empty result means no path within 4 hops. Use for "how is X connected to Y" questions.'
RETURN
  WITH e AS (
    SELECT subject_id AS src, object_id AS dst FROM ${catalog}.${schema}.gold_triplets
    UNION
    SELECT object_id AS src, subject_id AS dst FROM ${catalog}.${schema}.gold_triplets
  ),
  paths AS (
    SELECT 1 AS hops, concat_ws(' -> ', connection_path.a, connection_path.b) AS path
    FROM e WHERE src = connection_path.a AND dst = connection_path.b
    UNION ALL
    SELECT 2, concat_ws(' -> ', connection_path.a, e1.dst, connection_path.b)
    FROM e e1 JOIN e e2 ON e1.dst = e2.src
    WHERE e1.src = connection_path.a AND e2.dst = connection_path.b
      AND e1.dst NOT IN (connection_path.a, connection_path.b)
    UNION ALL
    SELECT 3, concat_ws(' -> ', connection_path.a, e1.dst, e2.dst, connection_path.b)
    FROM e e1 JOIN e e2 ON e1.dst = e2.src JOIN e e3 ON e2.dst = e3.src
    WHERE e1.src = connection_path.a AND e3.dst = connection_path.b
      AND e1.dst NOT IN (connection_path.a, connection_path.b)
      AND e2.dst NOT IN (connection_path.a, connection_path.b)
      AND e1.dst <> e2.dst
    UNION ALL
    SELECT 4, concat_ws(' -> ', connection_path.a, e1.dst, e2.dst, e3.dst, connection_path.b)
    FROM e e1 JOIN e e2 ON e1.dst = e2.src JOIN e e3 ON e2.dst = e3.src JOIN e e4 ON e3.dst = e4.src
    WHERE e1.src = connection_path.a AND e4.dst = connection_path.b
      AND e1.dst NOT IN (connection_path.a, connection_path.b)
      AND e2.dst NOT IN (connection_path.a, connection_path.b)
      AND e3.dst NOT IN (connection_path.a, connection_path.b)
      AND e1.dst <> e2.dst AND e2.dst <> e3.dst AND e1.dst <> e3.dst
  )
  SELECT DISTINCT hops, path
  FROM paths
  ORDER BY hops, path
  LIMIT 100;

-- Full edge rows of the k-hop neighborhood (k capped at 2) for visualization
CREATE OR REPLACE FUNCTION ${catalog}.${schema}.subgraph_edges(
  entity_id STRING COMMENT 'Anchor entity ID at the center of the subgraph',
  k         INT    COMMENT 'Number of hops to include, 1 or 2. Values above 2 are capped at 2.'
)
RETURNS TABLE (
  subject_id   STRING,
  subject_type STRING,
  predicate    STRING,
  object_id    STRING,
  object_type  STRING,
  confidence   DOUBLE,
  hop          INT
)
COMMENT 'Complete edge rows of the neighborhood within k hops of an entity (k capped at 2, 1000 rows max). Returns the same columns as gold_triplets plus hop, ready to render as a subgraph visualization.'
RETURN
  WITH base AS (
    SELECT subject_id, subject_type, predicate, object_id, object_type, confidence, 1 AS hop
    FROM ${catalog}.${schema}.gold_triplets g
    WHERE g.subject_id = subgraph_edges.entity_id OR g.object_id = subgraph_edges.entity_id
  ),
  frontier AS (
    SELECT DISTINCT
      CASE WHEN b.subject_id = subgraph_edges.entity_id THEN b.object_id ELSE b.subject_id END AS node_id
    FROM base b
  ),
  second_hop AS (
    SELECT g.subject_id, g.subject_type, g.predicate, g.object_id, g.object_type, g.confidence, 2 AS hop
    FROM ${catalog}.${schema}.gold_triplets g
    JOIN frontier f ON g.subject_id = f.node_id OR g.object_id = f.node_id
    WHERE LEAST(subgraph_edges.k, 2) >= 2
      AND g.subject_id <> subgraph_edges.entity_id
      AND g.object_id <> subgraph_edges.entity_id
  )
  SELECT * FROM (
    SELECT subject_id, subject_type, predicate, object_id, object_type, confidence, hop FROM base
    UNION
    SELECT subject_id, subject_type, predicate, object_id, object_type, confidence, hop FROM second_hop
  ) s
  ORDER BY hop, subject_id, object_id
  LIMIT 1000;
