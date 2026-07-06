"""Generic, domain-neutral DomainPack.

A sensible default for arbitrary enterprise data: neutral entity types,
neutral predicate names, and Genie views that describe the knowledge graph
without assuming any particular business domain.
"""

from __future__ import annotations

from lakehouse_kg.domain import DomainPack, GenieViewSpec

GENERIC_SCHEMA_PROMPT = """\
You are a knowledge graph expert analyzing database schemas to discover hidden relationships.

Given these table schemas:
{schema_summary}

And these existing relationships already discovered:
{existing_predicates}

Answer the following questions. For each, provide specific triplets in JSON format:

1. What implicit relationships exist between entities that aren't captured by foreign keys?
   (e.g., entities that share behavioral patterns, records that belong to the same category)

2. What derived indicators can be inferred from the data model?
   (e.g., entity age vs activity frequency, geographic spread of events)

3. What temporal relationship patterns are relevant for understanding this data?
   (e.g., rapid entity creation followed by high-value events)

Return a JSON array of triplet objects, each with:
- subject_type, subject_description (describe the entity class, not a specific ID)
- predicate (relationship name in UPPER_SNAKE_CASE)
- object_type, object_description
- confidence (0.0-1.0)
- reasoning (why this relationship matters)

Return ONLY the JSON array, no other text.
"""

GENERIC_ENTITY_PROMPT = """\
You are a data analyst. Given this sample of entity data:

{entity_samples}

Identify any notable patterns or relationships that should be captured as knowledge graph triplets.
Focus on:
1. Entities that appear to be duplicates or near-duplicates (similar names, shared attributes)
2. Unusual entity-to-entity relationships
3. Geographic anomalies
4. Temporal coordination patterns

For each finding, output a JSON object with:
- subject_id, subject_type
- predicate
- object_id, object_type
- confidence (0.0-1.0)
- reasoning

Return ONLY a JSON array.
"""


def generic_pack() -> DomainPack:
    """Build the domain-neutral default pack."""
    return DomainPack(
        name="generic",
        description="Domain-neutral defaults for arbitrary enterprise data.",
        # --- Entity Resolution ---
        entity_type_patterns={
            "Person": ["person", "customer", "user", "employee", "client", "member", "contact"],
            "Organization": ["organization", "org", "company", "vendor", "merchant", "supplier", "business"],
            "Location": ["address", "location", "city", "state", "zip", "country", "region", "geo", "site"],
            "Event": ["event", "transaction", "activity", "incident", "order", "session"],
            "Asset": ["asset", "account", "device", "product", "item", "resource", "equipment"],
            "Document": ["document", "doc", "record", "report", "file", "contract", "invoice"],
        },
        default_entity_type="Entity",
        identity_attribute_keywords=["address", "phone", "email", "ip", "zip", "device"],
        # --- Relationship Agent ---
        fk_predicates={},
        # --- Statistical Agent ---
        statistical_question="What behavioral patterns and statistical anomalies exist in the data?",
        velocity_question="Which entities show unusual event velocity (burst patterns)?",
        amount_question="Which events have values that are statistical outliers?",
        temporal_question="Are there unusual temporal patterns (late-night, weekend bursts)?",
        concentration_question="Do any entities show unusual concentration on specific counterparties?",
        event_table_keywords=["event", "transaction", "txn", "activity", "log", "order"],
        actor_id_columns=["entity_id", "actor_id", "account_id", "user_id", "customer_id"],
        event_id_columns=["event_id", "transaction_id", "activity_id", "order_id", "record_id"],
        counterparty_id_columns=["counterparty_id", "target_id", "merchant_id", "vendor_id", "destination_id"],
        timestamp_columns=["timestamp", "event_time", "event_date", "created_at", "occurred_at"],
        amount_columns=["amount", "value", "quantity", "total"],
        actor_entity_type="Entity",
        event_entity_type="Event",
        counterparty_entity_type="Entity",
        pattern_entity_type="Pattern",
        velocity_predicate="EXHIBITS_VELOCITY_ANOMALY",
        outlier_amount_predicate="HAS_OUTLIER_VALUE",
        high_value_event_predicate="HAS_HIGH_VALUE_EVENT",
        off_hours_predicate="EXHIBITS_OFF_HOURS_PATTERN",
        concentration_predicate="CONCENTRATED_AT_COUNTERPARTY",
        # --- ML Clustering Agent ---
        ml_question="Are there hidden communities or behavioral clusters among entities?",
        ml_clustering_question=(
            "Can entities be grouped by behavioral fingerprint (value patterns, timing, counterparty mix)?"
        ),
        cluster_predicate="BELONGS_TO_CLUSTER",
        similarity_predicate="BEHAVIORALLY_SIMILAR_TO",
        cluster_entity_type="BehavioralCluster",
        # --- LLM Semantic Agent ---
        llm_question="What semantic relationships can an LLM infer from the data?",
        llm_entity_question="Do entity samples reveal notable patterns or relationships?",
        llm_system_prompt="You are a knowledge graph expert. Respond only with valid JSON.",
        llm_schema_prompt=GENERIC_SCHEMA_PROMPT,
        llm_entity_prompt=GENERIC_ENTITY_PROMPT,
        default_llm_predicate="DETECTED_PATTERN",
        # --- Graph Topology Agent ---
        hub_question="Which entities are hubs connecting many others (highly connected nodes)?",
        community_question="Are there tightly connected communities (closely related entity groups)?",
        bridge_question="Which entities bridge different communities (connector nodes)?",
        risk_question="How does anomaly risk propagate through the network from flagged entities?",
        hub_predicate="IS_NETWORK_HUB",
        community_predicate="MEMBER_OF_COMMUNITY",
        bridge_predicate="IS_NETWORK_BRIDGE",
        risk_predicate="HAS_PROXIMITY_RISK",
        risk_seed_predicates=[
            "EXHIBITS_VELOCITY_ANOMALY",
            "HAS_OUTLIER_VALUE",
            "EXHIBITS_OFF_HOURS_PATTERN",
            "IS_NETWORK_HUB",
            "DETECTED_PATTERN",
        ],
        # --- Validation Agent ---
        contradictory_predicate_pairs=[],
        # --- Genie views ---
        genie_views=[
            GenieViewSpec(
                name="entity_relationships",
                description="All direct relationships between entities discovered from foreign keys and shared attributes. Use this to answer questions about how entities are connected.",
                sql_template="""
                    SELECT
                        subject_id,
                        subject_type,
                        predicate AS relationship,
                        object_id,
                        object_type,
                        confidence AS relationship_confidence,
                        source_agent AS discovered_by,
                        source_method AS discovery_method,
                        properties AS relationship_details,
                        created_at
                    FROM {triplet_table}
                    WHERE source_method IN ('foreign_key', 'shared_attribute')
                """,
            ),
            GenieViewSpec(
                name="detected_patterns",
                description="Statistical anomalies and behavioral patterns detected across entities and events. Use this to find entities with unusual velocity, outlier values, off-hours activity, or counterparty concentration.",
                sql_template="""
                    SELECT
                        subject_id AS entity_id,
                        subject_type AS entity_type,
                        predicate AS pattern_type,
                        object_id AS pattern_detail,
                        confidence AS pattern_score,
                        source_method AS detection_method,
                        properties AS pattern_metadata,
                        created_at AS detected_at
                    FROM {triplet_table}
                    WHERE predicate IN (
                        'EXHIBITS_VELOCITY_ANOMALY',
                        'HAS_OUTLIER_VALUE',
                        'EXHIBITS_OFF_HOURS_PATTERN',
                        'CONCENTRATED_AT_COUNTERPARTY',
                        'HAS_HIGH_VALUE_EVENT',
                        'DETECTED_PATTERN'
                    )
                    ORDER BY confidence DESC
                """,
            ),
            GenieViewSpec(
                name="entity_communities",
                description="Network communities and clusters of entities that interact together or share behavioral patterns. Use this to investigate groups of related entities.",
                sql_template="""
                    SELECT
                        subject_id AS entity_id,
                        subject_type AS entity_type,
                        predicate AS membership_type,
                        object_id AS group_id,
                        object_type AS group_type,
                        confidence,
                        properties AS group_metadata,
                        source_method AS clustering_method
                    FROM {triplet_table}
                    WHERE predicate IN (
                        'BELONGS_TO_CLUSTER',
                        'MEMBER_OF_COMMUNITY',
                        'BEHAVIORALLY_SIMILAR_TO'
                    )
                """,
            ),
            GenieViewSpec(
                name="network_key_players",
                description="Hub entities and bridge nodes in the network. Hubs connect many entities. Bridges connect different communities.",
                sql_template="""
                    SELECT
                        subject_id AS entity_id,
                        subject_type AS entity_type,
                        predicate AS network_role,
                        confidence AS role_strength,
                        properties AS role_metrics,
                        source_method AS analysis_method
                    FROM {triplet_table}
                    WHERE predicate IN ('IS_NETWORK_HUB', 'IS_NETWORK_BRIDGE')
                    ORDER BY confidence DESC
                """,
            ),
            GenieViewSpec(
                name="shared_attributes",
                description="Entities that share attribute values like addresses, phone numbers, emails, IPs, or devices. Shared attributes between otherwise unrelated entities are strong linkage signals.",
                sql_template="""
                    SELECT
                        subject_id AS entity_1_id,
                        subject_type AS entity_1_type,
                        predicate AS shared_attribute_type,
                        object_id AS entity_2_id,
                        object_type AS entity_2_type,
                        confidence,
                        properties AS shared_details
                    FROM {triplet_table}
                    WHERE predicate LIKE 'SHARES_%_WITH'
                """,
            ),
            GenieViewSpec(
                name="full_knowledge_graph",
                description="Complete knowledge graph of all generated triplets. Each row is a relationship (edge) between two entities (nodes). Use this for general graph exploration.",
                sql_template="""
                    SELECT
                        subject_id,
                        subject_type,
                        predicate AS relationship,
                        object_id,
                        object_type,
                        confidence,
                        source_agent,
                        source_method,
                        properties,
                        created_at
                    FROM {triplet_table}
                    ORDER BY confidence DESC
                """,
            ),
            GenieViewSpec(
                name="pipeline_execution_log",
                description="Execution log showing which AI agents ran, what questions they asked, how many triplets they generated, and how long they took. Use this for pipeline observability.",
                sql_template="""
                    SELECT *
                    FROM {log_table}
                    ORDER BY executed_at
                """,
            ),
        ],
    )


GENERIC_PACK = generic_pack()
