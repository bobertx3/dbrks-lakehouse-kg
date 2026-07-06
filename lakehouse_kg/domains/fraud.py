"""Fraud detection DomainPack — the shipped worked example.

Reproduces the original fraud-tuned pipeline behavior: entity-type patterns
for customers/accounts/merchants/devices, FK predicate names, fraud-flavored
LLM prompts, statistical fraud-pattern predicates, and the fraud Genie views.
Pairs with the synthetic data from lakehouse_kg.generators.synthetic_fraud.
"""

from __future__ import annotations

from lakehouse_kg.domain import DomainPack, GenieViewSpec

FRAUD_SCHEMA_PROMPT = """\
You are a fraud detection expert analyzing database schemas to discover hidden relationships.

Given these table schemas from a fraud detection system:
{schema_summary}

And these existing relationships already discovered:
{existing_predicates}

Answer the following questions. For each, provide specific triplets in JSON format:

1. What implicit relationships exist between entities that aren't captured by foreign keys?
   (e.g., customers who share behavioral patterns, merchants in the same fraud category)

2. What derived risk indicators can be inferred from the data model?
   (e.g., account age vs transaction frequency, geographic spread of transactions)

3. What temporal relationship patterns are relevant for fraud detection?
   (e.g., rapid account creation followed by high-value transactions)

Return a JSON array of triplet objects, each with:
- subject_type, subject_description (describe the entity class, not a specific ID)
- predicate (relationship name in UPPER_SNAKE_CASE)
- object_type, object_description
- confidence (0.0-1.0)
- reasoning (why this relationship matters for fraud detection)

Return ONLY the JSON array, no other text.
"""

FRAUD_ENTITY_PROMPT = """\
You are a fraud detection analyst. Given this sample of entity data:

{entity_samples}

Identify any suspicious patterns or relationships that should be captured as knowledge graph triplets.
Focus on:
1. Entities that appear to be synthetic identities (similar names, shared attributes)
2. Unusual account-merchant relationships
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


def fraud_pack() -> DomainPack:
    """Build the fraud detection pack (original pipeline behavior)."""
    return DomainPack(
        name="fraud",
        description="Fraud detection tuning: customers, accounts, merchants, devices, and fraud patterns.",
        # --- Entity Resolution ---
        entity_type_patterns={
            "Person": ["customer", "user", "person", "employee", "client", "member"],
            "Account": ["account", "acct", "wallet"],
            "Transaction": ["transaction", "txn", "transfer", "payment", "order"],
            "Merchant": ["merchant", "vendor", "seller", "store", "shop"],
            "Device": ["device", "terminal", "machine", "phone"],
            "Location": ["address", "location", "city", "state", "zip", "country", "geo"],
            "Alert": ["alert", "flag", "warning", "case", "incident"],
            "Card": ["card", "instrument"],
        },
        default_entity_type="Entity",
        identity_attribute_keywords=["address", "phone", "email", "ip", "zip", "device", "ssn"],
        # --- Relationship Agent ---
        fk_predicates={
            ("customers", "accounts"): "OWNS_ACCOUNT",
            ("accounts", "customers"): "BELONGS_TO_CUSTOMER",
            ("transactions", "accounts"): "DEBITED_FROM",
            ("transactions", "merchants"): "TRANSACTED_AT",
            ("devices", "customers"): "USED_BY",
            ("customers", "devices"): "USES_DEVICE",
            ("alerts", "transactions"): "FLAGGED_TRANSACTION",
        },
        # --- Statistical Agent ---
        statistical_question="What behavioral patterns and statistical anomalies suggest fraud risk?",
        velocity_question="Which accounts show unusual transaction velocity (burst patterns)?",
        amount_question="Which transactions have amounts that are statistical outliers?",
        temporal_question="Are there unusual temporal patterns (late-night, weekend bursts)?",
        concentration_question="Do any accounts show unusual concentration on specific merchants?",
        event_table_keywords=["transaction", "txn", "payment"],
        actor_id_columns=["account_id", "acct_id"],
        event_id_columns=["transaction_id", "txn_id"],
        counterparty_id_columns=["merchant_id", "merch_id"],
        timestamp_columns=["timestamp", "transaction_date", "txn_date", "created_at"],
        amount_columns=["amount", "txn_amount", "transaction_amount"],
        actor_entity_type="Account",
        event_entity_type="Transaction",
        counterparty_entity_type="Merchant",
        pattern_entity_type="Pattern",
        velocity_predicate="EXHIBITS_VELOCITY_ANOMALY",
        outlier_amount_predicate="HAS_OUTLIER_AMOUNT",
        high_value_event_predicate="HAS_HIGH_VALUE_TRANSACTION",
        off_hours_predicate="EXHIBITS_OFF_HOURS_PATTERN",
        concentration_predicate="CONCENTRATED_AT_MERCHANT",
        # --- ML Clustering Agent ---
        ml_question="Are there hidden communities or behavioral clusters suggesting coordinated fraud?",
        ml_clustering_question=(
            "Can accounts be grouped by behavioral fingerprint (amount patterns, timing, merchant mix)?"
        ),
        cluster_predicate="BELONGS_TO_CLUSTER",
        similarity_predicate="BEHAVIORALLY_SIMILAR_TO",
        cluster_entity_type="BehavioralCluster",
        # --- LLM Semantic Agent ---
        llm_question="What semantic relationships and fraud indicators can an LLM infer from the data?",
        llm_entity_question="Do entity samples reveal suspicious patterns like synthetic identities?",
        llm_system_prompt="You are a fraud detection expert. Respond only with valid JSON.",
        llm_schema_prompt=FRAUD_SCHEMA_PROMPT,
        llm_entity_prompt=FRAUD_ENTITY_PROMPT,
        default_llm_predicate="SUSPICIOUS_PATTERN",
        # --- Graph Topology Agent ---
        hub_question="Which entities are hubs connecting many others (potential fraud coordinators)?",
        community_question="Are there tightly connected communities (potential fraud rings)?",
        bridge_question="Which entities bridge different communities (potential money mules)?",
        risk_question="How does fraud risk propagate through the network from known suspicious entities?",
        hub_predicate="IS_NETWORK_HUB",
        community_predicate="MEMBER_OF_COMMUNITY",
        bridge_predicate="IS_NETWORK_BRIDGE",
        risk_predicate="HAS_PROXIMITY_RISK",
        risk_seed_predicates=[
            "EXHIBITS_VELOCITY_ANOMALY",
            "HAS_OUTLIER_AMOUNT",
            "EXHIBITS_OFF_HOURS_PATTERN",
            "IS_NETWORK_HUB",
            "SUSPICIOUS_PATTERN",
        ],
        # --- Validation Agent ---
        contradictory_predicate_pairs=[
            ("OWNS_ACCOUNT", "DOES_NOT_OWN_ACCOUNT"),
            ("IS_LEGITIMATE", "IS_FRAUDULENT"),
        ],
        # --- Genie views ---
        genie_views=[
            GenieViewSpec(
                name="fraud_entity_relationships",
                description="All direct relationships between fraud-detection entities (customers, accounts, merchants, devices). Use this to answer questions about who owns what, who transacts where, and what devices are used.",
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
                name="suspicious_patterns",
                description="Statistical anomalies and suspicious behavioral patterns detected across accounts and transactions. Use this to find accounts with unusual velocity, high amounts, off-hours activity, or merchant concentration.",
                sql_template="""
                    SELECT
                        subject_id AS entity_id,
                        subject_type AS entity_type,
                        predicate AS pattern_type,
                        object_id AS pattern_detail,
                        confidence AS suspicion_score,
                        source_method AS detection_method,
                        properties AS pattern_metadata,
                        created_at AS detected_at
                    FROM {triplet_table}
                    WHERE predicate IN (
                        'EXHIBITS_VELOCITY_ANOMALY',
                        'HAS_OUTLIER_AMOUNT',
                        'EXHIBITS_OFF_HOURS_PATTERN',
                        'CONCENTRATED_AT_MERCHANT',
                        'HAS_HIGH_VALUE_TRANSACTION',
                        'SUSPICIOUS_PATTERN'
                    )
                    ORDER BY confidence DESC
                """,
            ),
            GenieViewSpec(
                name="fraud_risk_scores",
                description="Risk scores for entities based on network proximity to known suspicious activity. Higher scores mean closer association with fraud indicators. Use this to find high-risk customers or accounts.",
                sql_template="""
                    SELECT
                        subject_id AS entity_id,
                        subject_type AS entity_type,
                        confidence AS risk_score,
                        properties AS risk_factors,
                        source_method AS scoring_method,
                        created_at AS scored_at
                    FROM {triplet_table}
                    WHERE predicate = 'HAS_PROXIMITY_RISK'
                    ORDER BY confidence DESC
                """,
            ),
            GenieViewSpec(
                name="fraud_network_communities",
                description="Network communities and clusters of entities that transact together or share behavioral patterns. Potential fraud rings show up as tight communities. Use this to investigate coordinated fraud.",
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
                description="Hub entities and bridge nodes in the fraud network. Hubs connect many entities and may be fraud coordinators. Bridges connect different communities and may be money mules.",
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
                name="shared_identity_indicators",
                description="Entities that share identity attributes like addresses, phone numbers, emails, IPs, or devices. Shared attributes between unrelated entities may indicate synthetic identity fraud.",
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
                description="Complete knowledge graph of all fraud detection triplets. Each row is a relationship (edge) between two entities (nodes). Use this for general exploration of the fraud detection graph.",
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


FRAUD_PACK = fraud_pack()
