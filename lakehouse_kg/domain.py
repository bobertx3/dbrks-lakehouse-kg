"""Domain pack definitions for the lakehouse knowledge graph pipeline.

A DomainPack carries all domain-specific knowledge used by the agents:
entity-type classification patterns, predicate vocabularies, LLM prompt
templates, statistical pattern names, and Genie view specifications.
Agents receive the pack via PipelineConfig and contain no domain-specific
strings themselves. Ship a new domain by building a new DomainPack (see
lakehouse_kg.domains.fraud for a complete worked example).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenieViewSpec:
    """Specification for a Genie-optimized SQL view over the triplet table.

    The sql_template may reference {triplet_table} and {log_table}, which are
    substituted with fully qualified table names at creation time. The
    description is embedded as the view COMMENT, so avoid single quotes.
    """

    name: str
    description: str
    sql_template: str


@dataclass
class DomainPack:
    """All domain-specific knowledge consumed by the pipeline agents.

    Field groups map to the agents that consume them; every default is
    domain-neutral so a bare DomainPack(name=...) is usable, but prefer the
    prebuilt packs in lakehouse_kg.domains.
    """

    name: str
    description: str = ""

    # --- Entity Resolution Agent ---
    # Heuristic substring patterns mapping table/column names to entity types.
    entity_type_patterns: dict[str, list[str]] = field(default_factory=dict)
    # Fallback entity type when no pattern matches.
    default_entity_type: str = "Entity"
    # Keywords marking identity-like columns whose duplicate values link
    # entities (shared addresses, phones, devices, ...). Used by both the
    # entity agent (detection) and the relationship agent (triplet emission).
    identity_attribute_keywords: list[str] = field(default_factory=list)

    # --- Relationship Agent ---
    # (source_table, referenced_table) -> predicate name. Unmapped FK pairs
    # fall back to RELATED_TO_<TABLE>.
    fk_predicates: dict[tuple[str, str], str] = field(default_factory=dict)

    # --- Statistical Agent ---
    statistical_question: str = (
        "What behavioral patterns and statistical anomalies exist in the data?"
    )
    velocity_question: str = (
        "Which actors show unusual event velocity (burst patterns)?"
    )
    amount_question: str = (
        "Which events have values that are statistical outliers?"
    )
    temporal_question: str = (
        "Are there unusual temporal patterns (late-night, weekend bursts)?"
    )
    concentration_question: str = (
        "Do any actors show unusual concentration on specific counterparties?"
    )
    # Substring keywords for locating the event/fact table to analyze.
    event_table_keywords: list[str] = field(
        default_factory=lambda: ["event", "transaction", "txn", "activity", "log"]
    )
    # Candidate column names, tried in order (case-insensitive).
    actor_id_columns: list[str] = field(
        default_factory=lambda: ["entity_id", "actor_id", "account_id", "user_id", "customer_id"]
    )
    event_id_columns: list[str] = field(
        default_factory=lambda: ["event_id", "transaction_id", "activity_id", "record_id"]
    )
    counterparty_id_columns: list[str] = field(
        default_factory=lambda: ["counterparty_id", "target_id", "merchant_id", "destination_id"]
    )
    timestamp_columns: list[str] = field(
        default_factory=lambda: ["timestamp", "event_time", "event_date", "created_at"]
    )
    amount_columns: list[str] = field(
        default_factory=lambda: ["amount", "value", "quantity", "total"]
    )
    # Entity types assigned to statistical triplets.
    actor_entity_type: str = "Entity"
    event_entity_type: str = "Event"
    counterparty_entity_type: str = "Entity"
    pattern_entity_type: str = "Pattern"
    # Statistical pattern predicates.
    velocity_predicate: str = "EXHIBITS_VELOCITY_ANOMALY"
    outlier_amount_predicate: str = "HAS_OUTLIER_VALUE"
    high_value_event_predicate: str = "HAS_HIGH_VALUE_EVENT"
    off_hours_predicate: str = "EXHIBITS_OFF_HOURS_PATTERN"
    concentration_predicate: str = "CONCENTRATED_AT_COUNTERPARTY"

    # --- ML Clustering Agent ---
    ml_question: str = (
        "Are there hidden communities or behavioral clusters among entities?"
    )
    ml_clustering_question: str = (
        "Can actors be grouped by behavioral fingerprint (value patterns, timing, counterparty mix)?"
    )
    cluster_predicate: str = "BELONGS_TO_CLUSTER"
    similarity_predicate: str = "BEHAVIORALLY_SIMILAR_TO"
    cluster_entity_type: str = "BehavioralCluster"

    # --- LLM Semantic Agent ---
    llm_question: str = (
        "What semantic relationships can an LLM infer from the data?"
    )
    llm_entity_question: str = (
        "Do entity samples reveal notable patterns or relationships?"
    )
    # System prompt establishing the LLM persona for this domain.
    llm_system_prompt: str = (
        "You are a data analysis expert. Respond only with valid JSON."
    )
    # Prompt template with {schema_summary} and {existing_predicates} placeholders.
    llm_schema_prompt: str = ""
    # Prompt template with {entity_samples} placeholder.
    llm_entity_prompt: str = ""
    # Predicate used when the LLM omits one from an entity-level finding.
    default_llm_predicate: str = "DETECTED_PATTERN"

    # --- Graph Topology Agent ---
    hub_question: str = (
        "Which entities are hubs connecting many others?"
    )
    community_question: str = (
        "Are there tightly connected communities in the network?"
    )
    bridge_question: str = (
        "Which entities bridge different communities?"
    )
    risk_question: str = (
        "How does risk propagate through the network from flagged entities?"
    )
    hub_predicate: str = "IS_NETWORK_HUB"
    community_predicate: str = "MEMBER_OF_COMMUNITY"
    bridge_predicate: str = "IS_NETWORK_BRIDGE"
    risk_predicate: str = "HAS_PROXIMITY_RISK"
    # Predicates whose subjects seed the risk-propagation walk.
    risk_seed_predicates: list[str] = field(default_factory=list)

    # --- Validation Agent ---
    # Pairs of predicates that cannot both hold for the same subject; the
    # lower-confidence triplet of a conflicting pair is dropped.
    contradictory_predicate_pairs: list[tuple[str, str]] = field(default_factory=list)

    # --- Genie views ---
    genie_views: list[GenieViewSpec] = field(default_factory=list)
