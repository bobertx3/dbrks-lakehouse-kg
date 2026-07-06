"""Pipeline configuration for the lakehouse knowledge graph starter kit."""

from dataclasses import dataclass, field

from lakehouse_kg.domain import DomainPack
from lakehouse_kg.domains.generic import generic_pack


@dataclass
class PipelineConfig:
    """Top-level configuration for the agentic triplet pipeline."""

    catalog: str = "main"
    schema: str = "knowledge_graph"
    triplet_table: str = "gold_triplets"
    agent_log_table: str = "agent_execution_log"

    # Domain pack carrying all domain-specific knowledge (patterns,
    # predicates, prompts, Genie views). Defaults to the generic pack;
    # see lakehouse_kg.domains for the fraud worked example.
    domain: DomainPack = field(default_factory=generic_pack)

    # Source tables to analyze (populated at runtime via schema discovery)
    source_tables: list[str] = field(default_factory=list)

    # LLM configuration
    llm_endpoint: str = "databricks-meta-llama-3-3-70b-instruct"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.1

    # Statistical thresholds
    velocity_stddev_multiplier: float = 2.0
    high_amount_percentile: float = 95.0
    min_confidence: float = 0.3

    # ML clustering
    n_clusters: int = 8
    similarity_threshold: float = 0.85

    # Graph topology
    pagerank_threshold: float = 0.01
    community_resolution: float = 1.0

    # Execution control
    enable_llm_agent: bool = True
    enable_ml_agent: bool = True
    enable_graph_agent: bool = True
    max_triplets_per_agent: int = 5000
    batch_size: int = 1000

    @property
    def full_triplet_table(self) -> str:
        return f"`{self.catalog}`.`{self.schema}`.`{self.triplet_table}`"

    @property
    def full_log_table(self) -> str:
        return f"`{self.catalog}`.`{self.schema}`.`{self.agent_log_table}`"

    def table_ref(self, table_name: str) -> str:
        return f"`{self.catalog}`.`{self.schema}`.`{table_name}`"


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic fraud data generation (worked example)."""

    catalog: str = "main"
    schema: str = "knowledge_graph"
    n_customers: int = 5000
    n_merchants: int = 500
    n_accounts: int = 6000
    n_transactions: int = 100_000
    n_devices: int = 4000
    n_alerts: int = 2000
    fraud_rate: float = 0.03
    shared_address_rate: float = 0.08
    shared_device_rate: float = 0.05
    seed: int = 42

    def table_ref(self, table_name: str) -> str:
        return f"`{self.catalog}`.`{self.schema}`.`{table_name}`"
