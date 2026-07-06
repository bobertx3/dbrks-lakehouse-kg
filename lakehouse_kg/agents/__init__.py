"""Agent modules for agentic triplet generation."""

__all__ = [
    "BaseAgent",
    "AgentResult",
    "SchemaDiscoveryAgent",
    "EntityResolutionAgent",
    "RelationshipAgent",
    "StatisticalAgent",
    "MLClusteringAgent",
    "LLMSemanticAgent",
    "GraphTopologyAgent",
    "ValidationAgent",
]


def __getattr__(name):
    if name in ("BaseAgent", "AgentResult", "Triplet"):
        from lakehouse_kg.agents.base import BaseAgent, AgentResult, Triplet
        return {"BaseAgent": BaseAgent, "AgentResult": AgentResult, "Triplet": Triplet}[name]
    if name == "SchemaDiscoveryAgent":
        from lakehouse_kg.agents.schema_agent import SchemaDiscoveryAgent
        return SchemaDiscoveryAgent
    if name == "EntityResolutionAgent":
        from lakehouse_kg.agents.entity_agent import EntityResolutionAgent
        return EntityResolutionAgent
    if name == "RelationshipAgent":
        from lakehouse_kg.agents.relationship_agent import RelationshipAgent
        return RelationshipAgent
    if name == "StatisticalAgent":
        from lakehouse_kg.agents.statistical_agent import StatisticalAgent
        return StatisticalAgent
    if name == "MLClusteringAgent":
        from lakehouse_kg.agents.ml_agent import MLClusteringAgent
        return MLClusteringAgent
    if name == "LLMSemanticAgent":
        from lakehouse_kg.agents.llm_agent import LLMSemanticAgent
        return LLMSemanticAgent
    if name == "GraphTopologyAgent":
        from lakehouse_kg.agents.graph_agent import GraphTopologyAgent
        return GraphTopologyAgent
    if name == "ValidationAgent":
        from lakehouse_kg.agents.validation_agent import ValidationAgent
        return ValidationAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
