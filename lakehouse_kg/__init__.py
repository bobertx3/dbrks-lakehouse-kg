"""Lakehouse Knowledge Graph Starter Kit — agentic triplet generation on Databricks."""

__version__ = "0.1.0"

from lakehouse_kg.config import PipelineConfig
from lakehouse_kg.domain import DomainPack


def __getattr__(name):
    if name == "AgenticOrchestrator":
        from lakehouse_kg.orchestrator import AgenticOrchestrator
        return AgenticOrchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["PipelineConfig", "DomainPack", "AgenticOrchestrator"]
