"""Base agent class and shared types for the agentic triplet pipeline."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from pyspark.sql import SparkSession, DataFrame

from lakehouse_kg.config import PipelineConfig

logger = logging.getLogger("lakehouse_kg")


@dataclass
class Triplet:
    """A single knowledge graph triplet with metadata."""

    subject_id: str
    subject_type: str
    predicate: str
    object_id: str
    object_type: str
    confidence: float
    source_agent: str
    source_method: str
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        self.confidence = max(0.0, min(1.0, self.confidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": str(self.subject_id),
            "subject_type": self.subject_type,
            "predicate": self.predicate,
            "object_id": str(self.object_id),
            "object_type": self.object_type,
            "confidence": round(self.confidence, 4),
            "source_agent": self.source_agent,
            "source_method": self.source_method,
            "properties": json.dumps(self.properties),
            "created_at": self.created_at,
        }


@dataclass
class AgentResult:
    """Result returned by an agent after execution."""

    agent_name: str
    triplets: list[Triplet] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    questions_asked: list[str] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def triplet_count(self) -> int:
        return len(self.triplets)

    def to_dataframe(self):
        import pandas as pd

        if not self.triplets:
            return pd.DataFrame(columns=[
                "subject_id", "subject_type", "predicate", "object_id",
                "object_type", "confidence", "source_agent", "source_method",
                "properties", "created_at",
            ])
        return pd.DataFrame([t.to_dict() for t in self.triplets])


class BaseAgent(ABC):
    """Base class for all agents in the triplet generation pipeline.

    Each agent represents a specialized reasoning step that asks a specific
    question about the data and generates triplets based on its analysis.
    Domain-specific knowledge (entity patterns, predicates, prompts) comes
    from the DomainPack on the pipeline config, exposed as self.domain.
    """

    def __init__(self, spark, config: PipelineConfig):
        self.spark = spark
        self.config = config
        self.domain = config.domain
        self.logger = logging.getLogger(f"lakehouse_kg.{self.name}")

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this agent."""

    @property
    @abstractmethod
    def question(self) -> str:
        """The key question this agent answers about the data."""

    @abstractmethod
    def execute(self, context: dict[str, Any]) -> AgentResult:
        """Run the agent's analysis and return triplets.

        Args:
            context: Shared context dict accumulated by prior agents.
                     Keys include 'schemas', 'entities', 'prior_triplets', etc.

        Returns:
            AgentResult with generated triplets and metadata.
        """

    def run(self, context: dict[str, Any]) -> AgentResult:
        """Execute the agent with timing and error handling."""
        self.logger.info(f"Agent '{self.name}' asking: {self.question}")
        start = time.time()
        try:
            result = self.execute(context)
            result.duration_seconds = time.time() - start
            result.questions_asked.insert(0, self.question)
            self.logger.info(
                f"Agent '{self.name}' generated {result.triplet_count} triplets "
                f"in {result.duration_seconds:.1f}s"
            )
            return result
        except Exception as e:
            self.logger.error(f"Agent '{self.name}' failed: {e}", exc_info=True)
            return AgentResult(
                agent_name=self.name,
                duration_seconds=time.time() - start,
                error=str(e),
                questions_asked=[self.question],
            )

    def _read_table(self, table_name: str) -> DataFrame:
        """Read a table from the configured catalog.schema."""
        full_name = self.config.table_ref(table_name)
        return self.spark.table(full_name)

    def _make_triplet(
        self,
        subject_id: str,
        subject_type: str,
        predicate: str,
        object_id: str,
        object_type: str,
        confidence: float,
        method: str,
        properties: Optional[dict] = None,
    ) -> Triplet:
        return Triplet(
            subject_id=subject_id,
            subject_type=subject_type,
            predicate=predicate,
            object_id=object_id,
            object_type=object_type,
            confidence=confidence,
            source_agent=self.name,
            source_method=method,
            properties=properties or {},
        )
