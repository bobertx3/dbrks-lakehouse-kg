"""Multi-agent orchestrator for agentic triplet generation.

The orchestrator coordinates a sequence of specialized agents, each asking
a different question about the data. Context flows forward: each agent
enriches the shared context dict, which downstream agents consume.

Execution order:
  1. Schema Discovery    → "What data do we have?"
  2. Entity Resolution   → "What are the entities?"
  3. Relationship Agent  → "What direct relationships exist?"
  4. Statistical Agent   → "What statistical anomalies exist?"
  5. ML Clustering Agent → "What hidden clusters exist?"
  6. LLM Semantic Agent  → "What can a model infer semantically?"
  7. Graph Topology Agent→ "What network patterns emerge?"
  8. Validation Agent    → "Are these triplets valid and scored?"
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    TimestampType,
)

from lakehouse_kg.config import PipelineConfig
from lakehouse_kg.agents.base import AgentResult, Triplet
from lakehouse_kg.agents.schema_agent import SchemaDiscoveryAgent
from lakehouse_kg.agents.entity_agent import EntityResolutionAgent
from lakehouse_kg.agents.relationship_agent import RelationshipAgent
from lakehouse_kg.agents.statistical_agent import StatisticalAgent
from lakehouse_kg.agents.ml_agent import MLClusteringAgent
from lakehouse_kg.agents.llm_agent import LLMSemanticAgent
from lakehouse_kg.agents.graph_agent import GraphTopologyAgent
from lakehouse_kg.agents.validation_agent import ValidationAgent

logger = logging.getLogger("lakehouse_kg.orchestrator")

# The 10-column triplet contract: confidence is DOUBLE, created_at is TIMESTAMP.
TRIPLET_SCHEMA = StructType([
    StructField("subject_id", StringType(), False),
    StructField("subject_type", StringType(), False),
    StructField("predicate", StringType(), False),
    StructField("object_id", StringType(), False),
    StructField("object_type", StringType(), False),
    StructField("confidence", DoubleType(), False),
    StructField("source_agent", StringType(), False),
    StructField("source_method", StringType(), False),
    StructField("properties", StringType(), True),
    StructField("created_at", TimestampType(), False),
])


class AgenticOrchestrator:
    """Coordinates the multi-agent triplet generation pipeline.

    Usage:
        config = PipelineConfig(catalog="main", schema="knowledge_graph")
        orchestrator = AgenticOrchestrator(spark, config)
        summary = orchestrator.run()
    """

    def __init__(self, spark: SparkSession, config: PipelineConfig):
        self.spark = spark
        self.config = config
        self.context: dict[str, Any] = {}
        self.results: list[AgentResult] = []
        self.all_triplets: list[Triplet] = []
        self.execution_log: list[dict] = []

    def run(self) -> dict[str, Any]:
        """Execute the full agentic pipeline and write results to Delta."""
        pipeline_start = time.time()
        logger.info("=" * 70)
        logger.info("AGENTIC TRIPLET GENERATION PIPELINE")
        logger.info(f"Domain pack: {self.config.domain.name}")
        logger.info("=" * 70)

        # Phase 1: Discovery agents (no triplets, just context)
        self._run_discovery_phase()

        # Phase 2: Generation agents (produce triplets)
        self._run_generation_phase()

        # Phase 3: Enrichment (graph topology on generated triplets)
        self._run_enrichment_phase()

        # Phase 4: Validation (final quality pass)
        validated = self._run_validation_phase()

        # Phase 5: Persist to Delta Lake
        self._write_to_delta(validated)
        self._write_execution_log()

        total_time = time.time() - pipeline_start
        summary = self._build_summary(validated, total_time)

        logger.info("=" * 70)
        logger.info(f"Pipeline complete: {len(validated)} triplets in {total_time:.1f}s")
        logger.info("=" * 70)

        return summary

    def _run_discovery_phase(self):
        """Run schema discovery and entity resolution to build context."""
        logger.info("-" * 40)
        logger.info("PHASE 1: DISCOVERY")
        logger.info("-" * 40)

        # Agent 1: Schema Discovery
        schema_agent = SchemaDiscoveryAgent(self.spark, self.config)
        schema_result = schema_agent.run(self.context)
        self.results.append(schema_result)
        self._log_agent(schema_result)

        if not schema_result.success:
            raise RuntimeError(f"Schema discovery failed: {schema_result.error}")

        self.context.update(schema_result.metadata)

        # Agent 2: Entity Resolution
        entity_agent = EntityResolutionAgent(self.spark, self.config)
        entity_result = entity_agent.run(self.context)
        self.results.append(entity_result)
        self._log_agent(entity_result)

        self.context.update(entity_result.metadata)

    def _run_generation_phase(self):
        """Run relationship, statistical, ML, and LLM agents to generate triplets."""
        logger.info("-" * 40)
        logger.info("PHASE 2: GENERATION")
        logger.info("-" * 40)

        # Agent 3: Rule-based Relationships
        rel_agent = RelationshipAgent(self.spark, self.config)
        rel_result = rel_agent.run(self.context)
        self.results.append(rel_result)
        self._log_agent(rel_result)
        self.all_triplets.extend(rel_result.triplets)

        # Agent 4: Statistical Analysis
        stat_agent = StatisticalAgent(self.spark, self.config)
        stat_result = stat_agent.run(self.context)
        self.results.append(stat_result)
        self._log_agent(stat_result)
        self.all_triplets.extend(stat_result.triplets)

        # Agent 5: ML Clustering (optional)
        if self.config.enable_ml_agent:
            ml_agent = MLClusteringAgent(self.spark, self.config)
            ml_result = ml_agent.run(self.context)
            self.results.append(ml_result)
            self._log_agent(ml_result)
            self.all_triplets.extend(ml_result.triplets)

        # Agent 6: LLM Semantic (optional)
        if self.config.enable_llm_agent:
            self.context["prior_triplets"] = self.all_triplets
            llm_agent = LLMSemanticAgent(self.spark, self.config)
            llm_result = llm_agent.run(self.context)
            self.results.append(llm_result)
            self._log_agent(llm_result)
            self.all_triplets.extend(llm_result.triplets)

    def _run_enrichment_phase(self):
        """Run graph topology analysis on the accumulated triplets."""
        logger.info("-" * 40)
        logger.info("PHASE 3: ENRICHMENT")
        logger.info("-" * 40)

        if self.config.enable_graph_agent:
            self.context["prior_triplets"] = self.all_triplets
            graph_agent = GraphTopologyAgent(self.spark, self.config)
            graph_result = graph_agent.run(self.context)
            self.results.append(graph_result)
            self._log_agent(graph_result)
            self.all_triplets.extend(graph_result.triplets)

    def _run_validation_phase(self) -> list[Triplet]:
        """Run the validation agent for final quality control."""
        logger.info("-" * 40)
        logger.info("PHASE 4: VALIDATION")
        logger.info("-" * 40)

        self.context["prior_triplets"] = self.all_triplets
        val_agent = ValidationAgent(self.spark, self.config)
        val_result = val_agent.run(self.context)
        self.results.append(val_result)
        self._log_agent(val_result)
        return val_result.triplets

    def _write_to_delta(self, triplets: list[Triplet]):
        """Write validated triplets to the Delta Lake gold table."""
        if not triplets:
            logger.warning("No triplets to write.")
            return

        rows = [t.to_dict() for t in triplets]
        pdf = pd.DataFrame(rows)
        sdf = self.spark.createDataFrame(pdf, schema=TRIPLET_SCHEMA)

        full_table = self.config.full_triplet_table
        logger.info(f"Writing {len(triplets)} triplets to {full_table}")

        sdf.write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).saveAsTable(full_table)

        try:
            self.spark.sql(f"OPTIMIZE {full_table} ZORDER BY (subject_type, predicate, object_type)")
            logger.info(f"Table {full_table} optimized with ZORDER")
        except Exception as e:
            logger.warning(f"OPTIMIZE ZORDER skipped (may not be supported on this compute): {e}")

    def _write_execution_log(self):
        """Persist the agent execution log for observability."""
        if not self.execution_log:
            return

        log_pdf = pd.DataFrame(self.execution_log)
        log_sdf = self.spark.createDataFrame(log_pdf)

        full_table = self.config.full_log_table
        log_sdf.write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).saveAsTable(full_table)
        logger.info(f"Execution log written to {full_table}")

    def _log_agent(self, result: AgentResult):
        """Record agent execution for the log table."""
        self.execution_log.append({
            "agent_name": result.agent_name,
            "success": result.success,
            "triplet_count": result.triplet_count,
            "duration_seconds": round(result.duration_seconds, 2),
            "error": result.error or "",
            "questions_asked": json.dumps(result.questions_asked),
            "answers": json.dumps(result.answers),
            "executed_at": datetime.utcnow().isoformat(),
        })

    def _build_summary(self, validated: list[Triplet], total_time: float) -> dict:
        """Build a summary dict for display."""
        from collections import Counter

        return {
            "domain": self.config.domain.name,
            "total_triplets": len(validated),
            "total_time_seconds": round(total_time, 1),
            "agents_executed": len(self.results),
            "agents_succeeded": sum(1 for r in self.results if r.success),
            "agents_failed": sum(1 for r in self.results if not r.success),
            "triplets_by_agent": dict(Counter(t.source_agent for t in validated)),
            "triplets_by_method": dict(Counter(t.source_method for t in validated)),
            "triplets_by_predicate": dict(Counter(t.predicate for t in validated)),
            "avg_confidence": (
                round(sum(t.confidence for t in validated) / len(validated), 3)
                if validated
                else 0
            ),
            "output_table": self.config.full_triplet_table,
            "log_table": self.config.full_log_table,
            "agent_details": [
                {
                    "name": r.agent_name,
                    "question": r.questions_asked[0] if r.questions_asked else "",
                    "triplets": r.triplet_count,
                    "duration": round(r.duration_seconds, 1),
                    "success": r.success,
                }
                for r in self.results
            ],
        }
