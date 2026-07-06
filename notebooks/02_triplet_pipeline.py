# Databricks notebook source

# MAGIC %md
# MAGIC # 02 — Agentic Triplet Generation Pipeline
# MAGIC
# MAGIC This notebook runs the multi-agent pipeline that automatically discovers and generates
# MAGIC knowledge graph triplets from lakehouse datasets. Each agent asks a specific
# MAGIC question about the data, using a different analytical technique:
# MAGIC
# MAGIC | Phase | Agent | Question | Method |
# MAGIC |-------|-------|----------|--------|
# MAGIC | Discovery | Schema Discovery | "What data do we have?" | Schema introspection |
# MAGIC | Discovery | Entity Resolution | "What are the entities?" | Heuristic classification |
# MAGIC | Generation | Relationship Agent | "What direct relationships exist?" | FK/join analysis |
# MAGIC | Generation | Statistical Agent | "What anomalies exist?" | Distributional statistics |
# MAGIC | Generation | ML Clustering Agent | "What hidden clusters exist?" | KMeans + cosine similarity |
# MAGIC | Generation | LLM Semantic Agent | "What can a model infer?" | Foundation model reasoning |
# MAGIC | Enrichment | Graph Topology Agent | "What network patterns emerge?" | PageRank, Louvain, betweenness |
# MAGIC | Validation | Validation Agent | "Are triplets valid and scored?" | Dedup, consistency, corroboration |
# MAGIC
# MAGIC Domain knowledge (entity types, predicates, LLM prompts, Genie views) is pluggable
# MAGIC via the `domain` widget: `generic` for arbitrary data, `fraud` for the shipped
# MAGIC fraud-detection worked example (pairs with notebook 01).
# MAGIC
# MAGIC **Output:** `{catalog}.{schema}.gold_triplets` Delta table, optimized for Genie.

# COMMAND ----------

# MAGIC %pip install scikit-learn networkx openai -q
# MAGIC %pip install -e /Workspace/Users/<your-user>/lakehouse-kg-starter --no-deps -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "Unity Catalog")
dbutils.widgets.text("schema", "knowledge_graph", "Schema")
dbutils.widgets.dropdown("domain", "fraud", ["generic", "fraud"], "Domain Pack")
dbutils.widgets.text("source_tables", "customers,merchants,accounts,transactions,devices,alerts", "Source Tables (comma-sep, blank=all)")
dbutils.widgets.dropdown("enable_llm", "true", ["true", "false"], "Enable LLM Agent")
dbutils.widgets.dropdown("enable_ml", "true", ["true", "false"], "Enable ML Agent")
dbutils.widgets.dropdown("enable_graph", "true", ["true", "false"], "Enable Graph Agent")
dbutils.widgets.text("llm_endpoint", "databricks-meta-llama-3-3-70b-instruct", "LLM Endpoint")
dbutils.widgets.text("min_confidence", "0.3", "Min Confidence Threshold")

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
domain_name = dbutils.widgets.get("domain")
source_tables_raw = dbutils.widgets.get("source_tables").strip()
source_tables = [t.strip() for t in source_tables_raw.split(",") if t.strip()] if source_tables_raw else []
enable_llm = dbutils.widgets.get("enable_llm") == "true"
enable_ml = dbutils.widgets.get("enable_ml") == "true"
enable_graph = dbutils.widgets.get("enable_graph") == "true"
llm_endpoint = dbutils.widgets.get("llm_endpoint")
min_confidence = float(dbutils.widgets.get("min_confidence"))

print(f"Target: {catalog}.{schema}")
print(f"Domain pack: {domain_name}")
print(f"Source tables: {source_tables if source_tables else 'ALL (auto-discover)'}")
print(f"LLM Agent: {'enabled' if enable_llm else 'disabled'} ({llm_endpoint})")
print(f"ML Agent: {'enabled' if enable_ml else 'disabled'}")
print(f"Graph Agent: {'enabled' if enable_graph else 'disabled'}")
print(f"Min confidence: {min_confidence}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize Pipeline

# COMMAND ----------

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

from lakehouse_kg.config import PipelineConfig
from lakehouse_kg.domains import get_domain_pack
from lakehouse_kg.orchestrator import AgenticOrchestrator

domain_pack = get_domain_pack(domain_name)
print(f"Domain pack '{domain_pack.name}': {domain_pack.description}")

config = PipelineConfig(
    catalog=catalog,
    schema=schema,
    domain=domain_pack,
    source_tables=source_tables,
    enable_llm_agent=enable_llm,
    enable_ml_agent=enable_ml,
    enable_graph_agent=enable_graph,
    llm_endpoint=llm_endpoint,
    min_confidence=min_confidence,
)

orchestrator = AgenticOrchestrator(spark, config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execute Agentic Pipeline
# MAGIC
# MAGIC The orchestrator runs all agents in sequence. Each agent:
# MAGIC 1. Asks its primary question
# MAGIC 2. May ask follow-up sub-questions based on what it finds
# MAGIC 3. Generates triplets using its specialized method
# MAGIC 4. Passes enriched context to the next agent

# COMMAND ----------

summary = orchestrator.run()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline Results

# COMMAND ----------

print("=" * 70)
print("AGENTIC TRIPLET GENERATION — RESULTS SUMMARY")
print("=" * 70)
print(f"\nDomain pack: {summary['domain']}")
print(f"Total triplets generated: {summary['total_triplets']:,}")
print(f"Pipeline duration: {summary['total_time_seconds']:.1f}s")
print(f"Agents executed: {summary['agents_executed']} ({summary['agents_succeeded']} succeeded, {summary['agents_failed']} failed)")
print(f"Average confidence: {summary['avg_confidence']:.3f}")
print(f"\nOutput table: {summary['output_table']}")
print(f"Log table: {summary['log_table']}")

# COMMAND ----------

print("\nTriplets by Agent:")
print("-" * 40)
for agent, count in sorted(summary["triplets_by_agent"].items(), key=lambda x: -x[1]):
    print(f"  {agent:30s} {count:>6,}")

print("\nTriplets by Method:")
print("-" * 40)
for method, count in sorted(summary["triplets_by_method"].items(), key=lambda x: -x[1]):
    print(f"  {method:30s} {count:>6,}")

print("\nTriplets by Predicate:")
print("-" * 40)
for pred, count in sorted(summary["triplets_by_predicate"].items(), key=lambda x: -x[1]):
    print(f"  {pred:40s} {count:>6,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent Execution Details
# MAGIC
# MAGIC Each row shows the primary question asked by an agent, how many triplets it generated,
# MAGIC and how long it took.

# COMMAND ----------

import pandas as pd

agent_df = pd.DataFrame(summary["agent_details"])
display(spark.createDataFrame(agent_df))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample Triplets

# COMMAND ----------

triplet_df = spark.table(config.full_triplet_table)
print(f"\nGold triplets table: {config.full_triplet_table}")
print(f"Total rows: {triplet_df.count():,}")

# COMMAND ----------

display(triplet_df.limit(50))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Triplet Distribution Analysis

# COMMAND ----------

from pyspark.sql import functions as F

print("Subject type distribution:")
display(
    triplet_df.groupBy("subject_type")
    .agg(F.count("*").alias("count"), F.mean("confidence").alias("avg_confidence"))
    .orderBy("count", ascending=False)
)

# COMMAND ----------

print("Predicate distribution:")
display(
    triplet_df.groupBy("predicate")
    .agg(F.count("*").alias("count"), F.mean("confidence").alias("avg_confidence"))
    .orderBy("count", ascending=False)
)

# COMMAND ----------

print("Confidence distribution:")
display(
    triplet_df.withColumn(
        "confidence_bucket",
        F.when(F.col("confidence") >= 0.9, "0.9-1.0")
        .when(F.col("confidence") >= 0.7, "0.7-0.9")
        .when(F.col("confidence") >= 0.5, "0.5-0.7")
        .otherwise("0.3-0.5")
    )
    .groupBy("confidence_bucket")
    .count()
    .orderBy("confidence_bucket")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execution Log

# COMMAND ----------

display(spark.table(config.full_log_table))

# COMMAND ----------

print("Agentic Triplet Generation complete!")
print(f"\nNext step: Run notebook 03 to create Genie-optimized views.")
