"""Genie Space view creation utilities.

Creates optimized SQL views over the gold_triplets Delta table that are
designed for natural-language querying in Databricks Genie. The view
specifications (names, descriptions, SQL) come from the domain pack on the
pipeline config, so each domain ships views phrased in its own vocabulary.
"""

from __future__ import annotations

import logging

from pyspark.sql import SparkSession

from lakehouse_kg.config import PipelineConfig

logger = logging.getLogger("lakehouse_kg.genie")


def create_genie_views(spark: SparkSession, config: PipelineConfig) -> list[str]:
    """Create all Genie-optimized views for the configured domain and return their full names."""
    triplet_table = config.full_triplet_table
    log_table = config.full_log_table
    schema_ref = f"{config.catalog}.{config.schema}"

    created_views = []
    for view in config.domain.genie_views:
        full_name = f"{schema_ref}.{view.name}"
        view_sql = view.sql_template.format(
            triplet_table=triplet_table,
            log_table=log_table,
        )
        try:
            spark.sql(f"DROP VIEW IF EXISTS {full_name}")
            spark.sql(f"""
                CREATE VIEW {full_name}
                COMMENT '{view.description}'
                AS {view_sql}
            """)
            created_views.append(full_name)
            logger.info(f"Created Genie view: {full_name}")
        except Exception as e:
            logger.error(f"Failed to create view {full_name}: {e}")

    return created_views
