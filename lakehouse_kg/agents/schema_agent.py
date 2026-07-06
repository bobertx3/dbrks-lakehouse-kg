"""Schema Discovery Agent - Analyzes table structures to understand the data landscape."""

from __future__ import annotations

from typing import Any

from pyspark.sql import functions as F

from lakehouse_kg.agents.base import BaseAgent, AgentResult


class SchemaDiscoveryAgent(BaseAgent):
    """Discovers and catalogs all table schemas, column types, cardinality,
    null rates, and potential key columns. This agent answers the foundational
    question before any triplet generation can begin."""

    @property
    def name(self) -> str:
        return "schema_discovery"

    @property
    def question(self) -> str:
        return "What datasets are available and what do they contain?"

    def execute(self, context: dict[str, Any]) -> AgentResult:
        result = AgentResult(agent_name=self.name)
        schemas: dict[str, dict] = {}

        tables = self._discover_tables()
        result.questions_asked.append(
            f"Found {len(tables)} tables in {self.config.catalog}.{self.config.schema}"
        )

        for table_name in tables:
            try:
                table_info = self._analyze_table(table_name)
                schemas[table_name] = table_info
                result.answers.append(
                    f"Table '{table_name}': {table_info['row_count']} rows, "
                    f"{len(table_info['columns'])} columns, "
                    f"potential keys: {table_info['potential_keys']}"
                )
            except Exception as e:
                self.logger.warning(f"Failed to analyze table '{table_name}': {e}")

        result.metadata["schemas"] = schemas
        result.metadata["table_names"] = list(schemas.keys())
        return result

    def _discover_tables(self) -> list[str]:
        if self.config.source_tables:
            return self.config.source_tables

        catalog, schema = self.config.catalog, self.config.schema
        tables_df = self.spark.sql(f"SHOW TABLES IN `{catalog}`.`{schema}`")
        return tables_df.toPandas()["tableName"].tolist()

    def _analyze_table(self, table_name: str) -> dict[str, Any]:
        df = self._read_table(table_name)
        fields = df.schema.fields

        # Single pass: count + all per-column distinct/null stats in one Spark action
        stat_exprs = [F.count("*").alias("__row_count__")]
        for i, col_info in enumerate(fields):
            qc = F.col(f"`{col_info.name}`")
            stat_exprs.append(F.countDistinct(qc).alias(f"__d{i}__"))
            stat_exprs.append(F.count(F.when(qc.isNull(), 1)).alias(f"__n{i}__"))

        all_stats = df.select(*stat_exprs).first()
        row_count = all_stats["__row_count__"]

        columns = {}
        potential_keys = []
        potential_fks = []

        for i, col_info in enumerate(fields):
            col_name = col_info.name
            distinct_count = all_stats[f"__d{i}__"]
            null_count = all_stats[f"__n{i}__"]
            uniqueness_ratio = distinct_count / row_count if row_count > 0 else 0

            columns[col_name] = {
                "type": str(col_info.dataType),
                "distinct_count": distinct_count,
                "null_count": null_count,
                "null_rate": null_count / row_count if row_count > 0 else 0,
                "uniqueness_ratio": uniqueness_ratio,
            }

            if uniqueness_ratio > 0.99 and null_count == 0:
                potential_keys.append(col_name)

            if (
                col_name.endswith("_id")
                and col_name != f"{table_name.rstrip('s')}_id"
                and uniqueness_ratio < 0.9
            ):
                potential_fks.append(col_name)

        return {
            "row_count": row_count,
            "columns": columns,
            "potential_keys": potential_keys,
            "potential_fks": potential_fks,
            "column_names": list(columns.keys()),
        }
