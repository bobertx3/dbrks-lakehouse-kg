"""Statistical Agent - Discovers behavioral patterns through statistical analysis."""

from __future__ import annotations

from typing import Any

from pyspark.sql import functions as F

from lakehouse_kg.agents.base import BaseAgent, AgentResult, Triplet


class StatisticalAgent(BaseAgent):
    """Analyzes event data for statistical anomalies. Uses velocity analysis,
    value distributions, temporal patterns, and co-occurrence statistics —
    no ML models, pure distributional reasoning. Table/column heuristics,
    entity types, and pattern predicates come from the domain pack."""

    @property
    def name(self) -> str:
        return "statistical_analysis"

    @property
    def question(self) -> str:
        return self.domain.statistical_question

    def execute(self, context: dict[str, Any]) -> AgentResult:
        result = AgentResult(agent_name=self.name)
        schemas = context.get("schemas", {})

        event_table = self._find_event_table(schemas)
        if not event_table:
            result.answers.append("No event table found — skipping statistical analysis.")
            return result

        event_df = self._read_table(event_table)

        # Sub-question 1: Velocity anomalies
        result.questions_asked.append(self.domain.velocity_question)
        velocity_triplets = self._velocity_analysis(event_df)
        result.triplets.extend(velocity_triplets)
        result.answers.append(f"Found {len(velocity_triplets)} velocity anomaly triplets")

        # Sub-question 2: Amount anomalies
        result.questions_asked.append(self.domain.amount_question)
        amount_triplets = self._amount_analysis(event_df)
        result.triplets.extend(amount_triplets)
        result.answers.append(f"Found {len(amount_triplets)} amount anomaly triplets")

        # Sub-question 3: Temporal patterns
        result.questions_asked.append(self.domain.temporal_question)
        temporal_triplets = self._temporal_analysis(event_df)
        result.triplets.extend(temporal_triplets)
        result.answers.append(f"Found {len(temporal_triplets)} temporal pattern triplets")

        # Sub-question 4: Counterparty concentration
        result.questions_asked.append(self.domain.concentration_question)
        concentration_triplets = self._counterparty_concentration(event_df)
        result.triplets.extend(concentration_triplets)
        result.answers.append(f"Found {len(concentration_triplets)} concentration triplets")

        return result

    def _find_event_table(self, schemas: dict) -> str | None:
        for name in schemas:
            if any(k in name.lower() for k in self.domain.event_table_keywords):
                return name
        return None

    def _velocity_analysis(self, event_df) -> list[Triplet]:
        """Detect actors with burst event patterns."""
        triplets = []
        actor_col = self._find_column(event_df, self.domain.actor_id_columns)
        ts_col = self._find_column(event_df, self.domain.timestamp_columns)
        if not actor_col or not ts_col:
            return triplets

        hourly_counts = (
            event_df.withColumn("event_hour", F.date_trunc("hour", F.col(ts_col)))
            .groupBy(actor_col, "event_hour")
            .agg(F.count("*").alias("event_count"))
        )

        stats = hourly_counts.groupBy(actor_col).agg(
            F.mean("event_count").alias("avg_count"),
            F.stddev("event_count").alias("std_count"),
            F.max("event_count").alias("max_count"),
        ).filter(F.col("std_count") > 0)

        threshold = self.config.velocity_stddev_multiplier
        anomalies = (
            stats.filter(
                F.col("max_count") > F.col("avg_count") + threshold * F.col("std_count")
            )
            .withColumn(
                "confidence",
                F.lit(0.5) + F.lit(0.5) * F.least(
                    F.lit(1.0),
                    (F.col("max_count") - F.col("avg_count")) / (F.col("std_count") * 5),
                ),
            )
            .limit(self.config.batch_size)
        )

        pdf = anomalies.toPandas()
        for _, row in pdf.iterrows():
            triplets.append(
                self._make_triplet(
                    subject_id=str(row[actor_col]),
                    subject_type=self.domain.actor_entity_type,
                    predicate=self.domain.velocity_predicate,
                    object_id=f"velocity_burst_{row[actor_col]}",
                    object_type=self.domain.pattern_entity_type,
                    confidence=float(row["confidence"]),
                    method="velocity_analysis",
                    properties={
                        "avg_hourly_events": round(float(row["avg_count"]), 2),
                        "max_hourly_events": int(row["max_count"]),
                        "std_dev": round(float(row["std_count"]), 2),
                    },
                )
            )
        return triplets

    def _amount_analysis(self, event_df) -> list[Triplet]:
        """Detect events with outlier amounts."""
        triplets = []
        amt_col = self._find_column(event_df, self.domain.amount_columns)
        event_id_col = self._find_column(event_df, self.domain.event_id_columns)
        actor_col = self._find_column(event_df, self.domain.actor_id_columns)
        if not amt_col or not event_id_col:
            return triplets

        percentile = self.config.high_amount_percentile / 100.0
        threshold = event_df.approxQuantile(amt_col, [percentile], 0.01)[0]

        has_actor = actor_col and actor_col in event_df.columns

        outliers = (
            event_df.filter(F.col(amt_col) > threshold)
            .withColumn(
                "ratio",
                F.when(F.lit(threshold) > 0, F.col(amt_col) / F.lit(threshold))
                .otherwise(F.lit(1.0)),
            )
            .withColumn(
                "confidence",
                F.least(F.lit(1.0), F.lit(0.6) + F.lit(0.1) * F.col("ratio")),
            )
            .limit(self.config.batch_size)
        )

        pdf = outliers.toPandas()
        for _, row in pdf.iterrows():
            triplets.append(
                self._make_triplet(
                    subject_id=str(row[event_id_col]),
                    subject_type=self.domain.event_entity_type,
                    predicate=self.domain.outlier_amount_predicate,
                    object_id=f"high_amount_p{int(self.config.high_amount_percentile)}",
                    object_type=self.domain.pattern_entity_type,
                    confidence=float(row["confidence"]),
                    method="amount_analysis",
                    properties={
                        "amount": float(row[amt_col]),
                        "threshold": round(threshold, 2),
                        "ratio_to_threshold": round(float(row["ratio"]), 2),
                    },
                )
            )

            if has_actor:
                triplets.append(
                    self._make_triplet(
                        subject_id=str(row[actor_col]),
                        subject_type=self.domain.actor_entity_type,
                        predicate=self.domain.high_value_event_predicate,
                        object_id=str(row[event_id_col]),
                        object_type=self.domain.event_entity_type,
                        confidence=float(row["confidence"]),
                        method="amount_analysis",
                        properties={"amount": float(row[amt_col])},
                    )
                )
        return triplets

    def _temporal_analysis(self, event_df) -> list[Triplet]:
        """Detect actors with unusual temporal event patterns."""
        triplets = []
        actor_col = self._find_column(event_df, self.domain.actor_id_columns)
        ts_col = self._find_column(event_df, self.domain.timestamp_columns)
        if not actor_col or not ts_col:
            return triplets

        enriched = event_df.withColumn("hour", F.hour(F.col(ts_col)))

        off_hours = enriched.filter(
            (F.col("hour") >= 0) & (F.col("hour") < 6)
        )

        off_hour_pct = (
            off_hours.groupBy(actor_col)
            .agg(F.count("*").alias("off_hour_count"))
            .join(
                enriched.groupBy(actor_col).agg(F.count("*").alias("total_count")),
                actor_col,
            )
            .withColumn("off_hour_ratio", F.col("off_hour_count") / F.col("total_count"))
            .filter(F.col("off_hour_ratio") > 0.3)
            .filter(F.col("off_hour_count") >= 3)
            .withColumn(
                "confidence",
                F.lit(0.5) + F.lit(0.5) * F.col("off_hour_ratio"),
            )
            .limit(self.config.batch_size)
        )

        pdf = off_hour_pct.toPandas()
        for _, row in pdf.iterrows():
            triplets.append(
                self._make_triplet(
                    subject_id=str(row[actor_col]),
                    subject_type=self.domain.actor_entity_type,
                    predicate=self.domain.off_hours_predicate,
                    object_id=f"off_hours_{row[actor_col]}",
                    object_type=self.domain.pattern_entity_type,
                    confidence=float(row["confidence"]),
                    method="temporal_analysis",
                    properties={
                        "off_hour_count": int(row["off_hour_count"]),
                        "total_count": int(row["total_count"]),
                        "off_hour_ratio": round(float(row["off_hour_ratio"]), 3),
                    },
                )
            )
        return triplets

    def _counterparty_concentration(self, event_df) -> list[Triplet]:
        """Detect actors that concentrate events on few counterparties."""
        triplets = []
        actor_col = self._find_column(event_df, self.domain.actor_id_columns)
        cp_col = self._find_column(event_df, self.domain.counterparty_id_columns)
        if not actor_col or not cp_col:
            return triplets

        actor_cp = event_df.groupBy(actor_col, cp_col).agg(
            F.count("*").alias("event_count"),
            F.sum("amount").alias("total_amount") if "amount" in event_df.columns else F.lit(0).alias("total_amount"),
        )

        actor_total = event_df.groupBy(actor_col).agg(F.count("*").alias("actor_total"))

        concentration = (
            actor_cp.join(actor_total, actor_col)
            .withColumn("concentration", F.col("event_count") / F.col("actor_total"))
            .filter(F.col("concentration") > 0.5)
            .filter(F.col("event_count") >= 5)
            .withColumn(
                "confidence",
                F.lit(0.5) + F.lit(0.5) * F.col("concentration"),
            )
            .limit(self.config.batch_size)
        )

        pdf = concentration.toPandas()
        for _, row in pdf.iterrows():
            triplets.append(
                self._make_triplet(
                    subject_id=str(row[actor_col]),
                    subject_type=self.domain.actor_entity_type,
                    predicate=self.domain.concentration_predicate,
                    object_id=str(row[cp_col]),
                    object_type=self.domain.counterparty_entity_type,
                    confidence=float(row["confidence"]),
                    method="concentration_analysis",
                    properties={
                        "event_count": int(row["event_count"]),
                        "total_events": int(row["actor_total"]),
                        "concentration_ratio": round(float(row["concentration"]), 3),
                    },
                )
            )
        return triplets

    @staticmethod
    def _find_column(df, candidates: list[str]) -> str | None:
        cols_lower = {c.lower(): c for c in df.columns}
        for candidate in candidates:
            if candidate.lower() in cols_lower:
                return cols_lower[candidate.lower()]
        return None
