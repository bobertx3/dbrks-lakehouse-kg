"""Relationship Agent - Discovers direct structural relationships between entities."""

from __future__ import annotations

from typing import Any

from pyspark.sql import functions as F

from lakehouse_kg.agents.base import BaseAgent, AgentResult, Triplet


class RelationshipAgent(BaseAgent):
    """Discovers direct FK-based relationships and generates structural triplets.
    This is the rule-based backbone — high confidence, no ML required.
    Predicate names for known table pairs come from the domain pack's
    fk_predicates map; unmapped pairs fall back to RELATED_TO_<TABLE>."""

    @property
    def name(self) -> str:
        return "relationship_discovery"

    @property
    def question(self) -> str:
        return "What direct structural relationships exist between entities?"

    def execute(self, context: dict[str, Any]) -> AgentResult:
        result = AgentResult(agent_name=self.name)
        fk_map = context.get("foreign_keys", {})
        schemas = context.get("schemas", {})
        shared_attrs = context.get("shared_attributes", [])

        # Phase 1: FK-based triplets
        fk_triplets = self._generate_fk_triplets(fk_map, schemas)
        result.triplets.extend(fk_triplets)
        result.questions_asked.append(
            f"What FK joins exist? Found {len(fk_map)} tables with foreign keys."
        )
        result.answers.append(f"Generated {len(fk_triplets)} FK-based triplets")

        # Phase 2: Shared-attribute triplets (e.g., same address, same IP)
        shared_triplets = self._generate_shared_attribute_triplets(shared_attrs, schemas)
        result.triplets.extend(shared_triplets)
        result.questions_asked.append(
            f"Do any entities share attributes? Found {len(shared_attrs)} shared columns."
        )
        result.answers.append(f"Generated {len(shared_triplets)} shared-attribute triplets")

        result.metadata["fk_triplet_count"] = len(fk_triplets)
        result.metadata["shared_triplet_count"] = len(shared_triplets)
        return result

    def _generate_fk_triplets(
        self, fk_map: dict[str, list[dict]], schemas: dict[str, dict]
    ) -> list[Triplet]:
        triplets = []
        fk_predicates = self.domain.fk_predicates

        for table_name, fk_list in fk_map.items():
            for fk_info in fk_list:
                fk_col = fk_info["fk_column"]
                ref_table = fk_info["referenced_table"]

                predicate = fk_predicates.get(
                    (table_name, ref_table),
                    fk_predicates.get(
                        (ref_table, table_name),
                        f"RELATED_TO_{ref_table.upper()}",
                    ),
                )

                try:
                    source_df = self._read_table(table_name)
                    pk_col = self._get_primary_key(table_name, schemas)
                    if not pk_col:
                        continue

                    pairs = (
                        source_df.select(
                            F.col(pk_col).alias("subject"),
                            F.col(fk_col).alias("object"),
                        )
                        .filter(F.col("object").isNotNull())
                        .distinct()
                    )

                    pairs = pairs.limit(self.config.max_triplets_per_agent)

                    subject_type = self._entity_type(table_name)
                    object_type = self._entity_type(ref_table)

                    pdf = pairs.toPandas()
                    for _, row in pdf.iterrows():
                        triplets.append(
                            self._make_triplet(
                                subject_id=str(row["subject"]),
                                subject_type=subject_type,
                                predicate=predicate,
                                object_id=str(row["object"]),
                                object_type=object_type,
                                confidence=1.0,
                                method="foreign_key",
                                properties={
                                    "source_table": table_name,
                                    "fk_column": fk_col,
                                    "ref_table": ref_table,
                                },
                            )
                        )
                except Exception as e:
                    self.logger.warning(f"FK triplet generation failed for {table_name}.{fk_col}: {e}")

        return triplets

    def _generate_shared_attribute_triplets(
        self, shared_attrs: list[dict], schemas: dict[str, dict]
    ) -> list[Triplet]:
        """Generate triplets for entities sharing the same attribute value.
        Handles both cross-table matches (e.g., matching city values across
        tables) and within-table matches (e.g., two entities at the same
        address)."""
        triplets = []
        identity_keywords = self.domain.identity_attribute_keywords
        interesting_attrs = [
            a for a in shared_attrs
            if any(k in a["column"].lower() for k in identity_keywords)
        ]

        # Deduplicate: entity agent may emit the same column/table pair
        seen_pairs: set[tuple[str, str, str]] = set()

        for attr in interesting_attrs:
            tables = attr["tables"]
            col = attr["column"]

            # Generate all unique table pairs (including self-pairs for same-table)
            for i in range(len(tables)):
                for j in range(i, len(tables)):
                    t1, t2 = tables[i], tables[j]
                    pair_key = (col, min(t1, t2), max(t1, t2))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    pk1 = self._get_primary_key(t1, schemas)
                    pk2 = self._get_primary_key(t2, schemas)
                    if not pk1 or not pk2:
                        continue

                    try:
                        df1 = self._read_table(t1).select(
                            F.col(pk1).alias("id1"), F.col(col).alias("shared_val")
                        ).filter(F.col("shared_val").isNotNull())
                        df2 = self._read_table(t2).select(
                            F.col(pk2).alias("id2"), F.col(col).alias("shared_val")
                        ).filter(F.col("shared_val").isNotNull())

                        joined = df1.join(df2, "shared_val").select("id1", "id2", "shared_val")
                        if t1 == t2:
                            joined = joined.filter(F.col("id1") < F.col("id2"))

                        joined = joined.limit(self.config.batch_size)

                        predicate = f"SHARES_{col.upper()}_WITH"
                        pdf = joined.toPandas()
                        for _, row in pdf.iterrows():
                            triplets.append(
                                self._make_triplet(
                                    subject_id=str(row["id1"]),
                                    subject_type=self._entity_type(t1),
                                    predicate=predicate,
                                    object_id=str(row["id2"]),
                                    object_type=self._entity_type(t2),
                                    confidence=0.9,
                                    method="shared_attribute",
                                    properties={
                                        "shared_column": col,
                                        "shared_value": str(row["shared_val"]),
                                    },
                                )
                            )
                    except Exception as e:
                        self.logger.warning(
                            f"Shared attribute triplet generation failed for {col} "
                            f"({t1}<->{t2}): {e}"
                        )

        return triplets

    def _get_primary_key(self, table_name: str, schemas: dict) -> str | None:
        table_info = schemas.get(table_name, {})
        keys = table_info.get("potential_keys", [])
        if keys:
            return keys[0]
        singular = table_name.rstrip("s")
        candidate = f"{singular}_id"
        if candidate in table_info.get("column_names", []):
            return candidate
        return None

    def _entity_type(self, table_name: str) -> str:
        return table_name.rstrip("s").capitalize()
