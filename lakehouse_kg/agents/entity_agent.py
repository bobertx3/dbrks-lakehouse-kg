"""Entity Resolution Agent - Identifies and classifies entities across datasets."""

from __future__ import annotations

from typing import Any

from lakehouse_kg.agents.base import BaseAgent, AgentResult


class EntityResolutionAgent(BaseAgent):
    """Identifies which columns represent entities and classifies them by type.
    Uses schema metadata from the Schema Discovery Agent and the domain pack's
    entity-type patterns to make determinations."""

    @property
    def name(self) -> str:
        return "entity_resolution"

    @property
    def question(self) -> str:
        return "What are the key entities in this data and how do they relate structurally?"

    def execute(self, context: dict[str, Any]) -> AgentResult:
        result = AgentResult(agent_name=self.name)
        schemas = context.get("schemas", {})

        if not schemas:
            result.error = "No schema information available. Run SchemaDiscoveryAgent first."
            return result

        entities: dict[str, dict] = {}
        entity_columns: dict[str, list[dict]] = {}
        fk_map: dict[str, list[dict]] = {}

        for table_name, table_info in schemas.items():
            table_entity_type = self._classify_table(table_name)

            for pk in table_info.get("potential_keys", []):
                entity_key = f"{table_name}.{pk}"
                entities[entity_key] = {
                    "table": table_name,
                    "id_column": pk,
                    "entity_type": table_entity_type,
                    "row_count": table_info["row_count"],
                }

            for col_name, col_meta in table_info.get("columns", {}).items():
                col_entity_type = self._classify_column(col_name)
                if col_entity_type:
                    entity_columns.setdefault(col_entity_type, []).append({
                        "table": table_name,
                        "column": col_name,
                        "distinct_count": col_meta["distinct_count"],
                    })

            for fk in table_info.get("potential_fks", []):
                referenced_table = self._infer_referenced_table(fk, schemas)
                if referenced_table:
                    fk_map.setdefault(table_name, []).append({
                        "fk_column": fk,
                        "referenced_table": referenced_table,
                    })

        # Detect shared-attribute entities (e.g., two entities at the same address)
        shared_attrs = self._find_shared_attributes(schemas)

        result.metadata["entities"] = entities
        result.metadata["entity_columns"] = entity_columns
        result.metadata["foreign_keys"] = fk_map
        result.metadata["shared_attributes"] = shared_attrs
        result.answers.append(
            f"Identified {len(entities)} primary entities across "
            f"{len(schemas)} tables with {sum(len(v) for v in fk_map.values())} FK relationships"
        )
        return result

    def _classify_table(self, table_name: str) -> str:
        name_lower = table_name.lower()
        for entity_type, patterns in self.domain.entity_type_patterns.items():
            if any(p in name_lower for p in patterns):
                return entity_type
        return self.domain.default_entity_type

    def _classify_column(self, col_name: str) -> str | None:
        name_lower = col_name.lower()
        if not name_lower.endswith("_id"):
            return None
        prefix = name_lower.replace("_id", "")
        for entity_type, patterns in self.domain.entity_type_patterns.items():
            if any(p in prefix for p in patterns):
                return entity_type
        return None

    def _infer_referenced_table(
        self, fk_column: str, schemas: dict[str, dict]
    ) -> str | None:
        """Guess which table a foreign key references based on naming convention."""
        prefix = fk_column.replace("_id", "")
        candidates = [prefix, f"{prefix}s", f"{prefix}es"]
        for candidate in candidates:
            if candidate in schemas:
                return candidate
        return None

    def _find_shared_attributes(self, schemas: dict[str, dict]) -> list[dict]:
        """Find columns that appear in multiple tables (potential join points)
        AND identity-like columns within a single table that have duplicate values
        (e.g., entities sharing an address)."""
        col_to_tables: dict[str, list[str]] = {}
        for table_name, table_info in schemas.items():
            for col_name in table_info.get("column_names", []):
                col_to_tables.setdefault(col_name, []).append(table_name)

        shared = []
        # Cross-table shared columns
        for col_name, tables in col_to_tables.items():
            if len(tables) > 1 and not col_name.startswith("_"):
                shared.append({"column": col_name, "tables": tables})

        # Within-table identity columns with duplicates (duplicate-identity patterns)
        identity_keywords = self.domain.identity_attribute_keywords
        for table_name, table_info in schemas.items():
            for col_name, col_meta in table_info.get("columns", {}).items():
                if not any(k in col_name.lower() for k in identity_keywords):
                    continue
                if col_meta["uniqueness_ratio"] < 0.95 and col_meta["distinct_count"] > 1:
                    shared.append({"column": col_name, "tables": [table_name, table_name]})

        return shared
