"""LLM Semantic Agent - Uses foundation models for semantic relationship discovery."""

from __future__ import annotations

import json
import re
from typing import Any

from lakehouse_kg.agents.base import BaseAgent, AgentResult, Triplet


class LLMSemanticAgent(BaseAgent):
    """Uses a foundation model to discover semantic relationships that
    rule-based and statistical methods miss. Asks the LLM to reason about
    schema structure and entity patterns. The persona, prompt templates,
    and fallback predicate come from the domain pack."""

    @property
    def name(self) -> str:
        return "llm_semantic"

    @property
    def question(self) -> str:
        return self.domain.llm_question

    def execute(self, context: dict[str, Any]) -> AgentResult:
        result = AgentResult(agent_name=self.name)
        schemas = context.get("schemas", {})
        prior_predicates = self._collect_prior_predicates(context)

        # Sub-question 1: Schema-level semantic relationships
        result.questions_asked.append(
            "What implicit relationships exist in the schema that aren't captured by FKs?"
        )
        schema_triplets = self._schema_level_analysis(schemas, prior_predicates)
        result.triplets.extend(schema_triplets)
        result.answers.append(
            f"LLM identified {len(schema_triplets)} schema-level semantic relationships"
        )

        # Sub-question 2: Entity-level pattern analysis (sample-based)
        result.questions_asked.append(self.domain.llm_entity_question)
        entity_triplets = self._entity_level_analysis(schemas)
        result.triplets.extend(entity_triplets)
        result.answers.append(
            f"LLM identified {len(entity_triplets)} entity-level patterns"
        )

        return result

    def _schema_level_analysis(
        self, schemas: dict, prior_predicates: set[str]
    ) -> list[Triplet]:
        """Ask the LLM to reason about the schema and suggest relationships."""
        schema_summary = self._build_schema_summary(schemas)
        prompt = self.domain.llm_schema_prompt.format(
            schema_summary=schema_summary,
            existing_predicates=", ".join(sorted(prior_predicates)) or "None yet",
        )

        response = self._call_llm(prompt)
        return self._parse_schema_triplets(response)

    def _entity_level_analysis(self, schemas: dict) -> list[Triplet]:
        """Sample entities and ask the LLM to identify notable patterns."""
        entity_samples = self._collect_entity_samples(schemas)
        if not entity_samples:
            return []

        prompt = self.domain.llm_entity_prompt.format(entity_samples=entity_samples)
        response = self._call_llm(prompt)
        return self._parse_entity_triplets(response)

    def _call_llm(self, prompt: str) -> str:
        """Call the Databricks foundation model endpoint via the OpenAI-compatible API.

        Auth and workspace URL come from the SDK's unified credential resolution
        (notebook context, job context, serverless, env vars, or CLI profile),
        so this works identically on classic clusters, serverless, and locally.
        """
        try:
            from databricks.sdk import WorkspaceClient

            client = WorkspaceClient().serving_endpoints.get_open_ai_client()
            response = client.chat.completions.create(
                model=self.config.llm_endpoint,
                messages=[
                    {"role": "system", "content": self.domain.llm_system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.config.llm_max_tokens,
                temperature=self.config.llm_temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return self._call_llm_sdk_fallback(prompt)

    def _call_llm_sdk_fallback(self, prompt: str) -> str:
        """Fallback: query the endpoint through the SDK's typed serving API."""
        try:
            from databricks.sdk import WorkspaceClient
            from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

            w = WorkspaceClient()
            response = w.serving_endpoints.query(
                name=self.config.llm_endpoint,
                messages=[
                    ChatMessage(role=ChatMessageRole.SYSTEM, content=self.domain.llm_system_prompt),
                    ChatMessage(role=ChatMessageRole.USER, content=prompt),
                ],
                max_tokens=self.config.llm_max_tokens,
                temperature=self.config.llm_temperature,
            )
            return response.choices[0].message.content or "[]"
        except Exception as e:
            self.logger.error(f"SDK fallback also failed: {e}")
            return "[]"

    def _build_schema_summary(self, schemas: dict) -> str:
        lines = []
        for table_name, info in schemas.items():
            cols = info.get("columns", {})
            col_descriptions = []
            for col_name, col_meta in cols.items():
                col_descriptions.append(
                    f"  - {col_name} ({col_meta['type']}, "
                    f"unique: {col_meta['uniqueness_ratio']:.1%}, "
                    f"nulls: {col_meta['null_rate']:.1%})"
                )
            lines.append(
                f"Table: {table_name} ({info['row_count']} rows)\n"
                + "\n".join(col_descriptions)
            )
        return "\n\n".join(lines)

    def _collect_entity_samples(self, schemas: dict, sample_size: int = 20) -> str:
        """Collect small samples from entity tables for LLM analysis."""
        samples = []
        for table_name, info in schemas.items():
            if info["row_count"] == 0:
                continue
            try:
                df = self._read_table(table_name)
                sample = df.limit(sample_size).toPandas()
                samples.append(f"--- {table_name} (sample of {len(sample)}) ---\n{sample.to_string()}")
            except Exception:
                pass
        return "\n\n".join(samples[:5])

    def _collect_prior_predicates(self, context: dict) -> set[str]:
        prior_triplets = context.get("prior_triplets", [])
        return {t.predicate for t in prior_triplets if hasattr(t, "predicate")}

    def _parse_schema_triplets(self, response: str) -> list[Triplet]:
        """Parse LLM response for schema-level relationship suggestions."""
        triplets = []
        parsed = self._safe_parse_json(response)
        for item in parsed:
            if not isinstance(item, dict):
                continue
            try:
                triplets.append(
                    self._make_triplet(
                        subject_id=item.get("subject_description", item.get("subject_type", "unknown")),
                        subject_type=item.get("subject_type", self.domain.default_entity_type),
                        predicate=item.get("predicate", "RELATED_TO"),
                        object_id=item.get("object_description", item.get("object_type", "unknown")),
                        object_type=item.get("object_type", self.domain.default_entity_type),
                        confidence=float(item.get("confidence", 0.5)),
                        method="llm_schema_analysis",
                        properties={
                            "reasoning": item.get("reasoning", ""),
                            "llm_endpoint": self.config.llm_endpoint,
                        },
                    )
                )
            except (KeyError, ValueError) as e:
                self.logger.debug(f"Skipping malformed LLM triplet: {e}")
        return triplets

    def _parse_entity_triplets(self, response: str) -> list[Triplet]:
        """Parse LLM response for entity-level patterns."""
        triplets = []
        parsed = self._safe_parse_json(response)
        for item in parsed:
            if not isinstance(item, dict):
                continue
            try:
                triplets.append(
                    self._make_triplet(
                        subject_id=str(item.get("subject_id", "unknown")),
                        subject_type=item.get("subject_type", self.domain.default_entity_type),
                        predicate=item.get("predicate", self.domain.default_llm_predicate),
                        object_id=str(item.get("object_id", "unknown")),
                        object_type=item.get("object_type", self.domain.default_entity_type),
                        confidence=float(item.get("confidence", 0.4)),
                        method="llm_entity_analysis",
                        properties={
                            "reasoning": item.get("reasoning", ""),
                            "llm_endpoint": self.config.llm_endpoint,
                        },
                    )
                )
            except (KeyError, ValueError) as e:
                self.logger.debug(f"Skipping malformed LLM entity triplet: {e}")
        return triplets

    @staticmethod
    def _safe_parse_json(text: str) -> list:
        """Robustly extract a JSON array from LLM output."""
        text = text.strip()
        # Try direct parse
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            pass
        # Try extracting from markdown code block
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        # Try finding array boundaries
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        return []
