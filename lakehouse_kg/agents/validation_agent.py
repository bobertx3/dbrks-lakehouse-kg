"""Validation Agent - Validates, deduplicates, and scores the full triplet set."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from lakehouse_kg.agents.base import BaseAgent, AgentResult, Triplet


class ValidationAgent(BaseAgent):
    """Final-stage agent that validates the complete triplet set for consistency,
    removes duplicates, resolves conflicts, and assigns final confidence scores
    based on multi-source corroboration. Contradictory predicate pairs come
    from the domain pack."""

    @property
    def name(self) -> str:
        return "validation"

    @property
    def question(self) -> str:
        return "Are the generated triplets valid, consistent, and properly scored?"

    def execute(self, context: dict[str, Any]) -> AgentResult:
        result = AgentResult(agent_name=self.name)
        prior_triplets: list[Triplet] = context.get("prior_triplets", [])

        if not prior_triplets:
            result.answers.append("No triplets to validate.")
            return result

        initial_count = len(prior_triplets)

        # Sub-question 1: Deduplication
        result.questions_asked.append(
            f"Are there duplicate triplets among the {initial_count} generated?"
        )
        deduplicated = self._deduplicate(prior_triplets)
        dup_count = initial_count - len(deduplicated)
        result.answers.append(f"Removed {dup_count} duplicate triplets")

        # Sub-question 2: Consistency check
        result.questions_asked.append(
            "Are there logically contradictory triplets?"
        )
        consistent, contradiction_count = self._check_consistency(deduplicated)
        result.answers.append(f"Found and resolved {contradiction_count} contradictions")

        # Sub-question 3: Corroboration boost
        result.questions_asked.append(
            "Which triplets are corroborated by multiple agents (higher trust)?"
        )
        corroborated = self._apply_corroboration(consistent, prior_triplets)
        result.answers.append("Applied multi-source corroboration scoring")

        # Sub-question 4: Confidence filtering
        result.questions_asked.append(
            f"Which triplets meet the minimum confidence threshold of {self.config.min_confidence}?"
        )
        filtered = [t for t in corroborated if t.confidence >= self.config.min_confidence]
        result.answers.append(
            f"Retained {len(filtered)} of {len(corroborated)} triplets above threshold"
        )

        result.triplets = filtered
        result.metadata["validation_summary"] = {
            "initial_count": initial_count,
            "after_dedup": len(deduplicated),
            "duplicates_removed": dup_count,
            "contradictions_resolved": contradiction_count,
            "after_filtering": len(filtered),
            "predicate_distribution": dict(Counter(t.predicate for t in filtered)),
            "agent_distribution": dict(Counter(t.source_agent for t in filtered)),
            "method_distribution": dict(Counter(t.source_method for t in filtered)),
            "avg_confidence": (
                sum(t.confidence for t in filtered) / len(filtered) if filtered else 0
            ),
        }
        return result

    def _deduplicate(self, triplets: list[Triplet]) -> list[Triplet]:
        """Remove exact duplicate triplets, keeping the highest confidence version."""
        seen: dict[str, Triplet] = {}
        for t in triplets:
            key = self._triplet_key(t)
            if key not in seen or t.confidence > seen[key].confidence:
                seen[key] = t
        return list(seen.values())

    def _check_consistency(
        self, triplets: list[Triplet]
    ) -> tuple[list[Triplet], int]:
        """Check for logically contradictory triplets and resolve conflicts."""
        contradiction_count = 0
        contradictory_pairs = self.domain.contradictory_predicate_pairs

        subject_predicates: dict[str, dict[str, Triplet]] = {}
        for t in triplets:
            subject_predicates.setdefault(t.subject_id, {})[t.predicate] = t

        to_remove = set()
        for subj, preds in subject_predicates.items():
            for pos, neg in contradictory_pairs:
                if pos in preds and neg in preds:
                    contradiction_count += 1
                    if preds[pos].confidence >= preds[neg].confidence:
                        to_remove.add(self._triplet_key(preds[neg]))
                    else:
                        to_remove.add(self._triplet_key(preds[pos]))

        filtered = [t for t in triplets if self._triplet_key(t) not in to_remove]
        return filtered, contradiction_count

    def _apply_corroboration(
        self, deduplicated: list[Triplet], all_originals: list[Triplet]
    ) -> list[Triplet]:
        """Boost confidence for triplets corroborated by multiple agents."""
        relationship_agents: dict[str, set[str]] = {}
        for t in all_originals:
            rel_key = f"{t.subject_id}|{t.predicate}|{t.object_id}"
            relationship_agents.setdefault(rel_key, set()).add(t.source_agent)

        # Also count relationships that are semantically similar
        entity_pair_agents: dict[str, set[str]] = {}
        for t in all_originals:
            pair_key = f"{t.subject_id}|{t.object_id}"
            entity_pair_agents.setdefault(pair_key, set()).add(t.source_agent)

        for t in deduplicated:
            rel_key = f"{t.subject_id}|{t.predicate}|{t.object_id}"
            pair_key = f"{t.subject_id}|{t.object_id}"

            exact_sources = len(relationship_agents.get(rel_key, set()))
            pair_sources = len(entity_pair_agents.get(pair_key, set()))

            if exact_sources > 1:
                boost = 0.1 * (exact_sources - 1)
                t.confidence = min(1.0, t.confidence + boost)
                base_props = json.loads(t.properties) if isinstance(t.properties, str) else t.properties
                t.properties = {
                    **base_props,
                    "corroboration_count": exact_sources,
                    "corroborating_agents": list(relationship_agents[rel_key]),
                }
            elif pair_sources > 1:
                boost = 0.05 * (pair_sources - 1)
                t.confidence = min(1.0, t.confidence + boost)

        return deduplicated

    @staticmethod
    def _triplet_key(t: Triplet) -> str:
        raw = f"{t.subject_id}|{t.subject_type}|{t.predicate}|{t.object_id}|{t.object_type}"
        return hashlib.md5(raw.encode()).hexdigest()
