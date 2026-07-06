"""Graph Topology Agent - Runs graph algorithms on generated triplets to find higher-order patterns."""

from __future__ import annotations

from typing import Any

import networkx as nx

from lakehouse_kg.agents.base import BaseAgent, AgentResult, Triplet


class GraphTopologyAgent(BaseAgent):
    """Analyzes the triplet graph itself to discover higher-order structural
    patterns: hub entities, bridges between communities, tightly connected
    groups, and risk propagation paths. This agent runs AFTER other agents
    have populated the initial triplet set. Predicates, question phrasing,
    and risk-seed predicates come from the domain pack."""

    @property
    def name(self) -> str:
        return "graph_topology"

    @property
    def question(self) -> str:
        return "What higher-order network patterns emerge from the generated triplets?"

    def execute(self, context: dict[str, Any]) -> AgentResult:
        result = AgentResult(agent_name=self.name)
        prior_triplets = context.get("prior_triplets", [])

        if not prior_triplets:
            result.answers.append("No prior triplets to analyze.")
            return result

        G = self._build_networkx_graph(prior_triplets)
        result.metadata["graph_stats"] = {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "components": nx.number_connected_components(G.to_undirected())
            if G.number_of_nodes() > 0
            else 0,
        }

        # Sub-question 1: Hub entities (high centrality)
        result.questions_asked.append(self.domain.hub_question)
        hub_triplets = self._hub_detection(G)
        result.triplets.extend(hub_triplets)
        result.answers.append(f"Found {len(hub_triplets)} hub entity triplets")

        # Sub-question 2: Community detection
        result.questions_asked.append(self.domain.community_question)
        community_triplets = self._community_detection(G)
        result.triplets.extend(community_triplets)
        result.answers.append(f"Found {len(community_triplets)} community membership triplets")

        # Sub-question 3: Bridge entities
        result.questions_asked.append(self.domain.bridge_question)
        bridge_triplets = self._bridge_detection(G)
        result.triplets.extend(bridge_triplets)
        result.answers.append(f"Found {len(bridge_triplets)} bridge entity triplets")

        # Sub-question 4: Risk propagation
        result.questions_asked.append(self.domain.risk_question)
        risk_triplets = self._risk_propagation(G, prior_triplets)
        result.triplets.extend(risk_triplets)
        result.answers.append(f"Generated {len(risk_triplets)} risk propagation triplets")

        return result

    def _build_networkx_graph(self, triplets: list[Triplet]) -> nx.DiGraph:
        G = nx.DiGraph()
        for t in triplets:
            node_s = f"{t.subject_type}:{t.subject_id}"
            node_o = f"{t.object_type}:{t.object_id}"
            G.add_node(node_s, entity_type=t.subject_type, entity_id=t.subject_id)
            G.add_node(node_o, entity_type=t.object_type, entity_id=t.object_id)
            G.add_edge(
                node_s,
                node_o,
                predicate=t.predicate,
                confidence=t.confidence,
                source_agent=t.source_agent,
            )
        return G

    def _hub_detection(self, G: nx.DiGraph) -> list[Triplet]:
        """Identify hub nodes using PageRank and degree centrality."""
        triplets = []
        if G.number_of_nodes() == 0:
            return triplets

        pagerank = nx.pagerank(G, alpha=0.85)
        degree_centrality = nx.degree_centrality(G)

        pr_threshold = sorted(pagerank.values(), reverse=True)[
            min(20, len(pagerank) - 1)
        ] if len(pagerank) > 20 else self.config.pagerank_threshold

        for node, pr in pagerank.items():
            if pr >= pr_threshold:
                node_data = G.nodes[node]
                dc = degree_centrality.get(node, 0)
                triplets.append(
                    self._make_triplet(
                        subject_id=node_data.get("entity_id", node),
                        subject_type=node_data.get("entity_type", self.domain.default_entity_type),
                        predicate=self.domain.hub_predicate,
                        object_id=f"hub_rank_{len(triplets) + 1}",
                        object_type="NetworkRole",
                        confidence=min(1.0, pr * 100),
                        method="pagerank",
                        properties={
                            "pagerank": round(pr, 6),
                            "degree_centrality": round(dc, 4),
                            "degree": G.degree(node),
                        },
                    )
                )
        return triplets

    def _community_detection(self, G: nx.DiGraph) -> list[Triplet]:
        """Detect communities using Louvain on the undirected projection."""
        triplets = []
        if G.number_of_nodes() < 3:
            return triplets

        try:
            G_undirected = G.to_undirected()
            communities = nx.community.louvain_communities(
                G_undirected, resolution=self.config.community_resolution, seed=42
            )

            for comm_id, community in enumerate(communities):
                if len(community) < 3:
                    continue
                for node in community:
                    node_data = G.nodes[node]
                    triplets.append(
                        self._make_triplet(
                            subject_id=node_data.get("entity_id", node),
                            subject_type=node_data.get("entity_type", self.domain.default_entity_type),
                            predicate=self.domain.community_predicate,
                            object_id=f"community_{comm_id}",
                            object_type="NetworkCommunity",
                            confidence=0.7,
                            method="louvain_community",
                            properties={
                                "community_id": comm_id,
                                "community_size": len(community),
                            },
                        )
                    )
        except Exception as e:
            self.logger.warning(f"Community detection failed: {e}")

        return triplets

    def _bridge_detection(self, G: nx.DiGraph) -> list[Triplet]:
        """Find bridge nodes connecting different communities (high betweenness centrality)."""
        triplets = []
        if G.number_of_nodes() < 5:
            return triplets

        try:
            betweenness = nx.betweenness_centrality(G)
            bc_sorted = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
            top_bridges = bc_sorted[: min(20, len(bc_sorted))]

            for node, bc in top_bridges:
                if bc < 0.01:
                    continue
                node_data = G.nodes[node]
                triplets.append(
                    self._make_triplet(
                        subject_id=node_data.get("entity_id", node),
                        subject_type=node_data.get("entity_type", self.domain.default_entity_type),
                        predicate=self.domain.bridge_predicate,
                        object_id=f"bridge_score_{round(bc, 3)}",
                        object_type="NetworkRole",
                        confidence=min(1.0, bc * 10),
                        method="betweenness_centrality",
                        properties={"betweenness_centrality": round(bc, 6)},
                    )
                )
        except Exception as e:
            self.logger.warning(f"Bridge detection failed: {e}")

        return triplets

    def _risk_propagation(
        self, G: nx.DiGraph, prior_triplets: list[Triplet]
    ) -> list[Triplet]:
        """Propagate risk scores from flagged entities through the network."""
        triplets = []

        seed_predicates = set(self.domain.risk_seed_predicates)

        seed_nodes = set()
        for t in prior_triplets:
            if t.predicate in seed_predicates:
                node = f"{t.subject_type}:{t.subject_id}"
                if node in G:
                    seed_nodes.add(node)

        if not seed_nodes:
            return triplets

        # BFS risk propagation: neighbors of flagged entities get risk scores
        risk_scores: dict[str, float] = {}
        for seed in seed_nodes:
            risk_scores[seed] = 1.0

        G_undirected = G.to_undirected()
        for seed in seed_nodes:
            try:
                for neighbor in G_undirected.neighbors(seed):
                    if neighbor not in seed_nodes:
                        current = risk_scores.get(neighbor, 0)
                        risk_scores[neighbor] = min(1.0, current + 0.5)

                        for n2 in G_undirected.neighbors(neighbor):
                            if n2 not in seed_nodes and n2 != seed:
                                current2 = risk_scores.get(n2, 0)
                                risk_scores[n2] = min(1.0, current2 + 0.25)
            except Exception:
                pass

        for node, score in risk_scores.items():
            if node in seed_nodes or score < self.config.min_confidence:
                continue
            node_data = G.nodes.get(node, {})
            triplets.append(
                self._make_triplet(
                    subject_id=node_data.get("entity_id", node),
                    subject_type=node_data.get("entity_type", self.domain.default_entity_type),
                    predicate=self.domain.risk_predicate,
                    object_id=f"risk_level_{round(score, 2)}",
                    object_type="RiskScore",
                    confidence=score * 0.8,
                    method="risk_propagation",
                    properties={
                        "propagated_risk_score": round(score, 3),
                        "hop_distance": 1 if score >= 0.5 else 2,
                    },
                )
            )

        return triplets
