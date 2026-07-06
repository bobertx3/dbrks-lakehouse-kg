"""ML Clustering Agent - Discovers hidden relationships through unsupervised learning."""

from __future__ import annotations

from typing import Any

import numpy as np
from pyspark.sql import functions as F

from lakehouse_kg.agents.base import BaseAgent, AgentResult, Triplet


class MLClusteringAgent(BaseAgent):
    """Uses clustering and similarity analysis to discover hidden entity
    relationships. Groups actors with similar behavioral fingerprints
    without requiring labeled data. Table/column heuristics, entity types,
    and predicates come from the domain pack."""

    @property
    def name(self) -> str:
        return "ml_clustering"

    @property
    def question(self) -> str:
        return self.domain.ml_question

    def execute(self, context: dict[str, Any]) -> AgentResult:
        result = AgentResult(agent_name=self.name)
        schemas = context.get("schemas", {})

        # Sub-question 1: Behavioral clustering
        result.questions_asked.append(self.domain.ml_clustering_question)
        cluster_triplets, cluster_meta = self._behavioral_clustering(schemas)
        result.triplets.extend(cluster_triplets)
        result.answers.append(
            f"Identified {cluster_meta.get('n_clusters', 0)} behavioral clusters, "
            f"generated {len(cluster_triplets)} triplets"
        )

        # Sub-question 2: Entity similarity within clusters
        result.questions_asked.append(
            "Within each cluster, which entities are most similar to each other?"
        )
        similarity_triplets = self._intra_cluster_similarity(cluster_meta)
        result.triplets.extend(similarity_triplets)
        result.answers.append(
            f"Found {len(similarity_triplets)} high-similarity entity pairs"
        )

        result.metadata.update(cluster_meta)
        return result

    def _behavioral_clustering(self, schemas: dict) -> tuple[list[Triplet], dict]:
        """Build behavioral feature vectors per actor and cluster them."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        triplets = []
        meta: dict[str, Any] = {}

        event_table = self._find_table(schemas, self.domain.event_table_keywords)
        if not event_table:
            return triplets, meta

        event_df = self._read_table(event_table)
        actor_col = self._find_col(event_df, self.domain.actor_id_columns)
        amt_col = self._find_col(event_df, self.domain.amount_columns)
        ts_col = self._find_col(event_df, self.domain.timestamp_columns)
        cp_col = self._find_col(event_df, self.domain.counterparty_id_columns)

        if not actor_col or not amt_col:
            return triplets, meta

        # Build per-actor feature vectors
        features_df = event_df.groupBy(actor_col).agg(
            F.count("*").alias("event_count"),
            F.mean(amt_col).alias("avg_amount"),
            F.stddev(amt_col).alias("std_amount"),
            F.max(amt_col).alias("max_amount"),
            F.min(amt_col).alias("min_amount"),
            F.sum(amt_col).alias("total_amount"),
        )

        if ts_col:
            ts_features = event_df.groupBy(actor_col).agg(
                F.mean(F.hour(F.col(ts_col))).alias("avg_hour"),
                F.stddev(F.hour(F.col(ts_col))).alias("std_hour"),
            )
            features_df = features_df.join(ts_features, actor_col)

        if cp_col:
            cp_features = event_df.groupBy(actor_col).agg(
                F.countDistinct(cp_col).alias("unique_counterparties"),
            )
            features_df = features_df.join(cp_features, actor_col)

        pdf = features_df.toPandas().fillna(0)
        if len(pdf) < self.config.n_clusters:
            return triplets, meta

        feature_cols = [c for c in pdf.columns if c != actor_col]
        X = pdf[feature_cols].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_clusters = min(self.config.n_clusters, len(pdf) // 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        pdf["cluster_id"] = labels
        pdf["distance_to_center"] = [
            float(np.linalg.norm(X_scaled[i] - kmeans.cluster_centers_[labels[i]]))
            for i in range(len(pdf))
        ]

        meta["n_clusters"] = n_clusters
        meta["cluster_assignments"] = pdf[[actor_col, "cluster_id", "distance_to_center"]].to_dict("records")
        meta["feature_cols"] = feature_cols
        meta["X_scaled"] = X_scaled
        meta["labels"] = labels
        meta["actor_col"] = actor_col
        meta["pdf"] = pdf

        # Generate cluster membership triplets
        for _, row in pdf.iterrows():
            cluster_label = f"behavioral_cluster_{int(row['cluster_id'])}"
            triplets.append(
                self._make_triplet(
                    subject_id=str(row[actor_col]),
                    subject_type=self.domain.actor_entity_type,
                    predicate=self.domain.cluster_predicate,
                    object_id=cluster_label,
                    object_type=self.domain.cluster_entity_type,
                    confidence=max(0.5, 1.0 - row["distance_to_center"] / 5.0),
                    method="kmeans_clustering",
                    properties={
                        "cluster_id": int(row["cluster_id"]),
                        "distance_to_center": round(row["distance_to_center"], 3),
                        "event_count": int(row.get("event_count", 0)),
                    },
                )
            )

        return triplets, meta

    def _intra_cluster_similarity(self, cluster_meta: dict) -> list[Triplet]:
        """Find highly similar actor pairs within the same cluster."""
        from sklearn.metrics.pairwise import cosine_similarity

        triplets = []
        X_scaled = cluster_meta.get("X_scaled")
        labels = cluster_meta.get("labels")
        pdf = cluster_meta.get("pdf")
        actor_col = cluster_meta.get("actor_col")

        if X_scaled is None or labels is None or pdf is None:
            return triplets

        for cluster_id in range(cluster_meta.get("n_clusters", 0)):
            mask = labels == cluster_id
            if mask.sum() < 2:
                continue

            cluster_X = X_scaled[mask]
            cluster_actors = pdf.loc[mask, actor_col].values

            # Only compute similarity for small enough clusters
            if len(cluster_actors) > 200:
                indices = np.random.RandomState(42).choice(
                    len(cluster_actors), 200, replace=False
                )
                cluster_X = cluster_X[indices]
                cluster_actors = cluster_actors[indices]

            sim_matrix = cosine_similarity(cluster_X)

            for i in range(len(cluster_actors)):
                for j in range(i + 1, len(cluster_actors)):
                    if sim_matrix[i, j] >= self.config.similarity_threshold:
                        triplets.append(
                            self._make_triplet(
                                subject_id=str(cluster_actors[i]),
                                subject_type=self.domain.actor_entity_type,
                                predicate=self.domain.similarity_predicate,
                                object_id=str(cluster_actors[j]),
                                object_type=self.domain.actor_entity_type,
                                confidence=float(sim_matrix[i, j]),
                                method="cosine_similarity",
                                properties={
                                    "cluster_id": int(cluster_id),
                                    "similarity_score": round(float(sim_matrix[i, j]), 4),
                                },
                            )
                        )
        return triplets

    def _find_table(self, schemas: dict, keywords: list[str]) -> str | None:
        for name in schemas:
            if any(k in name.lower() for k in keywords):
                return name
        return None

    @staticmethod
    def _find_col(df, candidates: list[str]) -> str | None:
        cols_lower = {c.lower(): c for c in df.columns}
        for c in candidates:
            if c.lower() in cols_lower:
                return cols_lower[c.lower()]
        return None
