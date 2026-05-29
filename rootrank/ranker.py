import networkx as nx


class RootCauseRanker:
    """Ranks components using graph structure + anomaly scores.

    Five algorithms:
      - rank_propagation     — upstream anomaly propagation (recommended default)
      - rank_pagerank        — Personalized PageRank with anomaly personalization
      - rank_standard_pagerank — uniform PageRank (graph-only baseline)
      - rank_betweenness     — betweenness centrality (structural importance)
      - rank_anomaly_only    — anomaly scores only (no-graph baseline)
    """

    def rank_propagation(
        self,
        graph: nx.DiGraph,
        anomaly_scores: dict[str, float],
        decay: float = 0.80,
        edge_semantics: str = "depends-on",
    ) -> list[tuple[str, float]]:
        """Propagate anomaly scores upstream along dependency edges.

        If B depends on A (edge B→A) and B shows anomalies, A's score is
        boosted — because A's failure would cause B's symptoms.  Processes
        nodes in topological order (leaves first) so propagation cascades
        upward through the entire dependency chain in a single pass.

        Args:
            graph: Dependency graph (depends-on edges).
            anomaly_scores: Ochiai scores dict — nodes not in the dict get 0.
            decay: Fraction of downstream anomaly that propagates upstream.
            edge_semantics: 'depends-on' (default) or 'calls' (auto-reversed).
        """
        g = graph.reverse(copy=True) if edge_semantics == "calls" else graph

        scores = {node: anomaly_scores.get(node, 0.0) for node in g.nodes()}

        try:
            order = list(nx.topological_sort(g))
        except nx.NetworkXUnfeasible:
            order = list(g.nodes())

        for node in order:
            node_score = scores[node]
            for dep in g.successors(node):
                boost = node_score * decay
                if boost > scores.get(dep, 0.0):
                    scores[dep] = boost

        return sorted(scores.items(), key=lambda item: item[1], reverse=True)

    def rank_pagerank(
        self,
        graph: nx.DiGraph,
        anomaly_scores: dict[str, float],
        alpha: float = 0.30,
        edge_semantics: str = "depends-on",
    ) -> list[tuple[str, float]]:
        """Personalized PageRank with anomaly scores as personalization vector.

        The personalization biases the random-surfer teleportation toward
        anomalous services.  The low default alpha (0.30) ensures the
        anomaly signal dominates the graph structure — the opposite of web
        PageRank where graph structure should dominate.

        Args:
            graph: Dependency graph (depends-on edges).
            anomaly_scores: Dict mapping node → suspiciousness score.
            alpha: Damping factor (0.15–0.50 recommended for RCA; default 0.30).
            edge_semantics: 'depends-on' or 'calls' (auto-reversed).
        """
        g = graph.reverse(copy=True) if edge_semantics == "calls" else graph

        total = sum(anomaly_scores.values())
        if total == 0:
            personalization = None
        else:
            personalization = {
                node: anomaly_scores.get(node, 0.0) / total
                for node in g.nodes()
            }

        raw = nx.pagerank(
            g,
            alpha=alpha,
            personalization=personalization,
            weight="weight",
            max_iter=100,
            tol=1e-06,
        )
        return sorted(raw.items(), key=lambda item: item[1], reverse=True)

    def rank_standard_pagerank(
        self,
        graph: nx.DiGraph,
        alpha: float = 0.30,
    ) -> list[tuple[str, float]]:
        """Standard (uniform) PageRank — structural importance baseline."""
        raw = nx.pagerank(
            graph, alpha=alpha, weight="weight", max_iter=100, tol=1e-06
        )
        return sorted(raw.items(), key=lambda item: item[1], reverse=True)

    def rank_betweenness(
        self, graph: nx.DiGraph
    ) -> list[tuple[str, float]]:
        """Betweenness centrality — structural bridge-ness.

        Edge weights are inverted (distance = 1/weight) because NetworkX
        treats weights as distances (lower = preferred path), whereas our
        dependency-graph weights represent call frequency (higher = more
        important connection).
        """
        inverted = graph.copy()
        for _u, _v, data in inverted.edges(data=True):
            w = data.get("weight", 1.0)
            data["distance"] = 1.0 / w if w > 0 else float("inf")

        scores = nx.betweenness_centrality(inverted, weight="distance")
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)

    def rank_anomaly_only(
        self, anomaly_scores: dict[str, float]
    ) -> list[tuple[str, float]]:
        """Rank by anomaly score alone — no-graph baseline."""
        return sorted(
            anomaly_scores.items(), key=lambda item: item[1], reverse=True
        )
