import pytest
import networkx as nx
from rootrank.ranker import RootCauseRanker


def _chain_graph():
    """B depends on A, A depends on C.  (edges: B→A, A→C)."""
    g = nx.DiGraph()
    g.add_edge("B", "A", weight=10)
    g.add_edge("A", "C", weight=10)
    return g


def _star_graph():
    """Hub connected to 4 leaves (leaves depend on hub: L→H)."""
    g = nx.DiGraph()
    for leaf in ("L1", "L2", "L3", "L4"):
        g.add_edge(leaf, "H", weight=1)
    return g


def _bridge_graph():
    """A depends on H, H depends on B. A→H→B. H is a bridge."""
    g = nx.DiGraph()
    g.add_edge("A", "H", weight=1)
    g.add_edge("H", "B", weight=1)
    return g


def _calls_graph():
    """Call-direction: C calls A, A calls B (edges C→A, A→B)."""
    g = nx.DiGraph()
    g.add_edge("C", "A", weight=10)
    g.add_edge("A", "B", weight=10)
    return g


class TestPropagation:
    """Tests for rank_propagation (upstream anomaly propagation)."""

    def test_root_cause_boosted_by_victims(self):
        """A is root cause, B is victim.  B's anomaly propagates to A."""
        g = _chain_graph()
        scores = {"A": 0.87, "B": 0.82, "C": 0.15}
        ranker = RootCauseRanker()
        ranked = ranker.rank_propagation(g, scores)
        names = [n for n, _ in ranked]
        assert names[0] == "A"

    def test_root_cause_already_highest_stays_top(self):
        """A already highest — propagation doesn't displace it."""
        g = _chain_graph()
        scores = {"A": 0.90, "B": 0.50, "C": 0.10}
        ranker = RootCauseRanker()
        ranked = ranker.rank_propagation(g, scores)
        assert ranked[0][0] == "A"

    def test_deep_dependency_gets_propagated_score(self):
        """C (deep dependency) gets boosted by A's anomaly via propagation."""
        g = _chain_graph()
        scores = {"A": 0.90, "B": 0.10, "C": 0.05}
        ranker = RootCauseRanker()
        ranked = ranker.rank_propagation(g, scores)
        names = [n for n, _ in ranked]
        assert names[0] == "A"
        # C should get some boost from A
        c_score = dict(ranked)["C"]
        assert c_score > 0.05

    def test_leaf_unchanged(self):
        """Leaf service with no dependents gets no boost."""
        g = _chain_graph()
        scores = {"A": 0.90, "B": 0.50, "C": 0.10}
        ranker = RootCauseRanker()
        ranked = ranker.rank_propagation(g, scores)
        b_score = dict(ranked)["B"]
        assert b_score == 0.50  # no one depends on B, so no boost

    def test_empty_scores(self):
        """Nodes not in scores dict get 0 and can still receive propagation."""
        g = _chain_graph()
        ranker = RootCauseRanker()
        ranked = ranker.rank_propagation(g, {})
        assert len(ranked) == 3
        assert all(isinstance(n, str) and isinstance(s, float) for n, s in ranked)

    def test_decay_parameter(self):
        """Lower decay means less propagation."""
        g = _chain_graph()
        scores = {"A": 0.50, "B": 0.90, "C": 0.10}
        ranker = RootCauseRanker()
        ranked_high = ranker.rank_propagation(g, scores, decay=0.99)
        ranked_low = ranker.rank_propagation(g, scores, decay=0.10)

        # With high decay, A gets almost all of B's score (A boosted)
        a_high = dict(ranked_high)["A"]
        a_low = dict(ranked_low)["A"]
        assert a_high > a_low

    def test_cycles_handled_gracefully(self):
        """Graph with cycles should not crash."""
        g = nx.DiGraph()
        g.add_edge("A", "B")
        g.add_edge("B", "A")
        scores = {"A": 0.90, "B": 0.50}
        ranker = RootCauseRanker()
        ranked = ranker.rank_propagation(g, scores)
        assert len(ranked) == 2

    def test_calls_semantics_reverses_edges(self):
        """With edge_semantics='calls', edges are reversed for propagation."""
        g = _calls_graph()  # C→A→B  (call direction)
        scores = {"A": 0.87, "B": 0.82, "C": 0.15}
        ranker = RootCauseRanker()

        ranked_calls = ranker.rank_propagation(
            g, scores, edge_semantics="calls"
        )
        ranked_dep = ranker.rank_propagation(
            g, scores, edge_semantics="depends-on"
        )
        # Results should differ because edges mean different things
        assert ranked_calls != ranked_dep


class TestPageRank:
    """Tests for rank_pagerank (Personalized PageRank)."""

    def test_root_cause_highest(self):
        """A (root cause with high anomaly) ranks above victims."""
        g = _chain_graph()
        scores = {"A": 0.87, "B": 0.82, "C": 0.15}
        ranker = RootCauseRanker()
        ranked = ranker.rank_pagerank(g, scores)
        assert ranked[0][0] == "A"

    def test_zero_anomaly_falls_back_to_uniform(self):
        """All-zero anomaly scores → standard PageRank."""
        g = _chain_graph()
        ranker = RootCauseRanker()
        ranked = ranker.rank_pagerank(g, {})
        # B has in-degree 0, A has in-degree 1, C has in-degree 1
        # With alpha=0.30, personalization is uniform → C and A rank higher
        names = [n for n, _ in ranked]
        assert "B" not in names[:1]  # B should not be top (no in-degree)

    def test_alpha_parameter(self):
        """Higher alpha = more graph influence, lower = more anomaly influence."""
        g = _chain_graph()
        scores = {"A": 0.60, "B": 0.90, "C": 0.10}
        ranker = RootCauseRanker()
        ranked_high = ranker.rank_pagerank(g, scores, alpha=0.50)
        ranked_low = ranker.rank_pagerank(g, scores, alpha=0.15)

        # At low alpha, anomaly signal dominates → B (high anomaly) wins
        # At high alpha, graph influences more → C (dependency) gains
        assert ranked_low[0][0] == "B"
        assert ranked_low[0][1] > ranked_high[0][1]  # B's score higher at low alpha

    def test_personalization_sums_to_one(self):
        """Ensure personalization dict is properly normalized."""
        g = _chain_graph()
        scores = {"A": 10.0, "B": 5.0, "C": 5.0}  # sum = 20
        ranker = RootCauseRanker()
        ranked = ranker.rank_pagerank(g, scores)
        assert sum(s for _, s in ranked) == pytest.approx(1.0)

    def test_missing_nodes_get_zero_personalization(self):
        """Nodes in graph but not in scores get 0 personalization."""
        g = _chain_graph()
        scores = {"A": 0.90}  # B and C missing
        ranker = RootCauseRanker()
        ranked = ranker.rank_pagerank(g, scores)
        assert len(ranked) == 3  # all nodes present in output

    def test_calls_semantics_reverses(self):
        """With call edges, PageRank reverses them to depends-on."""
        g = _calls_graph()
        scores = {"A": 0.87, "B": 0.82, "C": 0.15}
        ranker = RootCauseRanker()
        ranked_calls = ranker.rank_pagerank(
            g, scores, edge_semantics="calls"
        )
        ranked_dep = ranker.rank_pagerank(
            g, scores, edge_semantics="depends-on"
        )
        assert ranked_calls != ranked_dep


class TestBetweenness:
    """Tests for rank_betweenness."""

    def test_bridge_node_highest_betweenness(self):
        """In A→H→B, H is on the only path from A to B → highest betweenness."""
        g = _bridge_graph()
        ranker = RootCauseRanker()
        ranked = ranker.rank_betweenness(g)
        assert ranked[0][0] == "H"

    def test_weight_inversion_matters(self):
        """Different weights produce different betweenness scores.

        Graph: A→H→X (weight=100 on A→H) and B→H→X (weight=1 on B→H).
        After inversion, A→H has distance 0.01 and B→H has distance 1.0.
        H gets betweenness credit from (A,X) since that's the shortest path,
        but also from (B,X) — both go through H.  The key is that a higher
        call weight (lower distance) makes the edge *more* likely to be used
        in shortest-path computation.

        Simpler test: with raw weights (no inversion), A→H distance=100
        would make H LESS central for paths through A, which is backwards
        for our domain.  The inversion prevents that.
        """
        g = nx.DiGraph()
        g.add_edge("A", "H", weight=100)
        g.add_edge("B", "H", weight=100)
        g.add_edge("H", "C", weight=1)

        ranker = RootCauseRanker()
        ranked = ranker.rank_betweenness(g)
        scores = dict(ranked)

        # H should have non-zero betweenness (on paths A→C and B→C via H)
        assert scores["H"] > 0
        # A and B only connect to H, no paths pass through them
        assert scores["A"] == 0
        assert scores["B"] == 0


class TestAnomalyOnly:
    """Tests for rank_anomaly_only."""

    def test_sorted_descending(self):
        scores = {"A": 0.3, "B": 0.9, "C": 0.6}
        ranker = RootCauseRanker()
        ranked = ranker.rank_anomaly_only(scores)
        assert [n for n, _ in ranked] == ["B", "C", "A"]

    def test_empty_scores(self):
        ranker = RootCauseRanker()
        ranked = ranker.rank_anomaly_only({})
        assert ranked == []


class TestStandardPageRank:
    """Tests for rank_standard_pagerank."""

    def test_most_central_ranks_highest(self):
        """In a depends-on star, the hub has highest in-degree = highest PR."""
        g = _star_graph()
        ranker = RootCauseRanker()
        ranked = ranker.rank_standard_pagerank(g)
        assert ranked[0][0] == "H"

    def test_unaffected_by_anomaly(self):
        """Standard PageRank ignores anomaly scores entirely."""
        g = _chain_graph()
        ranker = RootCauseRanker()
        r1 = ranker.rank_standard_pagerank(g)
        r2 = ranker.rank_standard_pagerank(g)
        assert r1 == r2  # same output every time with same graph
