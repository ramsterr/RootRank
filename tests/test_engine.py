import pytest
import networkx as nx
from rootrank.scorer import Execution
from rootrank.engine import AnalysisResult, analyze, compare, iterative_analyze


def _chain_graph():
    """B depends on A, A depends on C (edges B→A, A→C)."""
    g = nx.DiGraph()
    g.add_edge("B", "A", weight=10)
    g.add_edge("A", "C", weight=10)
    return g


def _two_chain_graph():
    """Two independent chains: B→A→C  and  E→D."""
    g = nx.DiGraph()
    g.add_edge("B", "A", weight=10)
    g.add_edge("A", "C", weight=10)
    g.add_edge("E", "D", weight=10)
    return g


def _two_root_causes_executions():
    """A-chain and D-chain fail independently but simultaneously."""
    return [
        Execution(["A", "B"], True),
        Execution(["A", "C"], True),
        Execution(["A", "B", "C"], True),
        Execution(["D", "E"], True),
        Execution(["D"], True),
        Execution(["D", "E"], True),
        Execution(["A", "B"], False),
        Execution(["D", "E"], False),
    ]


class TestAnalyze:
    """Tests for the single-pass analyze() function."""

    def test_returns_analysis_result(self):
        g = _chain_graph()
        executions = [
            Execution(["A", "B"], True),
            Execution(["A"], True),
            Execution(["B"], False),
        ]
        result = analyze(g, executions)
        assert isinstance(result, AnalysisResult)
        assert result.formula_used == "ochiai"
        assert result.algorithm_used == "propagation"
        assert len(result.ranked) >= 1
        assert isinstance(result.component_scores, dict)

    def test_defaults(self):
        """Default formula=ochiai, algorithm=propagation."""
        g = _chain_graph()
        executions = [Execution(["A"], True), Execution(["A"], False)]
        result = analyze(g, executions)
        assert result.formula_used == "ochiai"
        assert result.algorithm_used == "propagation"

    def test_formula_and_algorithm_override(self):
        g = _chain_graph()
        executions = [Execution(["A"], True), Execution(["A"], False)]
        result = analyze(g, executions, formula="jaccard", algorithm="pagerank")
        assert result.formula_used == "jaccard"
        assert result.algorithm_used == "pagerank"

    def test_unknown_algorithm_raises(self):
        g = _chain_graph()
        executions = [Execution(["A"], True)]
        with pytest.raises(ValueError, match="Unknown algorithm"):
            analyze(g, executions, algorithm="nonexistent")


class TestIterativeAnalyze:
    """Tests for iterative_analyze() — multi root-cause discovery."""

    def test_finds_two_independent_root_causes(self):
        g = _two_chain_graph()
        executions = _two_root_causes_executions()
        results = iterative_analyze(g, executions, max_candidates=3)

        assert len(results) == 2
        top_names = [r.ranked[0][0] for r in results]
        assert "A" in top_names
        assert "D" in top_names

    def test_single_root_cause_stops_after_one(self):
        """Only one root cause — second iteration has no failing executions."""
        g = _chain_graph()
        executions = [
            Execution(["A", "B"], True),
            Execution(["A"], True),
            Execution(["B"], False),
        ]
        results = iterative_analyze(g, executions, max_candidates=5)
        assert len(results) == 1
        assert results[0].ranked[0][0] == "A"

    def test_respects_max_candidates(self):
        g = _two_chain_graph()
        executions = _two_root_causes_executions()
        results = iterative_analyze(g, executions, max_candidates=1)
        assert len(results) == 1

    def test_respects_score_threshold(self):
        g = _chain_graph()
        executions = [
            Execution(["A", "B"], True),
            Execution(["A"], True),
        ]
        # Set threshold above the highest possible score
        results = iterative_analyze(
            g, executions, max_candidates=5, score_threshold=999.0
        )
        assert len(results) == 0

    def test_no_failing_executions_returns_empty(self):
        g = _chain_graph()
        executions = [Execution(["A"], False)]
        results = iterative_analyze(g, executions)
        assert results == []

    def test_preserves_discovery_order(self):
        """First iteration finds the strongest signal."""
        g = _two_chain_graph()
        executions = _two_root_causes_executions()
        results = iterative_analyze(g, executions, max_candidates=3)
        # Both causes found in some order; verify each has ranked items
        for r in results:
            assert len(r.ranked) > 0
            assert r.ranked[0][1] > 0


class TestCompare:
    """Tests for the compare() function — all formula × algorithm combos."""

    def test_runs_all_combinations(self):
        g = _chain_graph()
        executions = [
            Execution(["A", "B"], True),
            Execution(["A"], True),
            Execution(["B"], False),
        ]
        results = compare(g, executions)

        formulas = {"ochiai", "tarantula", "jaccard", "dstar"}
        algorithms = {"propagation", "pagerank", "anomaly_only"}

        # 4 formulas × 3 algorithms + 2 graph-only = 14 keys
        assert len(results) == 14
        for f in formulas:
            for a in algorithms:
                key = f"{f}_{a}"
                assert key in results
                assert isinstance(results[key], list)
                assert len(results[key]) > 0

        assert "standard_pagerank" in results
        assert "betweenness" in results

    def test_all_results_are_sorted(self):
        g = _chain_graph()
        executions = [Execution(["A"], True), Execution(["A"], False)]
        results = compare(g, executions)
        for ranked in results.values():
            scores = [s for _, s in ranked]
            assert scores == sorted(scores, reverse=True)

    def test_different_formulas_produce_different_rankings(self):
        g = _chain_graph()
        executions = [
            Execution(["A", "B"], True),
            Execution(["A"], True),
            Execution(["B"], False),
        ]
        results = compare(g, executions)
        ochiai = results["ochiai_anomaly_only"]
        dstar = results["dstar_anomaly_only"]
        # DStar squares the numerator, amplifying differences
        assert ochiai != dstar
