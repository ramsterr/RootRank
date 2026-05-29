import pytest
from rootrank.scorer import Execution, SpectrumScorer


class TestOchiai:
    """Tests for the default Ochiai coefficient formula."""

    def test_walkthrough_example(self):
        """Plan Concept 1 walk-through with inventory-service and frontend.

        100 executions: 20 failing, 80 passing.
        inventory: 19 failing, 5 passing  -> Ochiai = 19/sqrt(20*24) = 0.8676
        frontend:  20 failing, 80 passing -> Ochiai = 20/sqrt(20*100) = 0.4472
        """
        executions = (
            [Execution(["inventory-service", "frontend"], True) for _ in range(19)]
            + [Execution(["frontend"], True)]
            + [Execution(["inventory-service", "frontend"], False) for _ in range(5)]
            + [Execution(["frontend"], False) for _ in range(75)]
        )
        scorer = SpectrumScorer("ochiai")
        scores = scorer.score(executions)
        assert round(scores["inventory-service"], 2) == 0.87
        assert round(scores["frontend"], 2) == 0.45

    def test_perfect_signal(self):
        """Component in ALL failing and NO passing -> Ochiai = 1.0."""
        executions = [
            Execution(["A", "B"], True),
            Execution(["A", "C"], True),
            Execution(["B", "C"], False),
        ]
        scorer = SpectrumScorer("ochiai")
        scores = scorer.score(executions)
        assert scores["A"] == 1.0

    def test_ubiquitous_component(self):
        """Component in ALL executions equally -> ~0.707 (no discriminating power).

        Ochiai = 10 / sqrt(10 * 20) = 10 / 14.14 = 0.707
        """
        executions = (
            [Execution(["A"], True) for _ in range(10)]
            + [Execution(["A"], False) for _ in range(10)]
        )
        scorer = SpectrumScorer("ochiai")
        scores = scorer.score(executions)
        assert round(scores["A"], 3) == 0.707

    def test_no_failing_executions(self):
        """When there are no failures, all components score 0.0."""
        executions = [Execution(["A", "B"], False) for _ in range(5)]
        scorer = SpectrumScorer("ochiai")
        scores = scorer.score(executions)
        assert scores == {"A": 0.0, "B": 0.0}

    def test_component_never_appeared(self):
        """Component that appears in zero executions -> 0.0."""
        executions = [Execution(["A"], True) for _ in range(3)]
        scorer = SpectrumScorer("ochiai")
        scores = scorer.score(executions)
        assert "B" not in scores  # B never seen, should not appear


class TestFormulas:
    """Tests covering Tarantula, Jaccard, and DStar formulas."""

    @staticmethod
    def _make_execs(n_ef, n_ep, n_f_minus_ef=0):
        """Build execution list from counts."""
        execs = []
        for _ in range(n_ef):
            execs.append(Execution(["X"], True))
        for _ in range(n_ep):
            execs.append(Execution(["X"], False))
        execs.append(Execution(["Y"], True))
        return execs

    def test_tarantula(self):
        """Tarantula = (ef/n_f) / (ef/n_f + ep/n_p)."""
        scorer = SpectrumScorer("tarantula")
        scores = scorer.score(self._make_execs(9, 1))
        # n_f=10 (9 X + 1 Y), n_p=1 (just the 1 X)
        # ef=9, ep=1, fail_ratio=0.9, pass_ratio=1.0, denom=1.9
        # tarantula = 0.9 / 1.9 = 0.4737
        assert round(scores["X"], 3) == 0.474

    def test_jaccard(self):
        """Jaccard = ef / (n_f + ep)."""
        scorer = SpectrumScorer("jaccard")
        scores = scorer.score(self._make_execs(9, 1))
        # n_f=10, ef=9, ep=1 -> 9/(10+1) = 0.818
        assert round(scores["X"], 3) == 0.818

    def test_dstar(self):
        """DStar(lambda=2) = ef^2 / (ep + n_nf)."""
        scorer = SpectrumScorer("dstar")
        scores = scorer.score(self._make_execs(9, 1))
        # n_f=10, ef=9, ep=1, n_nf=1 -> 81/2 = 40.5
        assert scores["X"] == 40.5

    def test_dstar_denom_zero(self):
        """Component in ALL failing, NO passing -> ef^2 (max suspicion)."""
        executions = (
            [Execution(["X"], True) for _ in range(5)]
            + [Execution(["Y"], True)]
        )
        scorer = SpectrumScorer("dstar")
        scores = scorer.score(executions)
        # n_f=6, ef=5, ep=0, n_nf=1 -> 25/1 = 25
        assert scores["X"] == 25.0

    def test_dstar_denom_strictly_zero(self):
        """Component in ALL failing, NO passing, NO other failures -> ef^2."""
        executions = [Execution(["X"], True) for _ in range(5)]
        scorer = SpectrumScorer("dstar")
        scores = scorer.score(executions)
        # n_f=5, ef=5, ep=0, n_nf=0 -> denom = 0, fallback = 25.0
        assert scores["X"] == 25.0

    def test_tarantula_denom_zero(self):
        """Component in 0 failing, 0 passing -> 0.0."""
        executions = [Execution(["A"], True) for _ in range(3)]
        scorer = SpectrumScorer("tarantula")
        scores = scorer.score(executions)
        # "B" never appeared -> ef=0, ep=0 -> fail_ratio=0, pass_ratio=0 -> 0
        assert "B" not in scores


class TestTemporal:
    """Tests for temporal weighting mode."""

    def test_uniform_without_timestamps(self):
        """Without timestamps or half_life, all weights are 1.0."""
        executions = [
            Execution(["A"], True),
            Execution(["A"], True),
            Execution(["A"], False),
        ]
        scorer = SpectrumScorer("ochiai", temporal_half_life=60)
        scores = scorer.score(executions)
        # Falls back to uniform since no timestamps
        assert round(scores["A"], 2) == round(2 / (2 * 3) ** 0.5, 2)

    def test_earlier_failures_weighted_higher(self):
        """Earlier failing traces contribute more to ef."""
        executions = [
            Execution(["early"], True, timestamp=0.0),
            Execution(["late"], True, timestamp=120.0),
            Execution(["early"], False, timestamp=200.0),
            Execution(["late"], False, timestamp=200.0),
        ]
        scorer_no_t = SpectrumScorer("ochiai")
        scorer_t = SpectrumScorer("ochiai", temporal_half_life=60)

        scores_no_t = scorer_no_t.score(executions)
        scores_t = scorer_t.score(executions)

        # Without temporal: both have same ef/ep, should be equal
        assert round(scores_no_t["early"], 2) == round(scores_no_t["late"], 2)

        # With temporal: early wins because its failure is at t=0 vs t=120
        assert scores_t["early"] > scores_t["late"]

    def test_all_same_timestamps(self):
        """All same timestamps -> all weights 1.0, identical to non-temporal."""
        executions = [
            Execution(["A"], True, timestamp=5.0),
            Execution(["A"], True, timestamp=5.0),
            Execution(["A"], False, timestamp=5.0),
        ]
        scorer_t = SpectrumScorer("ochiai", temporal_half_life=60)
        scorer = SpectrumScorer("ochiai")

        assert scorer_t.score(executions) == scorer.score(executions)


class TestSorting:
    """Tests for output ordering."""

    def test_highest_first(self):
        """Scores are returned in descending order."""
        executions = []
        for _ in range(8):
            executions.append(Execution(["high"], True))
        for _ in range(2):
            executions.append(Execution(["high"], False))
        for _ in range(3):
            executions.append(Execution(["low"], True))
        for _ in range(7):
            executions.append(Execution(["low"], False))

        scorer = SpectrumScorer("ochiai")
        scores = scorer.score(executions)
        items = list(scores.items())
        assert items[0][1] >= items[1][1]
        assert items[0][0] == "high"


class TestValidation:
    """Tests for input validation."""

    def test_unknown_formula_raises(self):
        with pytest.raises(ValueError, match="Unknown formula"):
            SpectrumScorer("nonexistent")

    def test_valid_formulas_accepted(self):
        for f in ("ochiai", "tarantula", "jaccard", "dstar"):
            SpectrumScorer(f)  # should not raise

    def test_empty_executions(self):
        """Empty execution list should not crash."""
        scores = SpectrumScorer("ochiai").score([])
        assert scores == {}

    def test_timestamp_is_optional(self):
        """Execution works without timestamp."""
        e = Execution(["A"], True)
        assert e.timestamp is None
        assert e.is_failing is True
        assert e.components == ["A"]
