from dataclasses import dataclass, field

import networkx as nx

from .ranker import RootCauseRanker
from .scorer import Execution, SpectrumScorer


@dataclass
class AnalysisResult:
    ranked: list[tuple[str, float]]
    formula_used: str
    algorithm_used: str
    component_scores: dict[str, float] = field(
        default_factory=dict
    )


def analyze(
    graph: nx.DiGraph,
    executions: list[Execution],
    formula: str = "ochiai",
    algorithm: str = "propagation",
    temporal_half_life: float | None = None,
) -> AnalysisResult:
    """Score and rank in a single call.

    Args:
        graph: Dependency graph (depends-on edges).
        executions: List of Execution objects.
        formula: SBFL formula name.
        algorithm: Ranking algorithm name.
        temporal_half_life: Seconds for temporal weighting, or None.
    """
    scorer = SpectrumScorer(
        formula=formula, temporal_half_life=temporal_half_life
    )
    scores = scorer.score(executions)

    ranker = RootCauseRanker()
    if algorithm == "propagation":
        ranked = ranker.rank_propagation(graph, scores)
    elif algorithm == "pagerank":
        ranked = ranker.rank_pagerank(graph, scores)
    elif algorithm == "anomaly_only":
        ranked = ranker.rank_anomaly_only(scores)
    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. "
            "Choose from: propagation, pagerank, anomaly_only"
        )

    return AnalysisResult(
        ranked=ranked,
        formula_used=formula,
        algorithm_used=algorithm,
        component_scores=scores,
    )


def iterative_analyze(
    graph: nx.DiGraph,
    executions: list[Execution],
    formula: str = "ochiai",
    algorithm: str = "propagation",
    max_candidates: int = 3,
    score_threshold: float = 0.0,
    temporal_half_life: float | None = None,
) -> list[AnalysisResult]:
    """Find root causes iteratively, removing each after discovery.

    After finding the top candidate:
    1. Record it with its score and sub-ranking.
    2. Remove all executions containing that component (they are
       "explained" by it).
    3. Remove the component from the graph.
    4. Re-run analysis on what remains.
    5. Stop when no failing executions remain, max_candidates is
       reached, or the top score falls below score_threshold.

    Returns:
        List of AnalysisResult, one per discovered root cause, in
        discovery order (most impactful first).
    """
    results: list[AnalysisResult] = []
    remaining_execs = list(executions)
    remaining_graph = graph.copy()

    for _ in range(max_candidates):
        if not any(ex.is_failing for ex in remaining_execs):
            break

        scorer = SpectrumScorer(
            formula=formula, temporal_half_life=temporal_half_life
        )
        scores = scorer.score(remaining_execs)
        if not scores:
            break

        ranker = RootCauseRanker()
        if algorithm == "propagation":
            ranked = ranker.rank_propagation(remaining_graph, scores)
        elif algorithm == "pagerank":
            ranked = ranker.rank_pagerank(remaining_graph, scores)
        elif algorithm == "anomaly_only":
            ranked = ranker.rank_anomaly_only(scores)
        else:
            raise ValueError(
                f"Unknown algorithm: {algorithm}. "
                "Choose from: propagation, pagerank, anomaly_only"
            )

        if not ranked:
            break

        top_name, top_score = ranked[0]
        if top_score < score_threshold:
            break

        results.append(
            AnalysisResult(
                ranked=ranked,
                formula_used=formula,
                algorithm_used=algorithm,
                component_scores=scores,
            )
        )

        remaining_execs = [
            ex for ex in remaining_execs if top_name not in ex.components
        ]

        if top_name in remaining_graph:
            remaining_graph.remove_node(top_name)

    return results


def compare(
    graph: nx.DiGraph,
    executions: list[Execution],
    temporal_half_life: float | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """Run all formula × ranking-algorithm combinations.

    Returns:
        Dict keyed by "{formula}_{algorithm}" with ranked list values.
    """
    formulas = ["ochiai", "tarantula", "jaccard", "dstar"]
    algorithms = ["propagation", "pagerank", "anomaly_only"]

    results: dict[str, list[tuple[str, float]]] = {}
    ranker = RootCauseRanker()

    for formula in formulas:
        scorer = SpectrumScorer(
            formula=formula, temporal_half_life=temporal_half_life
        )
        scores = scorer.score(executions)

        for algo in algorithms:
            key = f"{formula}_{algo}"
            if algo == "propagation":
                results[key] = ranker.rank_propagation(graph, scores)
            elif algo == "pagerank":
                results[key] = ranker.rank_pagerank(graph, scores)
            elif algo == "anomaly_only":
                results[key] = ranker.rank_anomaly_only(scores)

    results["standard_pagerank"] = ranker.rank_standard_pagerank(graph)
    results["betweenness"] = ranker.rank_betweenness(graph)

    return results
