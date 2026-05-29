"""JSON I/O and command-line interface for MicroRank."""

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import networkx as nx

from .engine import analyze, compare, iterative_analyze
from .scorer import Execution


def _load_executions(path: str | Path) -> list[Execution]:
    data = _read_json(path)
    execs = []
    for ex in data.get("executions", data if isinstance(data, list) else []):
        execs.append(
            Execution(
                components=ex["components"],
                is_failing=ex["is_failing"],
                timestamp=ex.get("timestamp"),
            )
        )
    return execs


def _load_graph(path: str | Path) -> nx.DiGraph:
    data = _read_json(path)
    g = nx.DiGraph()
    g.add_nodes_from(data.get("nodes", []))
    for edge in data.get("edges", []):
        g.add_edge(
            edge["source"],
            edge["target"],
            weight=edge.get("weight", 1.0),
        )
    return g


def _load_scores(path: str | Path) -> dict[str, float]:
    data = _read_json(path)
    if isinstance(data, dict):
        return {k: float(v) for k, v in data.items()}
    return {}


def _read_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _write_json(obj: Any, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _format_ranked(ranked: list[tuple[str, float]]) -> list[dict]:
    return [{"component": name, "score": round(score, 6)} for name, score in ranked]


def _print_ranked(ranked: list[tuple[str, float]]) -> None:
    print(f"{'Rank':<6}{'Component':<30}{'Score'}")
    print("-" * 50)
    for i, (name, score) in enumerate(ranked, 1):
        print(f"{i:<6}{name:<30}{score:.6f}")


def main(argv: list[str] | None = None) -> None:
    parser = ArgumentParser(
        prog="RootRank",
        description="Spectrum-based root cause analysis for any graph.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- score ----
    p_score = sub.add_parser("score", help="Compute suspiciousness scores")
    p_score.add_argument("--input", required=True, help="JSON executions file")
    p_score.add_argument("--formula", default="ochiai",
                         choices=["ochiai", "tarantula", "jaccard", "dstar"])
    p_score.add_argument("--temporal-half-life", type=float, default=None)
    p_score.add_argument("--output", default=None, help="JSON output file")

    # ---- rank ----
    p_rank = sub.add_parser("rank", help="Rank nodes using graph + anomaly scores")
    p_rank.add_argument("--graph", required=True, help="JSON graph file")
    p_rank.add_argument("--scores", required=True, help="JSON scores file")
    p_rank.add_argument("--algorithm", default="propagation",
                        choices=["propagation", "pagerank", "standard_pagerank",
                                 "betweenness", "anomaly_only"])
    p_rank.add_argument("--alpha", type=float, default=0.30)
    p_rank.add_argument("--edge-semantics", default="depends-on",
                        choices=["depends-on", "calls"])
    p_rank.add_argument("--output", default=None)

    # ---- analyze ----
    p_analyze = sub.add_parser("analyze", help="Score + rank in one step")
    p_analyze.add_argument("--graph", required=True)
    p_analyze.add_argument("--executions", required=True)
    p_analyze.add_argument("--formula", default="ochiai",
                           choices=["ochiai", "tarantula", "jaccard", "dstar"])
    p_analyze.add_argument("--algorithm", default="propagation",
                           choices=["propagation", "pagerank", "anomaly_only"])
    p_analyze.add_argument("--temporal-half-life", type=float, default=None)
    p_analyze.add_argument("--iterative", action="store_true")
    p_analyze.add_argument("--max-candidates", type=int, default=3)
    p_analyze.add_argument("--score-threshold", type=float, default=0.0)
    p_analyze.add_argument("--output", default=None)

    # ---- compare ----
    p_compare = sub.add_parser("compare",
                               help="Run all formula × algorithm combinations")
    p_compare.add_argument("--graph", required=True)
    p_compare.add_argument("--executions", required=True)
    p_compare.add_argument("--temporal-half-life", type=float, default=None)
    p_compare.add_argument("--output", default=None)

    args = parser.parse_args(argv)

    if args.command == "score":
        execs = _load_executions(args.input)
        from .scorer import SpectrumScorer
        scorer = SpectrumScorer(
            formula=args.formula, temporal_half_life=args.temporal_half_life
        )
        scores = scorer.score(execs)
        if args.output:
            _write_json(scores, args.output)
        else:
            _print_ranked(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    elif args.command == "rank":
        g = _load_graph(args.graph)
        scores = _load_scores(args.scores)
        from .ranker import RootCauseRanker
        ranker = RootCauseRanker()
        algo = args.algorithm
        if algo == "propagation":
            ranked = ranker.rank_propagation(g, scores)
        elif algo == "pagerank":
            ranked = ranker.rank_pagerank(
                g, scores, alpha=args.alpha,
                edge_semantics=args.edge_semantics
            )
        elif algo == "standard_pagerank":
            ranked = ranker.rank_standard_pagerank(g, alpha=args.alpha)
        elif algo == "betweenness":
            ranked = ranker.rank_betweenness(g)
        elif algo == "anomaly_only":
            ranked = ranker.rank_anomaly_only(scores)
        else:
            raise ValueError(f"Unknown algorithm: {algo}")
        if args.output:
            _write_json(_format_ranked(ranked), args.output)
        else:
            _print_ranked(ranked)

    elif args.command == "analyze":
        g = _load_graph(args.graph)
        execs = _load_executions(args.executions)
        if args.iterative:
            results = iterative_analyze(
                g, execs,
                formula=args.formula,
                algorithm=args.algorithm,
                max_candidates=args.max_candidates,
                score_threshold=args.score_threshold,
                temporal_half_life=args.temporal_half_life,
            )
            output_data = [
                {
                    "candidate": i + 1,
                    "ranked": _format_ranked(r.ranked),
                    "formula": r.formula_used,
                    "algorithm": r.algorithm_used,
                }
                for i, r in enumerate(results)
            ]
        else:
            result = analyze(
                g, execs,
                formula=args.formula,
                algorithm=args.algorithm,
                temporal_half_life=args.temporal_half_life,
            )
            output_data = {
                "ranked": _format_ranked(result.ranked),
                "formula": result.formula_used,
                "algorithm": result.algorithm_used,
            }
        if args.output:
            _write_json(output_data, args.output)
        else:
            if isinstance(output_data, list):
                for entry in output_data:
                    print(f"\n--- Root Cause {entry['candidate']} ---")
                    _print_ranked(
                        [(r["component"], r["score"]) for r in entry["ranked"]]
                    )
            else:
                _print_ranked(
                    [(r["component"], r["score"]) for r in output_data["ranked"]]
                )

    elif args.command == "compare":
        g = _load_graph(args.graph)
        execs = _load_executions(args.executions)
        results = compare(
            g, execs, temporal_half_life=args.temporal_half_life
        )
        output_data = {
            key: _format_ranked(ranked) for key, ranked in results.items()
        }
        if args.output:
            _write_json(output_data, args.output)
        else:
            for key, ranked in sorted(results.items()):
                print(f"\n=== {key} ===")
                _print_ranked(ranked)


if __name__ == "__main__":
    main()
