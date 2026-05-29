"""CLI integration tests using temp files and subprocess."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _write_json(path: Path, data):
    path.write_text(json.dumps(data))


_EXECUTIONS = {
    "executions": [
        {"components": ["A", "B"], "is_failing": True, "timestamp": 0.0},
        {"components": ["A", "C"], "is_failing": True, "timestamp": 10.0},
        {"components": ["A", "B"], "is_failing": True, "timestamp": 20.0},
        {"components": ["D", "E"], "is_failing": True, "timestamp": 100.0},
        {"components": ["D"], "is_failing": True, "timestamp": 110.0},
        {"components": ["A", "B"], "is_failing": False, "timestamp": 200.0},
        {"components": ["D", "E"], "is_failing": False, "timestamp": 200.0},
    ]
}

_GRAPH = {
    "nodes": ["A", "B", "C", "D", "E"],
    "edges": [
        {"source": "B", "target": "A", "weight": 10},
        {"source": "C", "target": "A", "weight": 10},
        {"source": "A", "target": "C", "weight": 5},
        {"source": "E", "target": "D", "weight": 10},
    ],
}


def _run_cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "rootrank.cli", *args],
        capture_output=True,
        text=True,
    )


class TestScoreCommand:

    def test_outputs_scores(self, tmp_path: Path):
        exec_path = tmp_path / "executions.json"
        _write_json(exec_path, _EXECUTIONS)

        result = _run_cli("score", "--input", str(exec_path))
        assert result.returncode == 0
        assert "A" in result.stdout
        assert "D" in result.stdout

    def test_formula_flag(self, tmp_path: Path):
        exec_path = tmp_path / "executions.json"
        _write_json(exec_path, _EXECUTIONS)

        r1 = _run_cli("score", "--input", str(exec_path), "--formula", "ochiai")
        r2 = _run_cli("score", "--input", str(exec_path), "--formula", "dstar")
        assert r1.returncode == 0
        assert r2.returncode == 0
        # DStar amplifies differences — output differs from Ochiai
        # At minimum, both succeed and contain component names
        assert r1.stdout != r2.stdout

    def test_output_to_file(self, tmp_path: Path):
        exec_path = tmp_path / "executions.json"
        out_path = tmp_path / "scores.json"
        _write_json(exec_path, _EXECUTIONS)

        result = _run_cli(
            "score", "--input", str(exec_path), "--output", str(out_path)
        )
        assert result.returncode == 0
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert isinstance(data, dict)
        assert "A" in data

    def test_temporal_half_life(self, tmp_path: Path):
        exec_path = tmp_path / "executions.json"
        _write_json(exec_path, _EXECUTIONS)

        r1 = _run_cli("score", "--input", str(exec_path))
        r2 = _run_cli(
            "score", "--input", str(exec_path), "--temporal-half-life", "60"
        )
        assert r1.returncode == 0
        assert r2.returncode == 0
        # Temporal scoring differs from flat
        assert r1.stdout != r2.stdout


class TestRankCommand:

    def test_propagation(self, tmp_path: Path):
        graph_path = tmp_path / "graph.json"
        scores_path = tmp_path / "scores.json"
        _write_json(graph_path, _GRAPH)
        _write_json(scores_path, {"A": 0.9, "B": 0.5, "C": 0.2, "D": 0.8, "E": 0.6})

        result = _run_cli(
            "rank", "--graph", str(graph_path), "--scores", str(scores_path)
        )
        assert result.returncode == 0
        assert "A" in result.stdout

    def test_pagerank(self, tmp_path: Path):
        graph_path = tmp_path / "graph.json"
        scores_path = tmp_path / "scores.json"
        _write_json(graph_path, _GRAPH)
        _write_json(scores_path, {"A": 0.9, "B": 0.5, "C": 0.2, "D": 0.8, "E": 0.6})

        result = _run_cli(
            "rank", "--graph", str(graph_path), "--scores", str(scores_path),
            "--algorithm", "pagerank", "--alpha", "0.30"
        )
        assert result.returncode == 0

    def test_calls_semantics(self, tmp_path: Path):
        graph_path = tmp_path / "graph.json"
        scores_path = tmp_path / "scores.json"
        _write_json(graph_path, _GRAPH)
        _write_json(scores_path, {"A": 0.9, "B": 0.5, "C": 0.2, "D": 0.8, "E": 0.6})

        r1 = _run_cli(
            "rank", "--graph", str(graph_path), "--scores", str(scores_path),
            "--algorithm", "pagerank", "--edge-semantics", "calls"
        )
        r2 = _run_cli(
            "rank", "--graph", str(graph_path), "--scores", str(scores_path),
            "--algorithm", "pagerank", "--edge-semantics", "depends-on"
        )
        assert r1.returncode == 0
        assert r2.returncode == 0
        assert r1.stdout != r2.stdout

    def test_output_to_file(self, tmp_path: Path):
        graph_path = tmp_path / "graph.json"
        scores_path = tmp_path / "scores.json"
        out_path = tmp_path / "ranked.json"
        _write_json(graph_path, _GRAPH)
        _write_json(scores_path, {"A": 0.9, "B": 0.5})

        result = _run_cli(
            "rank", "--graph", str(graph_path), "--scores", str(scores_path),
            "--output", str(out_path)
        )
        assert result.returncode == 0
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert isinstance(data, list)
        assert data[0]["component"] == "A"


class TestAnalyzeCommand:

    def test_single_pass(self, tmp_path: Path):
        graph_path = tmp_path / "graph.json"
        exec_path = tmp_path / "executions.json"
        _write_json(graph_path, _GRAPH)
        _write_json(exec_path, _EXECUTIONS)

        result = _run_cli(
            "analyze", "--graph", str(graph_path),
            "--executions", str(exec_path)
        )
        assert result.returncode == 0
        assert "A" in result.stdout or "D" in result.stdout

    def test_iterative_mode(self, tmp_path: Path):
        graph_path = tmp_path / "graph.json"
        exec_path = tmp_path / "executions.json"
        _write_json(graph_path, _GRAPH)
        _write_json(exec_path, _EXECUTIONS)

        result = _run_cli(
            "analyze", "--graph", str(graph_path),
            "--executions", str(exec_path), "--iterative"
        )
        assert result.returncode == 0
        assert "Root Cause 1" in result.stdout

    def test_iterative_json_output(self, tmp_path: Path):
        graph_path = tmp_path / "graph.json"
        exec_path = tmp_path / "executions.json"
        out_path = tmp_path / "results.json"
        _write_json(graph_path, _GRAPH)
        _write_json(exec_path, _EXECUTIONS)

        result = _run_cli(
            "analyze", "--graph", str(graph_path),
            "--executions", str(exec_path), "--iterative",
            "--output", str(out_path)
        )
        assert result.returncode == 0
        data = json.loads(out_path.read_text())
        assert isinstance(data, list)
        assert len(data) >= 1
        assert "ranked" in data[0]

    def test_output_to_file(self, tmp_path: Path):
        graph_path = tmp_path / "graph.json"
        exec_path = tmp_path / "executions.json"
        out_path = tmp_path / "results.json"
        _write_json(graph_path, _GRAPH)
        _write_json(exec_path, _EXECUTIONS)

        result = _run_cli(
            "analyze", "--graph", str(graph_path),
            "--executions", str(exec_path), "--output", str(out_path)
        )
        assert result.returncode == 0
        data = json.loads(out_path.read_text())
        assert "ranked" in data


class TestCompareCommand:

    def test_all_combinations(self, tmp_path: Path):
        graph_path = tmp_path / "graph.json"
        exec_path = tmp_path / "executions.json"
        _write_json(graph_path, _GRAPH)
        _write_json(exec_path, _EXECUTIONS)

        result = _run_cli(
            "compare", "--graph", str(graph_path),
            "--executions", str(exec_path)
        )
        assert result.returncode == 0
        assert "ochiai_propagation" in result.stdout
        assert "dstar_pagerank" in result.stdout

    def test_output_to_file(self, tmp_path: Path):
        graph_path = tmp_path / "graph.json"
        exec_path = tmp_path / "executions.json"
        out_path = tmp_path / "results.json"
        _write_json(graph_path, _GRAPH)
        _write_json(exec_path, _EXECUTIONS)

        result = _run_cli(
            "compare", "--graph", str(graph_path),
            "--executions", str(exec_path), "--output", str(out_path)
        )
        assert result.returncode == 0
        data = json.loads(out_path.read_text())
        assert "ochiai_propagation" in data
        assert "standard_pagerank" in data
        assert len(data) == 14


class TestEdgeCases:

    def test_empty_executions(self, tmp_path: Path):
        exec_path = tmp_path / "empty.json"
        _write_json(exec_path, {"executions": []})

        result = _run_cli("score", "--input", str(exec_path))
        assert result.returncode == 0

    def test_missing_file(self):
        result = _run_cli("score", "--input", "/nonexistent/file.json")
        assert result.returncode != 0
