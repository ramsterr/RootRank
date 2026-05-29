# RootRank

> **Spectrum-based root cause analysis for any graph with pass/fail traces.**

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

RootRank answers a single question: **"Which node in my dependency graph is the root cause of these failures?"** Drop in a JSON blob of pass/fail execution traces and your dependency graph — get back a ranked list of suspects.

## Use Cases

| Domain | What you give it | What you get |
|--------|-----------------|-------------|
| **Microservices** | Jaeger traces + dependency graph | Which service started the outage |
| **CI/CD** | Flaky test results + test-dependency graph | Which test/module is the real culprit |
| **Supply chain** | Shipment pass/fail logs + logistic graph | Which node is the bottleneck |
| **Circuit boards** | Pin test results + board topology | Which component caused the failure |
| **Any SBFL domain** | Any labeled traces + any directed graph | Ranked root cause suspects |

## Quickstart

```bash
pip install rootrank
```

```bash
# Score services by suspiciousness
RootRank score --input executions.json --formula ochiai

# Rank using graph structure
RootRank rank --graph graph.json --scores scores.json --algorithm propagation

# One-line analysis (score + rank)
RootRank analyze --graph graph.json --executions executions.json

# Find multiple independent root causes
RootRank analyze --graph graph.json --executions executions.json --iterative

# Compare all 14 formula × algorithm combinations
RootRank compare --graph graph.json --executions executions.json --output results.json
```

### Python API

```python
from rootrank import analyze, iterative_analyze, compare, SpectrumScorer, RootCauseRanker

# One-liner
result = analyze(graph, executions, formula="ochiai", algorithm="propagation")
print(result.ranked[:3])  # top 3 suspects

# Temporal weighting: earlier failures matter more
result = analyze(graph, executions, temporal_half_life=120)

# Multi-root-cause discovery
causes = iterative_analyze(graph, executions, max_candidates=5)
for i, cause in enumerate(causes):
    print(f"Cause #{i+1}: {cause.ranked[0]}")

# Compare everything
all_results = compare(graph, executions)
```

## How It Works

RootRank works in two stages, like a detective and a judge working together:

```
   Executions (JSON)           Dependency Graph (JSON)
          │                            │
          ▼                            │
   ┌──────────────┐                    │
   │ Stage 1: SBFL │  Ochiai /         │
   │  Spectrum     │  Tarantula /      │
   │  Scoring      │  Jaccard /        │
   │               │  DStar            │
   └──────┬───────┘                    │
          │ Anomaly scores             │
          ▼                            ▼
   ┌──────────────────────────────────────┐
   │  Stage 2: Graph-Based Ranking        │
   │                                      │
   │  Propagation ← upstream anomaly flow │
   │  PageRank    ← personalized by Ochiai│
   │  Betweenness ← structural bridges    │
   │  AnomalyOnly ← no-graph baseline     │
   └──────────────┬───────────────────────┘
                  │
                  ▼
          ┌──────────────┐
          │ Ranked List   │
          │ (sorted)      │
          └──────────────┘
```

### Stage 1 — Spectrum Scoring (SBFL)

Counts how often each service appears in failing vs. passing traces using the **Ochiai coefficient** (and three alternatives):

$$Ochiai(s) = \frac{n_{ef}}{\sqrt{n_f \cdot n_e}}$$

Where:
- $n_{ef}$ = failing executions containing service $s$
- $n_f$ = total failing executions
- $n_e$ = total executions containing $s$

A service in 95% of failures but only 5% of successes gets a high score (~0.87). A service in every trace (like the frontend) gets penalized (~0.45).

**Also supported:** Tarantula, Jaccard, DStar (λ=2). The `compare` command runs all four to find the best fit for your data.

### Stage 2 — Graph Ranking

Spectrum scores alone can mislead: a downstream victim of the root cause also appears in failing traces and gets a similar score. Graph ranking fixes this using the dependency structure.

RootRank uses **depends-on** edge semantics: `B → A` means *B depends on A*. Root causes (shared dependencies) naturally have high in-degree.

**Five ranking algorithms:**

| Algorithm | How it works | Best for |
|-----------|-------------|---------|
| **Propagation** (default) | Anomaly scores cascade upstream in one topological pass. If B depends on A and B fails, A gets boosted. | DAGs, simple interpretability |
| **Personalized PageRank** | PageRank with Ochiai scores as the teleportation vector. Low α (0.30) ensures anomaly signal dominates graph structure. | Dense cyclic graphs |
| **Standard PageRank** | Uniform PageRank — graph-only structural baseline. | Sanity check |
| **Betweenness Centrality** | How often a node lies on shortest paths between others. Edge weights inverted (higher call frequency → shorter distance). | Finding structural bridges |
| **Anomaly-Only** | Raw spectrum scores — no-graph baseline. | Sanity check |

### Temporal Weighting

If your executions carry timestamps, RootRank weights earlier failures higher — capturing anomaly-onset ordering:

```python
scorer = SpectrumScorer(formula="ochiai", temporal_half_life=120)
```

A service that fails at t=0 gets full weight; the same service failing at t=120s gets half weight. This naturally surfaces the service that *started* the cascade.

### Iterative Root Cause Removal

When multiple independent failures occur simultaneously (e.g., database down AND payment-gateway outage), a single pass may find only the strongest signal. Iterative mode peels back each layer:

```python
causes = iterative_analyze(graph, executions, max_candidates=5)
# Finds A, then removes all traces containing A, then finds D, etc.
```

## Input Formats

### Executions

```json
{
  "executions": [
    {"components": ["frontend", "cart-service", "database"], "is_failing": true, "timestamp": 1000.0},
    {"components": ["frontend", "cart-service"], "is_failing": false, "timestamp": 1005.0}
  ]
}
```

Timestamps are optional. When provided with `--temporal-half-life`, earlier failures receive exponentially higher weight.

### Dependency Graph

```json
{
  "nodes": ["frontend", "cart-service", "database", "payment-service"],
  "edges": [
    {"source": "frontend", "target": "cart-service", "weight": 95},
    {"source": "cart-service", "target": "database", "weight": 85},
    {"source": "frontend", "target": "payment-service", "weight": 60}
  ]
}
```

Edge semantics: `source → target` means *source depends on target*. This is the same direction as "source calls target" in microservice architectures. Set `--edge-semantics calls` if your graph uses the reverse convention and RootRank will auto-correct.

## API Reference

[`SpectrumScorer`](rootrank/scorer.py) — SBFL scoring with 4 formulas + temporal weighting  
[`RootCauseRanker`](rootrank/ranker.py) — 5 graph ranking algorithms  
[`analyze()`](rootrank/engine.py) — one-call scoring + ranking  
[`iterative_analyze()`](rootrank/engine.py) — multi-root-cause discovery  
[`compare()`](rootrank/engine.py) — all 14 formula × algorithm combinations

## Design Principles

- **Zero runtime dependencies beyond NetworkX** — works anywhere Python runs
- **Single-pass O(E) scoring** — not O(C×E)
- **All division-by-zero paths** guarded
- **Weights inverted for betweenness** — NetworkX treats weights as distances by default (a common footgun)
- **Low PageRank α (0.30)** — standard 0.85 drowns the anomaly signal in graph structure
- **Depends-on edge semantics by default** — prevents the "victim amplification" bug

## Install from Source

```bash
git clone https://github.com/ramsterr/RootRank.git
cd RootRank
pip install -e ".[dev]"
pytest          # 68 tests
```

## License

MIT
