from collections import defaultdict
from dataclasses import dataclass
from math import sqrt


@dataclass
class Execution:
    components: list[str]
    is_failing: bool
    timestamp: float | None = None


class SpectrumScorer:
    """Computes suspiciousness scores for components given execution data.

    Supports four spectrum-based fault localization (SBFL) formulas:
    Ochiai (default), Tarantula, Jaccard, DStar.

    Optional temporal weighting: if executions carry timestamps and
    temporal_half_life is set, earlier failures receive exponentially
    higher weight — capturing anomaly-onset ordering.
    """

    FORMULAS = frozenset({"ochiai", "tarantula", "jaccard", "dstar"})

    def __init__(self, formula: str = "ochiai",
                 temporal_half_life: float | None = None):
        if formula not in self.FORMULAS:
            raise ValueError(
                f"Unknown formula: {formula}. "
                f"Choose from {sorted(self.FORMULAS)}"
            )
        self.formula = formula
        self.temporal_half_life = temporal_half_life

    def score(self, executions: list[Execution]) -> dict[str, float]:

        n_ef = defaultdict(float)
        n_ep = defaultdict(float)
        components: set[str] = set()
        n_f = 0.0

        has_timestamps = all(e.timestamp is not None for e in executions)
        use_temporal = (
            self.temporal_half_life is not None
            and has_timestamps
            and len(executions) > 0
        )

        weights: dict[int, float] | None = None
        if use_temporal:
            t_first = min(e.timestamp for e in executions)  # type: ignore[type-var]
            weights = {}
            hl = self.temporal_half_life
            for ex in executions:
                dt = ex.timestamp - t_first  # type: ignore[operator]
                weights[id(ex)] = 1.0 if dt <= 0 else 0.5 ** (dt / hl)

        for ex in executions:
            w = weights[id(ex)] if weights is not None else 1.0
            components.update(ex.components)
            if ex.is_failing:
                n_f += w
                for comp in ex.components:
                    n_ef[comp] += w
            else:
                for comp in ex.components:
                    n_ep[comp] += w

        if n_f == 0:
            return dict.fromkeys(components, 0.0)

        n_p = (
            sum(weights[id(e)] for e in executions)
            if weights is not None
            else len(executions)
        ) - n_f

        scores: dict[str, float] = {}

        for comp in components:
            ef = n_ef[comp]
            ep = n_ep[comp]
            n_e = ef + ep

            if n_e == 0:
                scores[comp] = 0.0
                continue

            n_nf = n_f - ef

            if self.formula == "ochiai":
                denom = sqrt(n_f * n_e)
                scores[comp] = ef / denom if ef > 0 and denom > 0 else 0.0

            elif self.formula == "tarantula":
                fail_ratio = ef / n_f if n_f > 0 else 0.0
                pass_ratio = ep / n_p if n_p > 0 else 0.0
                denom = fail_ratio + pass_ratio
                scores[comp] = fail_ratio / denom if denom > 0 else 0.0

            elif self.formula == "jaccard":
                denom = n_f + ep
                scores[comp] = ef / denom if denom > 0 else 0.0

            elif self.formula == "dstar":
                denom = ep + n_nf
                scores[comp] = (
                    (ef ** 2) / denom if denom > 0 else float(ef ** 2)
                )

        return dict(
            sorted(scores.items(), key=lambda item: item[1], reverse=True)
        )
