"""Pure-Python numerics: confidence parsing, quantiles, histograms, rounding.

No numpy on purpose: the dependency list stays identical to the summarizer, and
a period holds at most a few million floats, which sorts in well under a second.
"""
import math
from bisect import bisect_right
from typing import Any, Sequence

from inspection_statistics.schemas.statistics import Quantiles

OK = "ok"
MISSING = "missing"      # None or an empty list: the model produced no confidence
ERROR = "error"          # a value that cannot be read as a finite number

QUANTILE_LEVELS: dict[str, float] = {
    "p05": 0.05, "p10": 0.10, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p90": 0.90, "p95": 0.95,
}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def as_float(value: Any) -> float | None:
    """A finite number becomes a float; anything else (None, bool, str, nan) becomes None."""
    return float(value) if _is_number(value) else None


def confidence_score(raw: Any) -> tuple[float | None, str]:
    """Reduce one prediction's confidence to a single float.

    ``max(confidence)`` when the field is a list, the value itself when it is a
    float. Returns ``(score, status)`` where status is OK, MISSING or ERROR.
    """
    if raw is None:
        return None, MISSING

    if _is_number(raw):
        return float(raw), OK

    if isinstance(raw, (list, tuple)):
        if not raw:
            return None, MISSING
        if all(_is_number(item) for item in raw):
            return float(max(raw)), OK
        return None, ERROR

    return None, ERROR


def rounded(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def quantile(ordered: Sequence[float], level: float) -> float | None:
    """Linear interpolation between order statistics (numpy's default method). ``ordered`` must be sorted."""
    count = len(ordered)
    if count == 0:
        return None
    if count == 1:
        return float(ordered[0])

    position = level * (count - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


def histogram_bin(value: float, edges: Sequence[float]) -> int:
    """Index of the bin holding ``value``. The last bin is closed; out-of-range values clamp to the edge bins."""
    index = bisect_right(edges, value) - 1
    return max(0, min(index, len(edges) - 2))


def describe(values: Sequence[float], digits: int) -> dict:
    """count / sum / sumSq / mean / std / min / max / quantiles of a sequence of floats.

    ``std`` is the population standard deviation, which is what ``sum``, ``sumSq``
    and ``count`` reproduce after combining several documents.
    """
    count = len(values)
    if count == 0:
        return {
            "count": 0, "sum": 0.0, "sum_sq": 0.0,
            "mean": None, "std": None, "min": None, "max": None,
            "quantiles": Quantiles(),
        }

    ordered = sorted(values)
    total = math.fsum(ordered)
    total_sq = math.fsum(value * value for value in ordered)
    mean = total / count
    variance = max(math.fsum((value - mean) ** 2 for value in ordered) / count, 0.0)

    return {
        "count": count,
        "sum": rounded(total, digits),
        "sum_sq": rounded(total_sq, digits),
        "mean": rounded(mean, digits),
        "std": rounded(math.sqrt(variance), digits),
        "min": rounded(ordered[0], digits),
        "max": rounded(ordered[-1], digits),
        "quantiles": Quantiles(**{
            name: rounded(quantile(ordered, level), digits) for name, level in QUANTILE_LEVELS.items()
        }),
    }
