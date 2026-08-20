from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ForecastInput:
    recent_occupancy: Sequence[int]
    capacity: int
    hour: int
    day_type: Literal["weekday", "weekend"] = "weekday"
    weather: Literal["clear", "rain"] = "clear"
    event_nearby: bool = False


@dataclass(frozen=True)
class ForecastResult:
    expected_occupancy: int
    expected_percent: int
    lower_bound: int
    upper_bound: int
    confidence: float


def forecast_occupancy(data: ForecastInput) -> ForecastResult:
    """Transparent baseline; future ML models must outperform it in evaluation."""
    if data.capacity <= 0:
        raise ValueError("capacity must be positive")
    history = [max(0, int(value)) for value in data.recent_occupancy[-6:]]
    if not history:
        raise ValueError("recent_occupancy must contain at least one observation")

    weights = list(range(1, len(history) + 1))
    baseline = sum(value * weight for value, weight in zip(history, weights)) / sum(weights)
    rush_hour = data.day_type == "weekday" and (7 <= data.hour <= 9 or 17 <= data.hour <= 20)
    estimate = baseline * (1.13 if rush_hour else 1)
    estimate *= 1.07 if data.weather == "rain" else 1
    estimate *= 1.10 if data.event_nearby else 1
    expected = min(data.capacity, max(0, round(estimate)))
    spread = max(3, round(data.capacity * 0.08))

    return ForecastResult(
        expected_occupancy=expected,
        expected_percent=round(expected / data.capacity * 100),
        lower_bound=max(0, expected - spread),
        upper_bound=min(data.capacity, expected + spread),
        confidence=0.86 if len(history) >= 6 else 0.68,
    )
