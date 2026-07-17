from dataclasses import dataclass
from typing import Literal

from .domain import PassengerCountEvent


@dataclass(frozen=True)
class QualityIssue:
    code: str
    severity: Literal["warning", "error"]
    message: str


def assess_event(
    event: PassengerCountEvent,
    previous: PassengerCountEvent | None = None,
) -> tuple[QualityIssue, ...]:
    """Return explainable data-quality findings without changing the source event."""
    issues: list[QualityIssue] = []

    if event.quality_score < 0.75:
        issues.append(
            QualityIssue(
                code="low_sensor_confidence",
                severity="warning",
                message="Sensor quality score is below the pilot threshold of 0.75.",
            )
        )

    if previous is None:
        return tuple(issues)

    if event.observed_at < previous.observed_at:
        issues.append(
            QualityIssue(
                code="out_of_order_event",
                severity="warning",
                message="Observation time is earlier than the last accepted event for this bus.",
            )
        )

    expected = previous.occupancy + event.boardings - event.alightings
    if abs(expected - event.occupancy) > 3:
        issues.append(
            QualityIssue(
                code="occupancy_delta_mismatch",
                severity="warning",
                message=(
                    "Reported occupancy differs from the boarding/alighting balance "
                    "by more than three passengers."
                ),
            )
        )

    return tuple(issues)
