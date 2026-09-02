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
    """Return explainable findings without silently changing a source event."""

    issues: list[QualityIssue] = []
    if event.confidence < 0.75:
        issues.append(
            QualityIssue(
                code="low_source_confidence",
                severity="warning",
                message="Source confidence is below the pilot threshold of 0.75.",
            )
        )
    issues.extend(
        QualityIssue(
            code=f"source_flag:{flag}",
            severity="warning",
            message=f"The source reported quality flag '{flag}'.",
        )
        for flag in event.quality_flags
        if flag != "verified_empty_reset"
    )
    if previous is None:
        return tuple(issues)
    if event.observed_at < previous.observed_at:
        issues.append(
            QualityIssue(
                code="out_of_order_event",
                severity="warning",
                message="Observation time is earlier than the latest event for this bus.",
            )
        )
    if event.alightings > previous.occupancy + event.boardings:
        issues.append(
            QualityIssue(
                code="impossible_negative_occupancy",
                severity="error",
                message="Alightings exceed passengers available before this event.",
            )
        )
    if "verified_empty_reset" not in event.quality_flags:
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
