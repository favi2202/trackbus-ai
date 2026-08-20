"""Canonical vendor-neutral passenger-count events."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4


class EventSource(StrEnum):
    """Approved source families; vendor-specific names belong in adapters."""

    APC = "apc"
    VISION = "vision"
    PAYMENT = "payment"
    MANUAL = "manual"
    IMPORT = "import"


@dataclass(frozen=True)
class PassengerCountEvent:
    event_id: str
    observed_at: str
    source: EventSource
    bus_id: str
    route_id: str
    stop_id: str
    door_id: str
    boardings: int
    alightings: int
    occupancy: int
    capacity: int
    confidence: float
    quality_flags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for name in (
            "event_id",
            "observed_at",
            "bus_id",
            "route_id",
            "stop_id",
            "door_id",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} cannot be empty")
        if min(self.boardings, self.alightings, self.occupancy) < 0:
            raise ValueError("counts and occupancy cannot be negative")
        if self.capacity <= 0 or self.occupancy > self.capacity:
            raise ValueError("occupancy must fit a positive capacity")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        try:
            parsed = datetime.fromisoformat(self.observed_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("observed_at must be an ISO-8601 timestamp") from exc
        if parsed.tzinfo is None:
            raise ValueError("observed_at must include a timezone")

    def to_payload(self) -> dict[str, Any]:
        """Return the versioned camelCase HTTP contract."""

        raw = asdict(self)
        return {
            "schemaVersion": "1.0",
            "eventId": raw["event_id"],
            "observedAt": raw["observed_at"],
            "source": self.source.value,
            "busId": raw["bus_id"],
            "routeId": raw["route_id"],
            "stopId": raw["stop_id"],
            "doorId": raw["door_id"],
            "boardings": raw["boardings"],
            "alightings": raw["alightings"],
            "occupancy": raw["occupancy"],
            "capacity": raw["capacity"],
            "confidence": raw["confidence"],
            "qualityFlags": list(self.quality_flags),
        }


def vision_event(
    *,
    direction: str,
    occupancy: int,
    capacity: int,
    confidence: float,
    bus_id: str,
    route_id: str,
    stop_id: str,
    door_id: str,
    quality_flags: tuple[str, ...] = (),
    observed_at: datetime | None = None,
) -> PassengerCountEvent:
    """Build one canonical event from a confirmed anonymous doorway crossing."""

    normalized = direction.upper()
    if normalized not in {"IN", "OUT"}:
        raise ValueError("direction must be IN or OUT")
    timestamp = observed_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return PassengerCountEvent(
        event_id=f"vision-{uuid4()}",
        observed_at=timestamp.isoformat(),
        source=EventSource.VISION,
        bus_id=bus_id,
        route_id=route_id,
        stop_id=stop_id,
        door_id=door_id,
        boardings=1 if normalized == "IN" else 0,
        alightings=1 if normalized == "OUT" else 0,
        occupancy=occupancy,
        capacity=capacity,
        confidence=confidence,
        quality_flags=quality_flags,
    )
