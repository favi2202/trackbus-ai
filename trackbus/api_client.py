"""Retry-safe HTTP delivery for canonical TrackBus edge events."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from trackbus.event_contract import EventSource, PassengerCountEvent

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeliveryResult:
    delivered: bool
    queued: bool
    status_code: int | None = None
    error: str | None = None


class TrackBusApiClient:
    def __init__(
        self, base_url: str | None, queue_directory: str | Path, *, timeout: float = 3.0
    ) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.queue_directory = Path(queue_directory)
        self.timeout = timeout

    @property
    def queued_count(self) -> int:
        if not self.queue_directory.exists():
            return 0
        return sum(1 for _ in self.queue_directory.glob("*.json"))

    def send(self, event: PassengerCountEvent) -> DeliveryResult:
        if not self.base_url:
            self._queue(event)
            return DeliveryResult(
                delivered=False, queued=True, error="API URL not configured"
            )
        try:
            status = self._post(event.to_payload())
        except (OSError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            self._queue(event)
            return DeliveryResult(delivered=False, queued=True, error=str(exc))
        if 200 <= status < 300:
            return DeliveryResult(delivered=True, queued=False, status_code=status)
        self._queue(event)
        return DeliveryResult(
            delivered=False, queued=True, status_code=status, error=f"HTTP {status}"
        )

    def flush(self) -> tuple[int, int]:
        if not self.base_url or not self.queue_directory.exists():
            return 0, self.queued_count
        delivered = 0
        for path in sorted(self.queue_directory.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                status = self._post(payload)
                if not 200 <= status < 300:
                    break
                path.unlink()
                delivered += 1
            except (OSError, ValueError, urllib.error.URLError, TimeoutError):
                break
        return delivered, self.queued_count

    def _post(self, payload: dict[str, object]) -> int:
        request = urllib.request.Request(
            f"{self.base_url}/v1/events/passenger-counts",
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "user-agent": "TrackBus-Vision/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status
        except urllib.error.HTTPError as error:
            return error.code

    def _queue(self, event: PassengerCountEvent) -> None:
        self.queue_directory.mkdir(parents=True, exist_ok=True)
        destination = self.queue_directory / f"{event.event_id}.json"
        if destination.exists():
            return
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(event.to_payload(), indent=2), encoding="utf-8")
        temporary.replace(destination)


def event_from_payload(payload: dict[str, object]) -> PassengerCountEvent:
    """Validate a queued payload using the same canonical record."""

    return PassengerCountEvent(
        event_id=str(payload["eventId"]),
        observed_at=str(payload["observedAt"]),
        source=EventSource(str(payload["source"])),
        bus_id=str(payload["busId"]),
        route_id=str(payload["routeId"]),
        stop_id=str(payload["stopId"]),
        door_id=str(payload["doorId"]),
        boardings=int(payload["boardings"]),
        alightings=int(payload["alightings"]),
        occupancy=int(payload["occupancy"]),
        capacity=int(payload["capacity"]),
        confidence=float(payload["confidence"]),
        quality_flags=tuple(str(item) for item in payload.get("qualityFlags", [])),
    )
