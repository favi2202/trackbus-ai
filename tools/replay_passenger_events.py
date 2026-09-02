#!/usr/bin/env python3
"""Replay a JSON event file against TrackBus. Dry-run is the safe default."""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        default="data/sample/passenger-count-events.json",
        help="JSON file containing an array of passenger-count events",
    )
    parser.add_argument("--target", default="http://localhost:8000")
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument(
        "--send",
        action="store_true",
        help=(
            "Actually send events. Without this flag the command only validates "
            "and prints."
        ),
    )
    return parser.parse_args()


def load_events(path: str) -> list[dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("input must be a JSON array")
    required = {
        "schemaVersion",
        "eventId",
        "observedAt",
        "source",
        "busId",
        "routeId",
        "stopId",
        "doorId",
        "boardings",
        "alightings",
        "occupancy",
        "capacity",
        "confidence",
        "qualityFlags",
    }
    for index, event in enumerate(payload):
        if not isinstance(event, dict) or not required.issubset(event):
            missing = required.difference(event if isinstance(event, dict) else {})
            raise ValueError(f"event {index} is invalid; missing: {sorted(missing)}")
    return payload


def send_event(target: str, event: dict[str, object]) -> tuple[int, str]:
    request = urllib.request.Request(
        f"{target.rstrip('/')}/v1/events/passenger-counts",
        data=json.dumps(event).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8")


def main() -> int:
    args = arguments()
    try:
        events = load_events(args.input)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 2

    mode = "SEND" if args.send else "DRY RUN"
    print(f"TrackBus replay · {mode} · {len(events)} events")
    for index, event in enumerate(events, start=1):
        label = (
            f"{event['eventId']} bus={event['busId']} occupancy={event['occupancy']}"
        )
        if not args.send:
            print(f"[{index}/{len(events)}] valid {label}")
            continue
        status_code, body = send_event(args.target, event)
        print(f"[{index}/{len(events)}] HTTP {status_code} {label} {body}")
        if status_code >= 400:
            return 1
        time.sleep(max(0, args.interval))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
