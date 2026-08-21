import json
from datetime import UTC, datetime
from io import BytesIO

from trackbus.api_client import TrackBusApiClient
from trackbus.event_contract import vision_event


def event():
    return vision_event(
        direction="IN",
        occupancy=1,
        capacity=40,
        confidence=0.9,
        bus_id="BUS",
        route_id="22",
        stop_id="STOP",
        door_id="DOOR",
        observed_at=datetime(2026, 8, 20, tzinfo=UTC),
    )


def test_missing_api_url_queues_event(tmp_path) -> None:
    client = TrackBusApiClient(None, tmp_path)
    item = event()
    result = client.send(item)
    assert result.queued and not result.delivered
    assert client.queued_count == 1


def test_successful_delivery_does_not_queue(tmp_path, monkeypatch) -> None:
    client = TrackBusApiClient("http://trackbus.test", tmp_path)
    monkeypatch.setattr(client, "_post", lambda _payload: 202)
    result = client.send(event())
    assert result.delivered and not result.queued
    assert client.queued_count == 0


def test_failed_http_status_queues_same_event_id(tmp_path, monkeypatch) -> None:
    client = TrackBusApiClient("http://trackbus.test", tmp_path)
    monkeypatch.setattr(client, "_post", lambda _payload: 503)
    item = event()
    client.send(item)
    client.send(item)
    queued = list(tmp_path.glob("*.json"))
    assert len(queued) == 1
    assert json.loads(queued[0].read_text())["eventId"] == item.event_id


def test_flush_delivers_and_removes_queue(tmp_path, monkeypatch) -> None:
    offline = TrackBusApiClient(None, tmp_path)
    offline.send(event())
    online = TrackBusApiClient("http://trackbus.test", tmp_path)
    monkeypatch.setattr(online, "_post", lambda _payload: 202)
    delivered, remaining = online.flush()
    assert (delivered, remaining) == (1, 0)


def test_api_key_is_sent_as_bearer_token(tmp_path, monkeypatch) -> None:
    captured = {}

    class Response:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def open_request(request, timeout):
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        captured["payload"] = json.load(BytesIO(request.data))
        return Response()

    client = TrackBusApiClient(
        "https://trackbus.test/api", tmp_path, api_key="camera-secret", timeout=4.0
    )
    monkeypatch.setattr("urllib.request.urlopen", open_request)

    item = event()
    result = client.send(item)

    assert result.delivered
    assert captured["authorization"] == "Bearer camera-secret"
    assert captured["timeout"] == 4.0
    assert captured["payload"]["eventId"] == item.event_id
