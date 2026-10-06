# CMN-C2-274 - Unit tests: SansanClient service (Sansan Open API v3 shape)
# Adapted from a sibling tool-calling golden suite (kaonavi_client -> sansan_client).
# Pure service layer (stdlib-only, no framework imports) - plain function tests.

import pytest

from src.services.sansan_client import SansanApiError, SansanClient


def test_find_record_success_with_injected_get():
    captured = {}

    def get(url, headers, body):
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = body
        return 200, {"data": [{"id": "sc-1001", "name": "Contact sc-1001"}]}

    client = SansanClient("https://sansan.example.test/v3/", get=get)
    resp = client.find_record("contact", "sc-1001", "key123")
    assert resp["data"][0]["id"] == "sc-1001"
    assert captured["url"] == "https://sansan.example.test/v3/contacts"
    # Sansan Open API auth: the per-call key travels in X-Sansan-Api-Key.
    assert captured["headers"]["X-Sansan-Api-Key"] == "key123"
    assert captured["headers"]["Content-Type"] == "application/json"
    assert captured["body"]["record_id"] == "sc-1001"


def test_find_record_deal_uses_deals_collection():
    captured = {}

    def get(url, headers, body):
        captured["url"] = url
        return 200, {"data": [{"id": "d-42", "name": "Deal d-42"}]}

    client = SansanClient("https://sansan.example.test/v3", get=get)
    client.find_record("deal", "d-42", "key")
    assert captured["url"] == "https://sansan.example.test/v3/deals"


def test_create_record_success_with_injected_post():
    captured = {}

    def post(url, headers, body):
        captured["url"] = url
        captured["body"] = body
        return 200, {"task_id": 42, "record_id": "sc-9002"}

    client = SansanClient("https://sansan.example.test/v3", post=post)
    payload = {"record_data": [{"id": "sc-9002", "name": "Alice"}]}
    resp = client.create_record("contact", payload, "key")
    assert resp["record_id"] == "sc-9002"
    assert captured["url"] == "https://sansan.example.test/v3/contacts"
    assert captured["body"] == payload


def test_update_record_success_with_injected_patch():
    captured = {}

    def patch(url, headers, body):
        captured["url"] = url
        captured["body"] = body
        return 200, {"task_id": 7, "record_id": "sc-1001"}

    client = SansanClient("https://sansan.example.test/v3", patch=patch)
    payload = {"record_data": [{"id": "sc-1001"}]}
    resp = client.update_record("contact", payload, "key")
    assert resp["record_id"] == "sc-1001"
    assert captured["body"] == payload


def test_non_2xx_raises_sansan_api_error():
    def post(url, headers, body):
        return 400, {"errors": ["record_data is malformed"]}

    client = SansanClient("https://sansan.example.test/v3", post=post)
    with pytest.raises(SansanApiError) as exc:
        client.create_record("contact", {"record_data": [{}]}, "key")
    assert exc.value.status_code == 400
    assert "record_data is malformed" in str(exc.value)


def test_default_stub_transport_lookup_shape():
    # No transport injected -> deterministic, network-free v1 stub.
    client = SansanClient()
    assert client.uses_stub_transport is True
    resp = client.find_record("contact", "sc-1001", "key")
    assert resp.get("_stub") is True
    record = resp["data"][0]
    assert record["id"] == "sc-1001"
    assert record["name"] == "Contact sc-1001"


def test_default_stub_transport_deal_label():
    client = SansanClient()
    resp = client.find_record("deal", "d-42", "key")
    assert resp.get("_stub") is True
    assert resp["data"][0]["name"] == "Deal d-42"


def test_default_stub_transport_create_echoes_record_id():
    client = SansanClient()
    resp = client.create_record("contact", {"record_data": [{"id": "sc-9002", "name": "Alice"}]}, "key")
    assert resp.get("_stub") is True
    assert resp["record_id"] == "sc-9002"
    assert isinstance(resp["task_id"], int)


def test_injected_transport_disables_stub_flag():
    client = SansanClient(get=lambda url, headers, body: (200, {"data": []}))
    assert client.uses_stub_transport is False
