# CMN-C2-274 - Unit tests: CallSansanApiNode (inner Step 4, tool side-effect)
# Adapted from a sibling tool-calling golden suite (Kaonavi -> Sansan).
#
# Canon: invoked via node(state) (BaseNode.__call__ routes the full security
# pipeline); inner domain node -> caller_trust_level = TrustLevel.ANONYMOUS.value.
# The ONE documented exception: the config-override call passes a 2nd (config)
# argument, which __call__ cannot forward - that single test stays a DIRECT
# execute(state, config=...) call (ANONYMOUS node, trust gate unaffected).
#
# The node builds its client locally (SDK v1 nodes are no-arg), so error-path
# transports are exercised by monkeypatching the module's SansanClient symbol
# (our own module attribute - never a sys.modules stub of shared.*).

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from shared.secrets.inmemory_provider import InMemoryProvider

from src.nodes.call_sansan_api_node import CallSansanApiNode
from src.services.sansan_client import SansanApiError
from src.schemas.state import to_json


@pytest.fixture(autouse=True)
def _mute_audit(monkeypatch):
    monkeypatch.setattr("src.nodes.call_sansan_api_node.emit_trace_event", lambda *a, **k: None)


def _state(**overrides) -> dict:
    state = {
        "sansan_payload": to_json({"record_id": "sc-1001"}),
        "intent": "lookup_contact",
        "target_id": "sc-1001",
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "correlation_id": "call-sansan-test",
        "session_id": "s1",
        "thread_id": "th1",
        "trace_id": "t1",
        "node_history": [],
        "error_log": [],
        "execution_time": {},
    }
    state.update(overrides)
    return state


class _FakeErrorClient:
    """Stands in for SansanClient: lookup raises the documented API error."""

    def __init__(self, *args, **kwargs):
        pass

    uses_stub_transport = True

    def find_record(self, record_type, record_id, api_key):
        raise SansanApiError(403, "forbidden by integration permissions")


class _FakeLiveClient:
    """Stands in for SansanClient with a LIVE (non-stub) transport."""

    captured: dict = {}

    def __init__(self, *args, **kwargs):
        pass

    uses_stub_transport = False

    def find_record(self, record_type, record_id, api_key):
        _FakeLiveClient.captured = {
            "record_type": record_type,
            "record_id": record_id,
            "api_key": api_key,
        }
        return {"data": [{"id": record_id, "name": f"Contact {record_id}"}]}


class TestCallSansanApiNode:
    def setup_method(self):
        self.node = CallSansanApiNode()

    def test_lookup_success_via_default_v1_stub(self):
        # Default transport = deterministic, network-free v1 stub; no secret
        # provider bound -> the node runs on the documented stub placeholder.
        result = self.node(_state())
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["record_id"] == "sc-1001"
        assert result["record_ref"] == "sansan://contacts/sc-1001"
        assert result["target_id"] == "sc-1001"
        assert result["record_label"] == "Contact sc-1001"

    def test_lookup_deal_uses_deals_collection(self):
        state = _state(
            intent="lookup_deal",
            target_id="d-42",
            sansan_payload=to_json({"record_id": "d-42"}),
        )
        result = self.node(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["record_id"] == "d-42"
        assert result["record_ref"] == "sansan://deals/d-42"

    def test_create_success_via_default_v1_stub(self):
        state = _state(
            intent="create_contact",
            target_id="sc-9002",
            sansan_payload=to_json({"record_data": [{"id": "sc-9002", "name": "Alice"}]}),
        )
        result = self.node(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["record_id"] == "sc-9002"
        assert result["record_ref"] == "sansan://contacts/sc-9002"

    def test_update_success_via_default_v1_stub(self):
        state = _state(
            intent="update_contact",
            target_id="sc-1001",
            sansan_payload=to_json({"record_data": [{"id": "sc-1001"}]}),
        )
        result = self.node(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["record_id"] == "sc-1001"

    def test_sansan_config_state_field_sets_base_url(self):
        # The inner graph injects the manifest `sansan:` section as the JSON
        # sansan_config state field; the stub transport still serves the call.
        state = _state(sansan_config=to_json({"base_url": "https://sansan.example.test/v3"}))
        result = self.node(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["record_ref"] == "sansan://contacts/sc-1001"

    def test_config_override_direct_execute_call(self):
        # Documented canon exception: execute(state, config=...) takes a 2nd
        # argument that __call__ cannot forward, so this ONE test calls execute
        # directly (ANONYMOUS node - the trust gate is not the subject here).
        config = {"configurable": {"sansan": {"base_url": "https://sansan.example.test/v3"}}}
        result = self.node.execute(_state(), config=config)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["record_id"] == "sc-1001"

    def test_missing_payload_errors(self):
        result = self.node(_state(sansan_payload=None))
        assert result["status"] == AgentStatus.ERROR.value
        assert result["error_log"]

    def test_lookup_with_unresolved_id_errors(self):
        state = _state(target_id="", sansan_payload=to_json({"record_id": ""}))
        result = self.node(state)
        assert result["status"] == AgentStatus.ERROR.value
        assert any("unresolved record id" in entry for entry in result["error_log"])

    def test_update_with_unresolved_id_errors(self):
        state = _state(
            intent="update_contact",
            target_id="",
            sansan_payload=to_json({"record_data": [{"id": ""}]}),
        )
        result = self.node(state)
        assert result["status"] == AgentStatus.ERROR.value

    def test_unknown_intent_errors(self):
        result = self.node(_state(intent="delete_contact"))
        assert result["status"] == AgentStatus.ERROR.value
        assert any("unknown intent" in entry for entry in result["error_log"])

    def test_api_error_surfaces_status_error(self, monkeypatch):
        monkeypatch.setattr("src.nodes.call_sansan_api_node.SansanClient", _FakeErrorClient)
        result = self.node(_state())
        assert result["status"] == AgentStatus.ERROR.value
        assert any("403" in entry for entry in result["error_log"])

    def test_live_transport_without_secret_refuses_call(self, monkeypatch):
        # With a LIVE transport a missing SANSAN_API_KEY is a hard error -
        # a real API is never called unauthenticated.
        monkeypatch.setattr("src.nodes.call_sansan_api_node.SansanClient", _FakeLiveClient)
        result = self.node(_state())
        assert result["status"] == AgentStatus.ERROR.value
        assert any("unauthenticated" in entry for entry in result["error_log"])

    def test_live_transport_reads_key_from_ctx_secrets(self, monkeypatch):
        monkeypatch.setattr("src.nodes.call_sansan_api_node.SansanClient", _FakeLiveClient)
        _FakeLiveClient.captured = {}
        with bound_secrets(InMemoryProvider({"SANSAN_API_KEY": "mock-key-for-testing"})):
            result = self.node(_state())
        assert result["status"] == AgentStatus.SUCCESS.value
        assert _FakeLiveClient.captured["api_key"] == "mock-key-for-testing"
        assert _FakeLiveClient.captured["record_type"] == "contact"
        assert _FakeLiveClient.captured["record_id"] == "sc-1001"

    def test_audit_emits_side_effect_signals_only(self, monkeypatch):
        events = []
        monkeypatch.setattr(
            "src.nodes.call_sansan_api_node.emit_trace_event",
            lambda *args, **kwargs: events.append(args),
        )
        self.node(_state())
        payloads = {args[0]: args[1] for args in events}
        # Emit-spy asserts on the payload (args[1]) - presence signals only.
        payload = payloads["call_sansan_api_complete"]
        assert payload["intent"] == "lookup_contact"
        assert payload["has_record_id"] is True
        assert payload["stub_transport"] is True
