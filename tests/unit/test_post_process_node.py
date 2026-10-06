# CMN-C2-274 - Unit tests: PostProcessNode (outer backbone, domain output gate)
# Adapted from a sibling tool-calling golden suite (Kaonavi -> Sansan).
#
# Canon: invoked via node(state) (BaseNode.__call__ routes the full security
# pipeline around execute()); this backbone formatter declares ANONYMOUS -> the
# state builder sets caller_trust_level = TrustLevel.ANONYMOUS.value. The
# domain output gate is the MODULE-LEVEL _security_gate_output() helper (the
# framework gate methods are @final and the real SDK auto-wraps _extra_ hooks),
# so the helper is also unit-tested directly as a plain function.
#
# Assertions on the gate are BEHAVIOURAL (status, what is cleared, what is
# carried), never a message's exact wording.
#
# The ERROR envelope is a closed set: on every non-success path the caller
# receives this module's own constants and nothing read from error_log or the
# gate's violations. A sentinel seeded into error_log must reach nowhere in
# what the node returns.

import json

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.post_process_node import (
    ERROR_REASONS,
    _REASON_OUTPUT_WITHHELD,
    _REASON_WORKFLOW_FAILED,
    PostProcessNode,
    _security_gate_output,
)
from src.schemas.state import to_json


@pytest.fixture(autouse=True)
def _mute_audit(monkeypatch):
    monkeypatch.setattr("src.nodes.post_process_node.emit_trace_event", lambda *a, **k: None)


def _state(**overrides) -> dict:
    state = {
        "status": AgentStatus.SUCCESS.value,
        "record_id": "sc-1001",
        "record_ref": "sansan://contacts/sc-1001",
        "record_label": "Contact sc-1001",
        "intent": "lookup_contact",
        "confirmation": "Retrieved Sansan contact 'Contact sc-1001' - ref=sansan://contacts/sc-1001 - id=sc-1001",
        "sansan_payload": to_json({"record_id": "sc-1001"}),
        "result": {"record_id": "sc-1001", "confirmation": "ok"},
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "correlation_id": "post-process-test",
        "node_history": [],
        "error_log": [],
        "execution_time": {},
    }
    state.update(overrides)
    return state


def _bearer(fill: str = "a") -> str:
    # Built at runtime so no credential-shaped literal is committed.
    return "Bearer " + fill * 24


def _sentinel() -> str:
    """An error_log line of the kind an upstream failure produces: a name and a
    credential-shaped token inside an echoed response body. Assembled at
    runtime so no credential-shaped literal is committed."""
    token = "sk-" + "live-" + "x" * 3
    return "boom: upstream said {'customer':'A. Tanaka','token':'" + token + "'}"


# Fragments of the sentinel that must survive nowhere in a returned mapping.
_SENTINEL_FRAGMENTS = ("A. Tanaka", "boom: upstream", "sk-" + "live-")


def _leaves(value):
    """Every key and scalar inside `value`, rendered as text, at any depth."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _leaves(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _leaves(item)
    else:
        yield str(value)


class TestPostProcessNode:
    def setup_method(self):
        self.node = PostProcessNode()

    def test_success_formats_output(self):
        result = self.node(_state())
        assert result["status"] == AgentStatus.SUCCESS.value
        # Regression guard: State carries the enum .value STRING, never the
        # bare AgentStatus enum member. AgentStatus subclasses str, so an
        # isinstance check cannot catch a bare-enum write; the exact-type
        # check can. noqa is deliberate.
        assert type(result["status"]) is str  # noqa: E721
        out = result["formatted_output"]
        assert out["record_id"] == "sc-1001"
        assert out["record_ref"] == "sansan://contacts/sc-1001"
        assert out["intent"] == "lookup_contact"
        assert out["confirmation"].startswith("Retrieved Sansan contact")
        # State-contract round-trip: the JSON sansan_payload string surfaces parsed.
        assert out["sansan_payload"] == {"record_id": "sc-1001"}
        assert "reason" not in out

    def test_error_status_preserved(self):
        """Inner-workflow error must not be masked as success. Real-SDK
        pipeline behavior: BaseNode.__call__ short-circuits on an incoming
        errored state (execute() is skipped), so the error status + error_log
        pass through untouched and no success shape is fabricated."""
        state = _state(
            status=AgentStatus.ERROR.value,
            record_id="",
            record_ref="",
            error_log=["CallSansanApiNode: Sansan API error 403"],
        )
        result = self.node(state)
        assert result["status"] == AgentStatus.ERROR.value
        assert "Sansan API error 403" in "\n".join(result["error_log"])
        assert "formatted_output" not in result

    def test_error_status_as_string_value_preserved(self):
        """The framework may carry status as the enum .value (string) at the boundary."""
        result = self.node(_state(status=AgentStatus.ERROR.value, error_log=["boom"]))
        assert result["status"] == AgentStatus.ERROR.value
        assert "formatted_output" not in result

    def test_gate_blocks_success_without_record_evidence(self):
        """Full node path: a SUCCESS output missing record_id/record_ref is blocked."""
        result = self.node(_state(record_id="", record_ref=""))
        assert result["status"] == AgentStatus.ERROR.value
        assert result["error_log"]


class TestViolationClearsOutput:
    """A violating gate must CLEAR the output-bearing fields - raising or
    returning an error alone is not containment.

    The response envelope falls back to state["result"] even on an error
    status, so a gate that leaves the blocked content in state still ships it
    inside the error envelope. These tests call execute() directly (no
    framework short-circuit on the pre-errored state) and assert the blocked
    content is gone from EVERY output-bearing field of the returned delta, and
    that what replaces formatted_output is the closed-set reason code alone.
    """

    def setup_method(self):
        self.node = PostProcessNode()

    def _assert_cleared(self, result, leaked: str, reason: str):
        assert result["status"] == AgentStatus.ERROR.value
        # The envelope is the reason code - truthy, and nothing else.
        assert result["formatted_output"] == {"reason": reason}
        assert result["result"] is None
        assert result["confirmation"] == ""
        assert result["sansan_payload"] is None
        assert result["record_label"] == ""
        assert leaked not in json.dumps(result, default=str)

    def test_credential_violation_clears_every_output_field(self):
        leaked = _bearer("a")
        result = self.node.execute(_state(confirmation=f"done - {leaked}"))
        self._assert_cleared(result, leaked, _REASON_OUTPUT_WITHHELD)

    def test_credential_nested_in_payload_clears_every_output_field(self):
        leaked = "sk-" + "b" * 24
        result = self.node.execute(
            _state(sansan_payload=to_json({"record_data": [{"custom_fields": [{"values": [f"key {leaked}"]}]}]}))
        )
        self._assert_cleared(result, leaked, _REASON_OUTPUT_WITHHELD)

    def test_credential_in_error_shape_is_contained_too(self):
        """A remote error body arrives in error_log. The error path publishes
        the reason code only, so the body cannot reach the caller through it."""
        leaked = "eyJ" + "c" * 16
        result = self.node.execute(_state(status=AgentStatus.ERROR.value, error_log=[f"upstream failure: {leaked}"]))
        self._assert_cleared(result, leaked, _REASON_WORKFLOW_FAILED)

    def test_violation_log_names_location_not_value(self):
        leaked = _bearer("d")
        result = self.node.execute(_state(confirmation=f"done - {leaked}"))
        joined = " ".join(result["error_log"])
        assert leaked not in joined
        assert "confirmation" in joined


class TestSecurityGateOutputHelper:
    """The module-level domain output gate as a plain function (not a node call)."""

    def test_passes_success_with_record_evidence(self):
        violations = _security_gate_output(
            {"record_id": "sc-1001", "record_ref": "sansan://contacts/sc-1001", "confirmation": "ok"},
            is_success=True,
        )
        assert violations == []

    def test_blocks_success_without_record_evidence(self):
        violations = _security_gate_output(
            {"record_id": "", "record_ref": "", "confirmation": "looks done"},
            is_success=True,
        )
        assert len(violations) == 1
        assert "record_id/record_ref" in violations[0]

    def test_blocks_credential_shaped_value(self):
        violations = _security_gate_output(
            {"record_id": "sc-1001", "note": _bearer("a")},
            is_success=True,
        )
        assert any("note" in v for v in violations)

    def test_walks_nested_structures(self):
        """A credential one mapping deep must not slip past a top-level scan."""
        bearer_like = _bearer("e")
        violations = _security_gate_output(
            {"record_id": "sc-1001", "sansan_payload": {"record_data": [{"name": bearer_like}]}},
            is_success=True,
        )
        assert violations, "nested credential-shaped value not caught"
        assert bearer_like not in " ".join(violations)

    def test_nested_control_clean_payload_passes(self):
        """The verifier control: the same nested shape without a credential passes."""
        violations = _security_gate_output(
            {"record_id": "sc-1001", "sansan_payload": {"record_data": [{"name": "Contact sc-1001"}]}},
            is_success=True,
        )
        assert violations == []

    def test_error_output_not_required_to_carry_evidence(self):
        violations = _security_gate_output({"record_id": "", "record_ref": ""}, is_success=False)
        assert violations == []


# Every non-success path this node has. `via` says how the path is reached:
# node(state) runs the framework pipeline; execute() is used for an errored
# state because BaseNode.__call__ short-circuits on it (and the backbone routes
# an error straight to finalize), so that branch is only reachable from inside.
# Credential-in-state cases use execute() as well, matching the containment
# tests above: the refusal proven is this node's own gate, not a framework one.
_ERROR_PATHS = [
    pytest.param(
        {"status": AgentStatus.ERROR.value, "record_id": "", "record_ref": ""},
        "execute",
        _REASON_WORKFLOW_FAILED,
        id="inner-workflow-error",
    ),
    pytest.param(
        {"status": AgentStatus.ERROR.value},
        "execute",
        _REASON_WORKFLOW_FAILED,
        id="inner-workflow-error-with-evidence-and-answer-still-in-state",
    ),
    pytest.param(
        {"status": AgentStatus.ERROR.value, "error_log": [_sentinel(), "CallSansanApiNode: rejected " + _bearer("f")]},
        "execute",
        _REASON_WORKFLOW_FAILED,
        id="inner-workflow-error-with-credential-in-error-log",
    ),
    pytest.param(
        {"record_id": "", "record_ref": ""},
        "call",
        _REASON_OUTPUT_WITHHELD,
        id="gate-missing-record-evidence",
    ),
    pytest.param(
        {"confirmation": "done - " + _bearer("g")},
        "execute",
        _REASON_OUTPUT_WITHHELD,
        id="gate-credential-in-confirmation",
    ),
    pytest.param(
        {"sansan_payload": to_json({"record_data": [{"custom_fields": [{"values": ["key sk-" + "h" * 24]}]}]})},
        "execute",
        _REASON_OUTPUT_WITHHELD,
        id="gate-credential-nested-in-payload",
    ),
]


class TestErrorEnvelopeIsClosedSet:
    """Whatever the non-success path, the caller-visible envelope is made of
    this module's own constants - never of node-authored text.

    error_log is seeded with a recognisable sentinel on every path: a name and
    a credential-shaped token inside an echoed upstream body, which is exactly
    what an API-call failure can put there. Truncating or redacting such a line
    is not a closed set, so it must appear nowhere in what the node returns.
    """

    def setup_method(self):
        self.node = PostProcessNode()

    def _drive(self, overrides: dict, via: str) -> dict:
        state = _state(**{"error_log": [_sentinel()], **overrides})
        return self.node.execute(state) if via == "execute" else self.node(state)

    @pytest.mark.parametrize(("overrides", "via", "reason"), _ERROR_PATHS)
    def test_envelope_values_are_declared_constants(self, overrides, via, reason):
        result = self._drive(overrides, via)

        assert result["status"] == AgentStatus.ERROR.value
        envelope = result["formatted_output"]
        assert set(envelope) == {"reason"}, envelope
        assert set(envelope.values()) <= ERROR_REASONS, envelope
        assert envelope["reason"] == reason

    @pytest.mark.parametrize(("overrides", "via", "reason"), _ERROR_PATHS)
    def test_envelope_stays_truthy_and_result_is_cleared(self, overrides, via, reason):
        """`formatted_output or result`: a falsy envelope re-opens the fallback."""
        result = self._drive(overrides, via)

        assert result["formatted_output"], "a falsy formatted_output re-opens the `result` fallback"
        assert result["result"] is None

    @pytest.mark.parametrize(("overrides", "via", "reason"), _ERROR_PATHS)
    def test_seeded_error_text_appears_nowhere_in_the_returned_mapping(self, overrides, via, reason):
        result = self._drive(overrides, via)

        leaves = list(_leaves(result))
        for fragment in _SENTINEL_FRAGMENTS:
            assert not any(fragment in leaf for leaf in leaves), (fragment, result)
        rendered = json.dumps(result, default=str)
        assert not any(fragment in rendered for fragment in _SENTINEL_FRAGMENTS), rendered

    @pytest.mark.parametrize(("overrides", "via", "reason"), _ERROR_PATHS)
    def test_record_evidence_and_content_are_cleared_on_every_path(self, overrides, via, reason):
        result = self._drive(overrides, via)

        for field in ("record_id", "record_ref", "target_id", "record_label", "confirmation", "sansan_payload"):
            assert field in result, f"{field} not cleared"
            assert not result[field], f"{field} still carries content"

    def test_inner_error_entries_are_not_re_emitted(self):
        """The state reducer appends error_log; re-emitting the inner entries
        would duplicate every line, and the caller never sees them anyway."""
        result = self.node.execute(_state(status=AgentStatus.ERROR.value, error_log=[_sentinel()]))

        assert "error_log" not in result

    def test_gate_violations_travel_in_error_log_only(self):
        result = self.node(_state(record_id="", record_ref="", error_log=[_sentinel()]))

        assert any("output gate" in entry for entry in result["error_log"])
        assert "output gate" not in json.dumps(result["formatted_output"])
        assert _sentinel() not in result["error_log"]

    def test_audit_events_carry_a_reason_and_a_count_only(self, monkeypatch):
        events = []
        monkeypatch.setattr(
            "src.nodes.post_process_node.emit_trace_event",
            lambda *args, **kwargs: events.append(args),
        )
        self.node.execute(_state(status=AgentStatus.ERROR.value, error_log=[_sentinel(), "second entry"]))
        self.node(_state(record_id="", record_ref="", error_log=[_sentinel()]))

        payloads = {args[0]: args[1] for args in events}
        assert payloads["post_process_workflow_failed"] == {"reason": _REASON_WORKFLOW_FAILED, "error_count": 2}
        assert payloads["post_process_blocked"] == {"reason": _REASON_OUTPUT_WITHHELD, "violations": 1}
        assert not any(fragment in json.dumps(list(payloads.values())) for fragment in _SENTINEL_FRAGMENTS)
