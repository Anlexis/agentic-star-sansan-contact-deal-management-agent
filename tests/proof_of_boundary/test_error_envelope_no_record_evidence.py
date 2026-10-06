"""Regression: an ERROR envelope must not disclose Sansan record evidence, and
must carry closed-set labels only.

Molt source review, 2026-09-04 (wave-7 re-review of develop HEAD): "existing-ERROR
path includes Sansan record evidence and retains output-bearing state."

The `errored` branch of PostProcessNode rebuilt `formatted_output` from
`record_id` / `record_ref` read straight back out of state, and returned ONLY
that plus `status` - so every other output-bearing field (`result`,
`record_label`, `confirmation`, `sansan_payload`) survived in state untouched.

Those identifiers ARE the Sansan write evidence: this node's own gate
(`_security_gate_output`, is_success=True) REFUSES a SUCCESS that lacks them. An
error envelope carrying them tells a caller who is being informed of a FAILURE
that a write occurred and which record it touched. Sansan records are business-card
contacts and sales deals - personal data - so the identifier, the display name,
the company and the email are all disclosure, not diagnostics.

The same review's family-level finding: an envelope that clears the evidence but
still projects `error_log` publishes node-authored text - upstream exception and
response bodies, identifiers, names, emails - and truncation, path stripping or
credential-only redaction of that text is not a closed-set error contract. The
caller-visible error must be closed-set labels only.

The containment contract is not "shape a nicer error mapping". It is: no un-gated
caller-facing content, record evidence or error text survives on any error path -
neither in the replacement `formatted_output`, nor in the state left behind for a
checkpoint or a downstream reader. `error_log` stays the internal channel.
"""

import json

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.graph.graph import SansanContactDealAgent
from src.nodes.call_sansan_api_node import CallSansanApiNode
from src.nodes.post_process_node import (
    ERROR_REASONS,
    _REASON_OUTPUT_WITHHELD,
    _REASON_WORKFLOW_FAILED,
    PostProcessNode,
)
from src.schemas.state import to_json

# Record evidence + the personal data hanging off it.
_RECORD_ID = "sc-1001"
_RECORD_REF = "sansan://contacts/sc-1001"
_PERSON = "Haruka Tanaka"
_COMPANY = "Acme Trading"
_EMAIL = "haruka@acme.example"

# An error_log line of the kind an upstream failure produces: a name and a
# credential-shaped token inside an echoed response body. Assembled at runtime
# so no credential-shaped literal is committed.
_SENTINEL = "boom: upstream said {'customer':'A. Tanaka','token':'" + "sk-" + "live-" + "x" * 3 + "'}"
_SENTINEL_FRAGMENTS = ("A. Tanaka", "boom: upstream", "sk-" + "live-")

# State keys the outer State accumulates (reducer appends) rather than overwrites.
_ACCUMULATED = ("error_log", "node_history")


def _errored_state() -> dict:
    return {
        "status": AgentStatus.ERROR.value,
        "error_log": ["CallSansanApiNode: Sansan API error 500"],
        "record_id": _RECORD_ID,
        "record_ref": _RECORD_REF,
        "target_id": _RECORD_ID,
        "record_label": _PERSON,
        "intent": "update_contact",
        "confirmation": f"Updated Sansan contact '{_PERSON}' - ref={_RECORD_REF} - id={_RECORD_ID}",
        "sansan_payload": to_json(
            {
                "record_data": [
                    {
                        "id": _RECORD_ID,
                        "name": _PERSON,
                        "custom_fields": [
                            {"name": "Company", "values": [_COMPANY]},
                            {"name": "Email", "values": [_EMAIL]},
                        ],
                    }
                ]
            }
        ),
        "result": {"record_id": _RECORD_ID, "confirmation": "done"},
    }


def _successful_state() -> dict:
    """The outer state after a clean inner run, shaped for the full node pipeline."""
    return {
        "status": AgentStatus.SUCCESS.value,
        "record_id": _RECORD_ID,
        "record_ref": _RECORD_REF,
        "target_id": _RECORD_ID,
        "record_label": "Contact sc-1001",
        "intent": "lookup_contact",
        "confirmation": f"Retrieved Sansan contact - ref={_RECORD_REF} - id={_RECORD_ID}",
        "sansan_payload": to_json({"record_id": _RECORD_ID}),
        "result": {"record_id": _RECORD_ID, "confirmation": "done"},
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "correlation_id": "pb-error-envelope",
        "node_history": [],
        "error_log": [],
        "execution_time": {},
    }


def _flatten(value) -> str:
    """Render every reachable string in a returned value - nesting is not cover."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_flatten(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten(v) for v in value)
    return str(value)


def _merge(state: dict, partial: dict) -> dict:
    """Merge a node's partial state the way the outer State reducer does."""
    merged = dict(state)
    for key, value in partial.items():
        if key in _ACCUMULATED:
            merged[key] = list(merged.get(key, [])) + list(value)
        else:
            merged[key] = value
    return merged


def _caller_envelope(state: dict, partial: dict) -> dict:
    """The invoke result the caller receives: the agent's own get_output()."""
    return SansanContactDealAgent().get_output(_merge(state, partial))


class TestErrorEnvelopeContainment:
    def test_error_envelope_carries_no_record_evidence(self):
        out = PostProcessNode().execute(_errored_state())

        assert out["status"] == AgentStatus.ERROR.value

        # formatted_output must be PRESENT and TRUTHY. The framework projects
        # `formatted_output or result` with no status check, so a falsy value
        # re-opens the fallback onto whatever survived in state.
        assert "formatted_output" in out
        assert out["formatted_output"], "falsy formatted_output re-opens the `or result` fallback"

        shipped = _flatten(out["formatted_output"])
        assert _RECORD_ID not in shipped, "record id shipped in the error envelope"
        assert _RECORD_REF not in shipped, "record ref shipped in the error envelope"
        assert "sansan://" not in shipped
        assert _PERSON not in shipped
        assert _COMPANY not in shipped
        assert _EMAIL not in shipped
        assert "Updated Sansan contact" not in shipped

    def test_error_envelope_is_the_reason_code_only(self):
        """Closed set: the envelope is exactly the declared constant - no
        error_log line, no diagnostics, nothing derived from state."""
        out = PostProcessNode().execute(_errored_state())

        assert out["formatted_output"] == {"reason": _REASON_WORKFLOW_FAILED}
        assert set(out["formatted_output"].values()) <= ERROR_REASONS
        assert "Sansan API error" not in _flatten(out["formatted_output"])

    def test_error_path_clears_output_bearing_state(self):
        """Omitting a field from one envelope is not clearing it from state."""
        out = PostProcessNode().execute(_errored_state())
        for field in (
            "result",
            "confirmation",
            "record_label",
            "sansan_payload",
            "record_id",
            "record_ref",
            "target_id",
        ):
            assert field in out, f"{field} not cleared on the error path"
            assert not out[field], f"{field} still carries content on the error path"

    def test_success_path_still_returns_the_answer(self):
        """Control: containment must not empty out the clean path."""
        out = PostProcessNode().execute(
            {
                "status": AgentStatus.SUCCESS.value,
                "record_id": _RECORD_ID,
                "record_ref": _RECORD_REF,
                "record_label": _PERSON,
                "intent": "lookup_contact",
                "confirmation": f"Retrieved Sansan contact - id={_RECORD_ID}",
                "sansan_payload": to_json({"record_id": _RECORD_ID}),
            }
        )
        assert out["status"] == AgentStatus.SUCCESS.value
        shipped = _flatten(out["formatted_output"])
        assert _RECORD_ID in shipped, "the success path must still return the record evidence"
        assert _RECORD_REF in shipped
        assert "reason" not in out["formatted_output"]


class TestErrorEnvelopeAtTheCaller:
    """What the caller receives on every non-success path, built the way the
    framework builds it: the node's partial state merged into the outer state,
    then the agent's own get_output() (`formatted_output or result`, no status
    check).

    The inner-workflow error is driven through execute(): under the real
    pipeline BaseNode.__call__ short-circuits post_process on an errored state
    and the backbone routes ERROR to finalize, so execute() is the surface on
    that path. Both gate blocks run through the full node pipeline.
    """

    def test_inner_workflow_error_reaches_the_caller_as_the_reason_code_only(self):
        state = _errored_state()
        state["error_log"] = [_SENTINEL, "CallSansanApiNode: Sansan API error 500"]
        partial = PostProcessNode().execute(dict(state))
        envelope = _caller_envelope(state, partial)
        rendered = json.dumps(envelope, default=str)

        assert envelope["status"] == AgentStatus.ERROR.value
        assert envelope["output"] == {"reason": _REASON_WORKFLOW_FAILED}
        for fragment in _SENTINEL_FRAGMENTS:
            assert fragment not in rendered, f"{fragment!r} reached the caller"
        for evidence in (_RECORD_ID, _RECORD_REF, _PERSON, _COMPANY, _EMAIL, "Sansan API error"):
            assert evidence not in rendered, f"{evidence!r} reached the caller"
        assert "error_log" not in envelope
        # Structured product keys stay withheld on a non-SUCCESS run.
        assert "record_id" not in envelope
        assert "record_ref" not in envelope

    @pytest.mark.parametrize("block", ["missing-evidence", "credential-in-payload"])
    def test_gate_block_reaches_the_caller_as_the_reason_code_only(self, block):
        """Both gate blocks through the FULL node pipeline, with an error_log
        line already present from upstream: the caller's output is exactly the
        closed-set envelope, and neither the upstream line nor the gate's own
        violation text appears anywhere in the caller's envelope."""
        state = _successful_state()
        state["error_log"] = [_SENTINEL]
        bearer_like = "Bearer " + "k" * 24
        if block == "missing-evidence":
            state["record_id"] = ""
            state["record_ref"] = ""
        else:
            state["sansan_payload"] = to_json({"record_data": [{"name": _PERSON, "note": bearer_like}]})
        partial = PostProcessNode()(dict(state))
        envelope = _caller_envelope(state, partial)
        rendered = json.dumps(envelope, default=str)

        assert envelope["status"] == AgentStatus.ERROR.value
        assert envelope["output"] == {"reason": _REASON_OUTPUT_WITHHELD}
        for fragment in _SENTINEL_FRAGMENTS:
            assert fragment not in rendered, f"{fragment!r} reached the caller"
        assert bearer_like not in rendered
        assert _PERSON not in rendered
        assert partial["error_log"], "the gate must record its violation operator-side"
        for violation in partial["error_log"]:
            assert violation not in rendered, "violation text reached the caller"
        assert "record_id" not in envelope

    def test_clean_run_control_still_delivers_the_answer(self):
        """A refuse-everything gate would pass the tests above; this one fails it."""
        state = _successful_state()
        partial = PostProcessNode()(dict(state))
        envelope = _caller_envelope(state, partial)

        assert envelope["status"] == AgentStatus.SUCCESS.value
        assert envelope["output"]["record_id"] == _RECORD_ID
        assert envelope["output"]["record_ref"] == _RECORD_REF
        assert "reason" not in envelope["output"]
        assert envelope["record_id"] == _RECORD_ID  # structured surface, SUCCESS only


class TestErrorLogNamesNoRecord:
    """The reasons written to error_log are a channel of their own.

    error_log is operator-side - the caller-visible envelope never projects it
    - but it is the audit trail and a checkpoint reader's view, so a reason
    must still be a closed-set label (record type, HTTP status, exception
    type), never the record value or an upstream body.
    """

    def _state(self, **overrides) -> dict:
        state = {
            "sansan_payload": to_json({"record_id": _RECORD_ID}),
            "intent": "lookup_contact",
            "target_id": _RECORD_ID,
            "correlation_id": "pb-error-envelope",
            "session_id": "pb-s1",
            "thread_id": "pb-th1",
            "trace_id": "pb-t1",
            "node_history": [],
            "error_log": [],
            "execution_time": {},
        }
        state.update(overrides)
        return state

    def test_record_not_found_reason_does_not_name_the_record(self, monkeypatch):
        monkeypatch.setattr("src.nodes.call_sansan_api_node.emit_trace_event", lambda *a, **k: None)

        class _EmptyLookupClient:
            uses_stub_transport = True

            def __init__(self, *args, **kwargs):
                pass

            def find_record(self, record_type, record_id, api_key):
                return {"data": []}

        monkeypatch.setattr("src.nodes.call_sansan_api_node.SansanClient", _EmptyLookupClient)
        out = CallSansanApiNode().execute(self._state())
        assert out["status"] == AgentStatus.ERROR.value
        joined = " ".join(out["error_log"])
        assert _RECORD_ID not in joined, "the not-found reason names the record id"
        # The reason still has to be actionable: it names the record TYPE
        # (a closed-set label), not the record.
        assert "contact" in joined

    def test_upstream_api_failure_reason_carries_no_upstream_body(self, monkeypatch):
        """A live tenant's error body is unbounded third-party text; only the
        HTTP status - a closed-set signal - belongs in the reason."""
        monkeypatch.setattr("src.nodes.call_sansan_api_node.emit_trace_event", lambda *a, **k: None)
        from src.services.sansan_client import SansanApiError

        class _ApiErrorClient:
            uses_stub_transport = True

            def __init__(self, *args, **kwargs):
                pass

            def find_record(self, record_type, record_id, api_key):
                raise SansanApiError(403, f"denied for {_PERSON} <{_EMAIL}> at {_COMPANY}")

        monkeypatch.setattr("src.nodes.call_sansan_api_node.SansanClient", _ApiErrorClient)
        out = CallSansanApiNode().execute(self._state())
        assert out["status"] == AgentStatus.ERROR.value
        joined = " ".join(out["error_log"])
        assert "403" in joined, "the HTTP status is the actionable signal - keep it"
        assert _PERSON not in joined
        assert _EMAIL not in joined
        assert _COMPANY not in joined

    def test_transport_failure_reason_carries_the_exception_class_only(self, monkeypatch):
        """A transport error's text can carry the request URL and the record
        id; the reason records the exception CLASS, never its message."""
        monkeypatch.setattr("src.nodes.call_sansan_api_node.emit_trace_event", lambda *a, **k: None)

        class _TransportFailureClient:
            uses_stub_transport = True

            def __init__(self, *args, **kwargs):
                pass

            def find_record(self, record_type, record_id, api_key):
                raise ConnectionError(f"peer closed https://sansan.example/contacts/{_RECORD_ID} for {_PERSON}")

        monkeypatch.setattr("src.nodes.call_sansan_api_node.SansanClient", _TransportFailureClient)
        out = CallSansanApiNode().execute(self._state())
        assert out["status"] == AgentStatus.ERROR.value
        joined = " ".join(out["error_log"])
        assert "ConnectionError" in joined
        assert _RECORD_ID not in joined
        assert _PERSON not in joined
        assert "https://" not in joined
