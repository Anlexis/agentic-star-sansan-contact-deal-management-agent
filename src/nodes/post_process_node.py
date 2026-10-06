"""AgentCore Platform v1.0 - outer post_process node.

Cat 2 outer backbone: finalize the response after the inner Sansan workflow
graph has run. GraphNode.merge_output() maps the inner result into the outer
state; this node shapes the caller-facing `formatted_output`.

The domain output gate is the MODULE-LEVEL `_security_gate_output()` below,
called from execute(). It is deliberately NOT an instance method and NOT the
framework `_extra_security_gate_output` hook - the framework gate methods are
@final on FunctionNode and the SDK auto-wraps `_extra_` hooks (which breaks the
.invoke() chain), so domain checks live in a module-level helper invoked inline.

The gate walks the WHOLE output structure, not just its top-level string
values: `sansan_payload` is a nested mapping whose entries carry
caller-derived text, so a scan that only looked at top-level strings would
step straight past a credential sitting in a record field.

On ANY error return - a gate violation or an inner-workflow failure - the node
must not merely fail: the response envelope falls back to state["result"] even
on an error status, so the content has to be CLEARED from every output-bearing
state field or it would still ship inside the error envelope. That includes the
Sansan record evidence (record_id / record_ref / target_id): a caller being told
the operation failed must not learn that a write happened or which record it
touched. Fail closed by overwriting those fields.

What the ERROR envelope may say
-------------------------------
The caller-visible ERROR envelope - `formatted_output` on any non-success path
- carries closed-set labels only: a constant reason code chosen by this module
(one of `ERROR_REASONS`). It never carries `error_log`, the gate's violation
entries, or any other node-authored text. Those lines can embed upstream
response bodies, identifiers, names or emails, and truncating, path-stripping
or credential-only redaction of them is not a closed set. `error_log` stays as
it is - it is the internal channel, the state reducer appends to it and the
audit trail needs it - it is simply never projected to the caller. One helper,
`_contain()`, produces the envelope on every non-success path, so no path can
publish more than the reason code. The envelope is truthy by construction: the
framework projects `formatted_output or result` with no status check, and a
falsy value would re-open the fallback onto whatever survived in state.

A violation entry written to error_log names the LOCATION of the problem and
never quotes what was found there.
"""

import re
from typing import Any

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import from_json

# Credential-shaped strings that must never reach the caller (defence in depth -
# the framework credential scan in FunctionNode also runs on every result).
_CREDENTIAL_LIKE_RE = re.compile(r"eyJ[A-Za-z0-9._-]{10,}|sk-[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9._-]{16,}")

# Reason codes - the ONLY values the caller-visible ERROR envelope may carry.
# Chosen here, never derived from state, so the envelope is a closed set.
_REASON_WORKFLOW_FAILED = "workflow_failed"  # the inner workflow reported an error
_REASON_OUTPUT_WITHHELD = "output_withheld"  # the output gate refused the response
ERROR_REASONS = frozenset({_REASON_WORKFLOW_FAILED, _REASON_OUTPUT_WITHHELD})

# The content must be cleared from EVERY output-bearing state field on EVERY
# error return - the gate-violation path and the inner-workflow error path
# alike. The response envelope reads formatted_output/result even on an error
# status, and the other fields feed downstream formatting. `formatted_output`
# itself is not listed here: _contain() replaces it with the closed-set envelope.
_CLEARED_ON_ERROR: "dict[str, Any]" = {
    "result": None,
    "confirmation": "",
    "sansan_payload": None,
    "record_label": "",
    # Sansan record evidence. Cleared alongside the content fields so the
    # identifiers cannot be picked up out of state by a checkpoint or a
    # downstream reader after the caller has been told the operation failed.
    "record_id": "",
    "record_ref": "",
    "target_id": "",
}


def _contain(reason: str, new_errors: "list[str] | None" = None) -> "dict[str, Any]":
    """The node result for any non-success outcome.

    Error status, every output-bearing field cleared, and an envelope made of
    closed-set labels only: `reason` is one of ERROR_REASONS. `new_errors` are
    the violations this node authored; they go to `error_log` - the internal
    channel, accumulated by the state reducer - and never enter the envelope.
    On an already-errored state nothing is re-emitted: error_log already holds
    the inner entries and the reducer appends, so re-emitting them here would
    duplicate every line.

    The envelope stays TRUTHY on purpose. The framework projects
    `formatted_output or result` with no status check, so an empty mapping
    here would fall through to `result` and defeat the clearing.
    """
    contained: dict[str, Any] = dict(_CLEARED_ON_ERROR)
    contained["formatted_output"] = {"reason": reason}
    contained["status"] = AgentStatus.ERROR.value
    if new_errors:
        contained["error_log"] = list(new_errors)
    return contained


def _walk_strings(value: object, path: str) -> "list[tuple[str, str]]":
    """Yield every (path, string) in the output, however deeply it is nested.

    Mappings, sequences and bare strings are all reachable representations of a
    caller-facing value, so all three are walked.
    """
    found: list[tuple[str, str]] = []
    if isinstance(value, str):
        found.append((path, value))
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(_walk_strings(item, f"{path}['{key}']"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_walk_strings(item, f"{path}[{index}]"))
    return found


def _security_gate_output(formatted_output: "dict[str, Any]", is_success: bool) -> "list[str]":
    """Domain output gate (module-level; called from PostProcessNode.execute()).

    Blocks (returns violations for):
      - a SUCCESS response with no record evidence (record_id/record_ref),
        which would misrepresent the Sansan action outcome to the caller;
      - any credential-shaped string ANYWHERE in the caller-facing output,
        including inside nested payload mappings and lists.

    Violations are error_log entries (internal); they never reach the caller.
    """
    problems: list[str] = []
    if is_success and not (formatted_output.get("record_id") or formatted_output.get("record_ref")):
        problems.append("PostProcess output gate: SUCCESS output missing record_id/record_ref evidence")
    for path, text in _walk_strings(formatted_output, "formatted_output"):
        if _CREDENTIAL_LIKE_RE.search(text):
            # Name the location, never the matched value.
            problems.append(f"PostProcess output gate: credential-like value in {path}")
    return problems


class PostProcessNode(FunctionNode):
    """Format the final agent output."""

    # Read-only formatting of the already-produced result - default permissive.
    required_trust_level = TrustLevel.ANONYMOUS

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        # The inner workflow failed: keep the error status (never mask it) and
        # publish nothing of it. error_log already carries the inner entries -
        # Sansan API error text included - and none of it is projected: an
        # identifier, a name or a third-party response body is not bounded by
        # truncation or credential-only redaction. The audit trail gets the
        # reason code and a count, never text. Under the real pipeline
        # BaseNode.__call__ short-circuits on an errored state before execute()
        # runs and the backbone routes ERROR straight to finalize, so this
        # branch is defence in depth for direct invocation.
        if state.get("status") == AgentStatus.ERROR.value:
            emit_trace_event(
                "post_process_workflow_failed",
                {"reason": _REASON_WORKFLOW_FAILED, "error_count": len(state.get("error_log") or [])},
                state,
            )
            return _contain(_REASON_WORKFLOW_FAILED)

        formatted_output = {
            "record_id": state.get("record_id", ""),
            "record_ref": state.get("record_ref", ""),
            "record_label": state.get("record_label", ""),
            "intent": state.get("intent", ""),
            "confirmation": state.get("confirmation", ""),
            "sansan_payload": from_json(state.get("sansan_payload"), {}),
        }

        # Domain output gate (module-level helper - see module docstring).
        violations = _security_gate_output(formatted_output, is_success=True)
        if violations:
            # Fail closed: the violations go to error_log (operator-side), never
            # to the caller, who receives the reason code only.
            emit_trace_event(
                "post_process_blocked",
                {"reason": _REASON_OUTPUT_WITHHELD, "violations": len(violations)},
                state,
            )
            return _contain(_REASON_OUTPUT_WITHHELD, violations)

        # Audit the final response shaping - outcome signals only, no payload content.
        emit_trace_event(
            "post_process_complete",
            {
                "intent": state.get("intent", ""),
                "has_record_id": bool(state.get("record_id")),
            },
            state,
        )

        return {
            "formatted_output": formatted_output,
            "status": AgentStatus.SUCCESS.value,
        }
