"""AgentCore Platform v1.0 - outer pre_process node.

Cat 2 outer backbone: validate the caller's request (raw NL text + the
structured caller fields) and serialize it into `validated_input` for the
inner Sansan workflow graph. This node OWNS the caller-data contract: every
field a caller can supply is checked here, against explicit bounds, before any
of it reaches the workflow. Business rules (intent, entity extraction) live in
the inner graph.

Caller contract (`input_context`), every field optional:

    record_id / record_hint /             target record: a Sansan contact or
    contact_id / deal_id                  deal record id (inert identifier)

Rules applied to it:
  - the value must be a STRING. A number, boolean, mapping or list is refused
    outright, never coerced: str(float("nan")) is "nan", which is
    identifier-shaped, so coercion would let a non-finite value name the
    target of a write.
  - the value must match the record-id SHAPE (alphanumeric with dash and
    underscore, 40 characters at most). The id renders into the caller-facing
    confirmation, so free text here would be caller-controlled output.
  - instruction-override text (directives aimed at the MODEL: role
    reassignment, system-prompt manipulation, chat-template control tokens) is
    REFUSED, fail closed, on BOTH caller text channels — the raw instruction
    text and every decoded string in input_context, keys included, at any
    depth — before anything is carried forward. The screen is the template's
    own (src/services/security.py), never delegated to the framework gate.
  - a refusal names the FIELD and never echoes the offending value; a hostile
    field NAME is masked, never echoed either.
  - an absent hint is simply absent: the workflow falls back to an id named in
    the instruction text.
"""

import json
import re
from typing import Any

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import to_json
from src.services.security import contains_instruction_override, sanitize_query

# A Sansan record id: alphanumeric identifier (no spaces). The same shape the
# inner field-inference step applies to ids named in the request text.
_RECORD_ID_SHAPE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")

# Aliases accepted for the target record, first supplied wins.
_RECORD_ALIASES = ("record_id", "record_hint", "contact_id", "deal_id")


class CallerFieldError(ValueError):
    """A caller-supplied field failed its contract. Carries the field name only."""


# Contract field names that may be echoed into a refusal message. Any other
# input_context key is caller-controlled text, so its spot in the reported
# path shows a placeholder — a hostile field NAME is never echoed either.
_KNOWN_CONTEXT_FIELDS = frozenset(_RECORD_ALIASES)


def _find_instruction_override(value: object, path: str = "input_context") -> "str | None":
    """Depth-first scan of every decoded string in the mapping — keys included.

    Returns the path of the first string carrying an instruction-override
    directive, or None. The walk runs on the PARSED mapping, so JSON escaping
    cannot smuggle a phrase past it, and it covers undeclared keys too: the
    screen must hold on what the caller SENT, not only on what the contract
    keeps. Path components outside the declared contract are masked, so the
    returned path is always safe to name in an error message.
    """
    if isinstance(value, str):
        return path if contains_instruction_override(value) else None
    if isinstance(value, dict):
        for key, item in value.items():
            safe_key = key if isinstance(key, str) and key in _KNOWN_CONTEXT_FIELDS else "<unrecognised-field>"
            key_path = f"{path}.{safe_key}"
            if isinstance(key, str) and contains_instruction_override(key):
                return key_path
            found = _find_instruction_override(item, key_path)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _find_instruction_override(item, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _validate_record_hint(raw: object, field: str) -> str:
    """Validate a target-record reference: an inert record identifier."""
    if not isinstance(raw, str):
        raise CallerFieldError(f"'{field}' must be a string")
    text = raw.strip()
    if not text:
        raise CallerFieldError(f"'{field}' must not be empty")
    if not _RECORD_ID_SHAPE_RE.match(text):
        raise CallerFieldError(f"'{field}' is not a valid record id")
    return text


def validate_caller_fields(input_context: object) -> "dict[str, object]":
    """Validate the caller contract. Raises CallerFieldError on any breach."""
    if input_context in (None, {}):
        return {}
    if not isinstance(input_context, dict):
        raise CallerFieldError("'input_context' must be a mapping")

    fields: dict[str, object] = {}

    supplied = [k for k in _RECORD_ALIASES if input_context.get(k) is not None]
    if supplied:
        fields["record_hint"] = _validate_record_hint(input_context[supplied[0]], supplied[0])

    return fields


class PreProcessNode(FunctionNode):
    """Validate the caller contract and shape the request for the inner graph."""

    # The outer backbone's SINGLE external trust gate. A real caller enters at
    # VERIFIED_EXTERNAL and the inner Sansan call runs under this same
    # (unelevated) context, so the external gate lives HERE, not on the inner
    # API node. An under-trusted (ANONYMOUS) caller is denied at this gate
    # before any call.
    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        user_input = state.get("user_input", "")
        input_context = state.get("input_context", {})  # read-only

        if not isinstance(user_input, str) or not user_input.strip():
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["PreProcessNode: user_input is empty or missing"],
            }

        # ── Instruction-override screen (template-owned, fail CLOSED) ────────
        # Runs on BOTH caller text channels before anything is carried
        # forward: a refusal leaves no validated_input, no record_hint and no
        # caller_fields for any downstream node. The screen lives in this
        # node's own execute() path — calling execute() directly still
        # refuses, so the guarantee does not depend on any framework gate
        # being present or configured on.
        if contains_instruction_override(user_input):
            emit_trace_event(
                "pre_process_validation_failed",
                {"reason": "instruction_override", "where": "user_input"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["PreProcessNode: request refused - instruction-override content in user_input"],
            }

        override_path = _find_instruction_override(input_context) if isinstance(input_context, (dict, list)) else None
        if override_path is not None:
            emit_trace_event(
                "pre_process_validation_failed",
                {"reason": "instruction_override", "where": override_path},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [f"PreProcessNode: request refused - instruction-override content in {override_path}"],
            }

        try:
            caller_fields = validate_caller_fields(input_context)
        except CallerFieldError as exc:
            # Fail closed, naming the field only - the rejected value is never
            # echoed into the log.
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [f"PreProcessNode: rejected caller input - {exc}"],
            }

        # Strip markup + cap length before serialization.
        sanitized_input = sanitize_query(user_input.strip())

        record_hint = str(caller_fields.get("record_hint", ""))
        validated_input = json.dumps({"text": sanitized_input, "record_hint": record_hint})

        # Audit the shaped request - hint presence only, never the text.
        emit_trace_event(
            "pre_process_complete",
            {
                "has_record_hint": bool(record_hint),
                "caller_fields": sorted(caller_fields),
            },
            state,
        )

        return {
            "validated_input": validated_input,
            "record_hint": record_hint,
            # Stored as a JSON string (the state contract keeps every value
            # msgpack-safe); the graph node reads it back at the boundary.
            "caller_fields": to_json(caller_fields),
            "status": AgentStatus.SUCCESS.value,
        }
