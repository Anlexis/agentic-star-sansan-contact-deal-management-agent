"""AgentCore Platform v1.0 - inner workflow Step 3: InferSansanFields.

Extracts the record id, display name (or deal title), and "Key: value" fields
from the (redacted) request and assembles a validated Sansan REST API request
body for the classified intent. The record id is taken only from an explicit
id in the text or the caller-supplied record_hint - an unresolved id is left
empty rather than invented (risk mitigation: never touch the wrong contact or
deal record; the executor surfaces the miss as status=error).

An optional per-invocation LLM call (Azure OpenAI) enhances record_label /
custom_fields extraction for freer-form phrasing than the "Key: value" regex
handles; any failure - missing secret, API error, malformed response -
degrades silently back to the regex result (see docs/02_design.md
"v1 Implementation Note" / "LLM Enhancement Note").

target_id resolution is intentionally NEVER routed through the LLM: it stays
regex-plus-caller-hint-only (unchanged from v1). That is the safety-critical
"never invent a target" path (risk mitigation above), record_hint is a
caller-supplied side channel the LLM prompt does not even see, and a
freeform-extracted id is exactly the kind of value this rule exists to keep
out of a write.
"""

import re
from typing import Any

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from shared.utils.llm_json import extract_json_object

from src.schemas.state import to_json
from src.services.llm_enhance import resolve_llm

_INFER_SYSTEM_PROMPT = (
    "Extract Sansan CRM record fields from the user's request. Reply with ONLY "
    'a JSON object: {"record_label": string or null, "custom_fields": '
    '[{"name": string, "values": [string]}]}. record_label is the contact\'s '
    "display name or the deal's title if stated, else null. custom_fields "
    "lists any other named attributes mentioned (e.g. title, company, stage) "
    "- do not include an id/record-number field, that is handled separately. "
    "Return {} if nothing can be extracted. Do not guess or invent values."
)

# A Sansan record id: alphanumeric identifier (no spaces).
_ID_SHAPE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")
# Explicit id mention in the request text, EN or JA
# ("contact id sc-1001" / "deal id: d-42" / "名刺ID sc-1001").
_ID_IN_TEXT_RE = re.compile(
    r"(?:contact|deal|record|card)\s+(?:id|code|number)\s*[:#]?\s*([A-Za-z0-9][A-Za-z0-9_-]{0,39})"
    r"|(?:名刺|商談|案件|連絡先)\s*(?:ID|番号|コード)\s*[:：#]?\s*([A-Za-z0-9][A-Za-z0-9_-]{0,39})",
    re.IGNORECASE,
)
# Quoted display name / deal title: named "Foo" / titled "Foo". Curly quotes as
# \u escapes so the source stays pure ASCII (push-safe).
_NAME_QUOTED_RE = re.compile(r'(?:named|called|titled|for)\s+["“]([^"”\n]+)["”]', re.IGNORECASE)
# "Key: value" field lines (ASCII or full-width colon). CJK ranges:
# hiragana/katakana + CJK unified ideographs, as \u escapes (push-safe).
_KV_RE = re.compile(r"^\s*([A-Za-z぀-ヿ一-鿿][\w \-぀-ヿ一-鿿]{0,40})[:：]\s*(.+?)\s*$")
# Keys that are the id/name themselves, not custom record fields.
_ID_KEYS = ("id", "record id", "contact id", "deal id", "code")
_NAME_KEYS = ("name", "contact name", "title", "deal title")


class InferSansanFieldsNode(FunctionNode):
    """Extract entities and assemble the Sansan REST API request body."""

    # Inner domain node - derives fields from already-validated text; the
    # external gate lives on the outer backbone pre_process.
    required_trust_level = TrustLevel.ANONYMOUS

    def __init__(self, llm: Any = None) -> None:
        # Test-double injection seam only - register_nodes() always constructs
        # this with no arguments; production always resolves fresh per call.
        self._llm = llm

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        text = state.get("validated_input", "") or ""
        intent = state.get("intent", "lookup_contact") or "lookup_contact"
        record_hint = state.get("record_hint", "") or ""

        if not text.strip():
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["InferSansanFieldsNode: missing validated_input"],
            }

        op = intent.split("_", 1)[0]
        regex_fields = self._parse_fields(text)
        # target_id resolution NEVER routes through the LLM - see module docstring.
        target_id = self._resolve_id(text, record_hint, regex_fields)

        llm_result = self._infer_via_llm(state, text)
        if llm_result is not None:
            record_label = llm_result["record_label"] or self._resolve_label(text, regex_fields)
            fields = llm_result["fields"] or regex_fields
            source = "llm"
        else:
            record_label = self._resolve_label(text, regex_fields)
            fields = regex_fields
            source = "heuristic"

        if op in ("create", "update"):
            payload = self._build_record_data(target_id, record_label, fields)
        else:  # lookup (read-only default)
            payload = {"record_id": target_id}

        # Audit the assembled payload shape - field signals only, not content.
        emit_trace_event(
            "infer_sansan_fields_complete",
            {"intent": intent, "has_target_id": bool(target_id), "n_fields": len(fields), "source": source},
            state,
        )

        return {
            "target_id": target_id,
            "record_label": record_label,
            "sansan_payload": to_json(payload),
            "status": AgentStatus.SUCCESS.value,
        }

    # -- LLM enhancement (optional; any failure degrades to the regex result) -

    def _infer_via_llm(self, state: dict[str, Any], text: str) -> "dict[str, Any] | None":
        """Return {"record_label": str, "fields": [(k, v), ...]} from an LLM call, or None."""
        llm = resolve_llm(self._llm, state)
        if llm is None:
            return None
        try:
            response = llm.complete(
                [
                    {"role": "system", "content": _INFER_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ]
            )
            parsed = extract_json_object(response.get("content"))
            if not parsed:
                return None

            record_label = parsed.get("record_label") or ""
            if not isinstance(record_label, str):
                return None
            record_label = record_label.strip()[:100]

            fields: list[tuple[str, str]] = []
            custom_fields = parsed.get("custom_fields")
            if custom_fields is not None:
                if not isinstance(custom_fields, list):
                    return None
                for item in custom_fields:
                    if not isinstance(item, dict):
                        return None
                    name, values = item.get("name"), item.get("values")
                    if not isinstance(name, str) or not isinstance(values, list) or not values:
                        return None
                    value = values[0]
                    if not isinstance(value, str):
                        return None
                    fields.append((name, value))

            return {"record_label": record_label, "fields": fields}
        except Exception:
            return None

    # -- extraction (deterministic fallback) -----------------------------------

    def _resolve_id(self, text: str, record_hint: str, fields: "list[tuple[str, str]]") -> str:
        """Explicit id only: text mention > id-shaped hint > 'Id:' field. Never invented."""
        m = _ID_IN_TEXT_RE.search(text)
        if m:
            return m.group(1) or m.group(2) or ""
        hint = record_hint.strip()
        if hint and _ID_SHAPE_RE.match(hint):
            return hint
        for key, value in fields:
            if key.strip().lower() in _ID_KEYS and _ID_SHAPE_RE.match(value.strip()):
                return value.strip()
        return ""  # unresolved - left empty, never invented

    def _resolve_label(self, text: str, fields: "list[tuple[str, str]]") -> str:
        m = _NAME_QUOTED_RE.search(text)
        if m:
            return m.group(1).strip()[:100]
        for key, value in fields:
            if key.strip().lower() in _NAME_KEYS:
                return value.strip()[:100]
        return ""

    def _parse_fields(self, text: str) -> "list[tuple[str, str]]":
        """Return the [(key, value), ...] record fields parsed from the request lines."""
        fields: list[tuple[str, str]] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            m = _KV_RE.match(stripped)
            if m:
                fields.append((m.group(1).strip(), m.group(2).strip()))
        return fields

    # -- payload assembly (normalized Sansan record_data envelope) --------------

    def _build_record_data(
        self, target_id: str, record_label: str, fields: "list[tuple[str, str]]"
    ) -> "dict[str, Any]":
        record: dict[str, Any] = {"id": target_id}
        if record_label:
            record["name"] = record_label
        custom_fields = []
        for key, value in fields:
            if key.strip().lower() in _ID_KEYS + _NAME_KEYS:
                continue
            custom_fields.append({"name": key, "values": [value]})
        if custom_fields:
            record["custom_fields"] = custom_fields
        return {"record_data": [record]}
