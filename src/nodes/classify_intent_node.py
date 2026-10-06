"""AgentCore Platform v1.0 - inner workflow Step 2: ClassifyIntent.

Classifies the (redacted) request on two axes - operation (lookup / create /
update) x record type (contact / deal) - into one of lookup_contact /
create_contact / update_contact / lookup_deal / create_deal / update_deal.

An optional per-invocation LLM call (Azure OpenAI) enhances the deterministic
keyword classification below; any failure - missing secret, API error,
malformed response - degrades silently back to the keyword result, so the
template stays testable and runnable without a live LLM (see docs/02_design.md,
v1 Implementation Note / LLM Enhancement Note).
Low-confidence / unknown operation falls back to the read-only "lookup"
default with a note - never a write.
"""

from typing import Any

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from shared.utils.llm_json import extract_json_object

from src.services.llm_enhance import resolve_llm

_VALID_RECORD_TYPES = ("contact", "deal")

_CLASSIFY_SYSTEM_PROMPT = (
    "Classify the user's Sansan CRM request. Reply with ONLY a JSON object: "
    '{"operation": "lookup"|"create"|"update", "record_type": "contact"|"deal"}. '
    "operation=lookup for read-only requests; create for adding a new record; "
    "update for changing an existing record. record_type=deal for sales "
    "opportunities/pipeline; contact for business-card contacts (the default)."
)

_VALID_OPS = ("lookup", "create", "update")

# Deterministic operation keyword signals (checked in priority order, writes
# first so a "update the deal then show it" style request classifies as the write).
_OP_KEYWORDS = (
    (
        "update",
        (
            "update",
            "change",
            "edit",
            "correct",
            "amend",
            "revise",
            "move the deal",
            "advance",
            "progress",
            "set the",
            "set her",
            "set his",
            "更新",
            "変更",
            "修正",
            "移動",
            "進行",
        ),
    ),
    (
        "create",
        (
            "create",
            "register",
            "onboard",
            "new contact",
            "new deal",
            "add a contact",
            "add a deal",
            "add a new",
            "log a deal",
            "登録",
            "追加",
            "新規",
            "作成",
            "起票",
        ),
    ),
    (
        "lookup",
        (
            "look up",
            "lookup",
            "find",
            "show",
            "get",
            "fetch",
            "retrieve",
            "search",
            "what is",
            "who is",
            "summarize",
            "contact for",
            "deal for",
            "details of",
            "照会",
            "検索",
            "参照",
            "確認",
        ),
    ),
)

# Record-type keyword signals: any deal signal -> deal; otherwise the
# business-card contact default (Sansan's primary record type).
_DEAL_WORDS = ("deal", "opportunity", "pipeline", "negotiation", "商談", "案件", "受注", "取引")


class ClassifyIntentNode(FunctionNode):
    """Classify the request into a Sansan contact/deal operation intent."""

    # Inner domain node, read-only classification of already-redacted text -
    # the external gate lives on the outer backbone pre_process.
    required_trust_level = TrustLevel.ANONYMOUS

    def __init__(self, llm: Any = None) -> None:
        # Test-double injection seam only - register_nodes() always constructs
        # this with no arguments; production always resolves fresh per call.
        self._llm = llm

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        text = state.get("validated_input", "") or ""
        if not text:
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["ClassifyIntentNode: missing validated_input"],
            }

        llm_result = self._classify_via_llm(state, text)
        if llm_result is not None:
            op, record_type, source = llm_result
        else:
            op = self._classify_op_via_keywords(text)
            record_type = self._classify_record_type(text)
            source = "heuristic"

        note: list[str] = []
        if op not in _VALID_OPS:
            note = [
                "ClassifyIntentNode: low-confidence classification, " f"defaulted to lookup_{record_type} (read-only)"
            ]
            op = "lookup"

        intent = f"{op}_{record_type}"

        # Audit the classification decision - intent label + source only, never the text.
        emit_trace_event(
            "classify_intent_complete",
            {"intent": intent, "defaulted": bool(note), "source": source},
            state,
        )

        result: dict[str, Any] = {"intent": intent, "status": AgentStatus.SUCCESS.value}
        if note:
            result["error_log"] = note  # non-fatal note; status stays SUCCESS
        return result

    # -- LLM enhancement (optional; any failure degrades to the heuristic) ----

    def _classify_via_llm(self, state: dict[str, Any], text: str) -> "tuple[str, str, str] | None":
        """Return (op, record_type, "llm") from an LLM call, or None on any failure."""
        llm = resolve_llm(self._llm, state)
        if llm is None:
            return None
        try:
            response = llm.complete(
                [
                    {"role": "system", "content": _CLASSIFY_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ]
            )
            parsed = extract_json_object(response.get("content"))
            op = parsed.get("operation")
            record_type = parsed.get("record_type")
            if op not in _VALID_OPS or record_type not in _VALID_RECORD_TYPES:
                return None
            return op, record_type, "llm"
        except Exception:
            return None

    # -- classification (deterministic fallback) ------------------------------

    def _classify_op_via_keywords(self, text: str) -> str:
        low = text.lower()
        for op, words in _OP_KEYWORDS:
            if any(w in low for w in words):
                return op
        # No signal at all: fall through to the read-only default via the
        # _VALID_OPS guard in execute() (returns a sentinel outside the set).
        return "unknown"

    def _classify_record_type(self, text: str) -> str:
        low = text.lower()
        if any(w in low for w in _DEAL_WORDS):
            return "deal"
        return "contact"
