# CMN-C2-274 - Unit tests: ClassifyIntentNode (inner Step 2)
# Adapted from a sibling tool-calling golden suite (Kaonavi -> Sansan).
# Intents: two deterministic keyword axes - operation (lookup / create /
# update, writes-first priority, read-only default) x record type (contact /
# deal, contact default) -> lookup_contact / create_contact / update_contact /
# lookup_deal / create_deal / update_deal (v1 - no LLM).
#
# Canon: invoked via node(state) (BaseNode.__call__ routes the full security
# pipeline); inner domain node -> caller_trust_level = TrustLevel.ANONYMOUS.value.

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.classify_intent_node import ClassifyIntentNode


@pytest.fixture(autouse=True)
def _mute_audit(monkeypatch):
    monkeypatch.setattr("src.nodes.classify_intent_node.emit_trace_event", lambda *a, **k: None)


def _state(text: str) -> dict:
    return {
        "validated_input": text,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "correlation_id": "classify-intent-test",
        "node_history": [],
        "error_log": [],
        "execution_time": {},
    }


class TestClassifyIntentNode:
    def setup_method(self):
        self.node = ClassifyIntentNode()

    def test_keyword_lookup_contact(self):
        result = self.node(_state("Look up the business card contact with contact id sc-1001."))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["intent"] == "lookup_contact"

    def test_keyword_create_contact(self):
        result = self.node(_state("Register a new business card contact with contact id sc-9002"))
        assert result["intent"] == "create_contact"

    def test_keyword_update_contact(self):
        result = self.node(_state("Update the record for contact id sc-1001 with the new title"))
        assert result["intent"] == "update_contact"

    def test_keyword_lookup_deal(self):
        # Any deal signal flips the record-type axis to deal.
        result = self.node(_state("Look up the deal with deal id d-42"))
        assert result["intent"] == "lookup_deal"

    def test_keyword_update_deal(self):
        result = self.node(_state("Advance the deal with deal id d-42 to the negotiation stage"))
        assert result["intent"] == "update_deal"

    def test_write_keyword_wins_over_lookup(self):
        # Priority order is writes-first: an "update ... then show it" style
        # request classifies as the write, never the read.
        result = self.node(_state("Update the title for contact id sc-1001 and show the record"))
        assert result["intent"] == "update_contact"

    def test_no_signal_defaults_to_readonly_lookup(self):
        result = self.node(_state("please handle this for the team"))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["intent"] == "lookup_contact"
        # Non-fatal low-confidence note travels in error_log; status stays SUCCESS.
        assert any("defaulted to lookup_contact" in entry for entry in result.get("error_log", []))

    def test_empty_input_errors(self):
        result = self.node(_state(""))
        assert result["status"] == AgentStatus.ERROR.value
        assert result["error_log"]

    def test_audit_emits_intent_label_only(self, monkeypatch):
        events = []
        monkeypatch.setattr(
            "src.nodes.classify_intent_node.emit_trace_event",
            lambda *args, **kwargs: events.append(args),
        )
        self.node(_state("Look up the business card contact with contact id sc-1001."))
        payloads = {args[0]: args[1] for args in events}
        # Emit-spy asserts on the payload (args[1]) - the label, never the text.
        assert payloads["classify_intent_complete"]["intent"] == "lookup_contact"
        assert payloads["classify_intent_complete"]["defaulted"] is False
        assert payloads["classify_intent_complete"]["source"] == "heuristic"


class _FakeLLM:
    """Test double: exposes complete(messages) -> {"content": ...} like AzureOpenAIClient."""

    def __init__(self, content=None, raises=False):
        self.content = content
        self.raises = raises
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        if self.raises:
            raise RuntimeError("simulated API error")
        return {"content": self.content}


class TestClassifyIntentNodeLLMEnhancement:
    """LLM enhancement path - see module docstring. No real network call in this suite."""

    def test_llm_response_overrides_the_heuristic_on_well_formed_json(self):
        # Text alone would keyword-classify as lookup_contact (no write signal);
        # the LLM call overrides it with create_deal.
        node = ClassifyIntentNode(llm=_FakeLLM('{"operation": "create", "record_type": "deal"}'))
        result = node(_state("please handle this for the team"))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["intent"] == "create_deal"

    def test_prose_wrapped_json_still_parses(self):
        node = ClassifyIntentNode(
            llm=_FakeLLM('Sure, here is the classification:\n```json\n{"operation": "update", "record_type": '
                         '"contact"}\n```')
        )
        result = node(_state("please handle this for the team"))
        assert result["intent"] == "update_contact"

    def test_malformed_response_falls_back_to_heuristic(self):
        node = ClassifyIntentNode(llm=_FakeLLM("not json at all"))
        result = node(_state("Look up the deal with deal id d-42"))
        assert result["intent"] == "lookup_deal"

    def test_wrong_shape_response_falls_back_to_heuristic(self):
        # Well-formed JSON, but not the expected keys/enum values.
        node = ClassifyIntentNode(llm=_FakeLLM('{"operation": "delete", "record_type": "contact"}'))
        result = node(_state("Look up the deal with deal id d-42"))
        assert result["intent"] == "lookup_deal"

    def test_llm_raising_falls_back_to_heuristic(self):
        node = ClassifyIntentNode(llm=_FakeLLM(raises=True))
        result = node(_state("Update the record for contact id sc-1001"))
        assert result["intent"] == "update_contact"

    def test_no_llm_injected_and_no_secret_bound_falls_back_to_heuristic(self):
        # The real production shape in any environment without a configured key.
        node = ClassifyIntentNode()
        result = node(_state("Register a new business card contact with contact id sc-9002"))
        assert result["intent"] == "create_contact"

    def test_empty_input_never_calls_the_llm(self):
        fake = _FakeLLM('{"operation": "lookup", "record_type": "contact"}')
        node = ClassifyIntentNode(llm=fake)
        result = node(_state(""))
        assert result["status"] == AgentStatus.ERROR.value
        assert fake.calls == 0

    def test_source_reported_on_audit_event(self, monkeypatch):
        events = []
        monkeypatch.setattr(
            "src.nodes.classify_intent_node.emit_trace_event",
            lambda *args, **kwargs: events.append(args),
        )
        node = ClassifyIntentNode(llm=_FakeLLM('{"operation": "lookup", "record_type": "deal"}'))
        node(_state("please handle this for the team"))
        payloads = {args[0]: args[1] for args in events}
        assert payloads["classify_intent_complete"]["source"] == "llm"
