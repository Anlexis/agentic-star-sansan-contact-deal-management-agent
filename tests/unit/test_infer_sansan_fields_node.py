# CMN-C2-274 - Unit tests: InferSansanFieldsNode (inner Step 3)
# Adapted from a sibling tool-calling golden suite (Kaonavi -> Sansan).
#
# Canon: invoked via node(state) (BaseNode.__call__ routes the full security
# pipeline); inner domain node -> caller_trust_level = TrustLevel.ANONYMOUS.value.
# Positive payloads are PII-free: the framework PII mask rewrites Title-Case
# bigrams in validated_input, so quoted display names use a single-word name.

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.infer_sansan_fields_node import InferSansanFieldsNode
from src.schemas.state import from_json


@pytest.fixture(autouse=True)
def _mute_audit(monkeypatch):
    monkeypatch.setattr("src.nodes.infer_sansan_fields_node.emit_trace_event", lambda *a, **k: None)


def _state(text: str, intent: str = "lookup_contact", record_hint: str = "", **overrides) -> dict:
    state = {
        "validated_input": text,
        "intent": intent,
        "record_hint": record_hint,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "correlation_id": "infer-fields-test",
        "node_history": [],
        "error_log": [],
        "execution_time": {},
    }
    state.update(overrides)
    return state


class TestInferSansanFieldsNode:
    def setup_method(self):
        self.node = InferSansanFieldsNode()

    def test_lookup_extracts_id_from_text(self):
        result = self.node(
            _state("Look up the business card contact with contact id sc-1001 and summarize the record on file.")
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["target_id"] == "sc-1001"
        # State contract: sansan_payload is stored as a JSON string, not a native dict.
        assert isinstance(result["sansan_payload"], str)
        assert from_json(result["sansan_payload"], {}) == {"record_id": "sc-1001"}

    def test_deal_id_extracted_from_text(self):
        result = self.node(_state("Look up the deal with deal id d-42", intent="lookup_deal"))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["target_id"] == "d-42"
        assert from_json(result["sansan_payload"], {}) == {"record_id": "d-42"}

    def test_id_shaped_hint_used_when_text_has_no_id(self):
        result = self.node(_state("Show the current contact summary", record_hint="sc-A123"))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["target_id"] == "sc-A123"
        assert from_json(result["sansan_payload"], {}) == {"record_id": "sc-A123"}

    def test_create_builds_record_data_payload(self):
        # Field values stay lower-case: the framework name mask rewrites
        # Title-Case word pairs even ACROSS newlines ("Sales\nGrade" would be
        # masked before execute() sees the text).
        text = 'Register a new contact named "Alice"\nDepartment: sales\nTitle: manager'
        result = self.node(_state(text, intent="create_contact", record_hint="sc-9002"))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["target_id"] == "sc-9002"
        assert result["record_label"] == "Alice"
        payload = from_json(result["sansan_payload"], {})
        record = payload["record_data"][0]
        assert record["id"] == "sc-9002"
        assert record["name"] == "Alice"
        assert {"name": "Department", "values": ["sales"]} in record["custom_fields"]
        # "Title" is a name-key alias, not a custom record field.
        assert all(f["name"] != "Title" for f in record["custom_fields"])

    def test_update_builds_record_data_payload(self):
        text = "Update the record for contact id sc-1001\nDepartment: marketing"
        result = self.node(_state(text, intent="update_contact"))
        assert result["status"] == AgentStatus.SUCCESS.value
        payload = from_json(result["sansan_payload"], {})
        record = payload["record_data"][0]
        assert record["id"] == "sc-1001"
        assert {"name": "Department", "values": ["marketing"]} in record["custom_fields"]

    def test_unresolved_id_left_empty_never_invented(self):
        result = self.node(_state("Show the contact summary for the flagged record"))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["target_id"] == ""
        assert from_json(result["sansan_payload"], {}) == {"record_id": ""}

    def test_non_id_shaped_hint_left_unresolved(self):
        result = self.node(_state("Show the contact summary", record_hint="not a valid id!"))
        assert result["target_id"] == ""

    def test_missing_input_errors(self):
        result = self.node(_state(""))
        assert result["status"] == AgentStatus.ERROR.value
        assert result["error_log"]


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


class TestInferSansanFieldsNodeLLMEnhancement:
    """LLM enhancement path - see module docstring. No real network call in this suite.

    target_id resolution never routes through the LLM (module docstring), so
    every case here still resolves target_id via the regex+hint path even
    when the LLM enhances record_label/custom_fields.
    """

    def test_llm_response_overrides_the_heuristic_on_well_formed_json(self):
        # Freeform phrasing the "Key: value" regex would not parse as a field.
        node = InferSansanFieldsNode(
            llm=_FakeLLM('{"record_label": "Alice", "custom_fields": [{"name": "Department", "values": ["sales"]}]}')
        )
        result = node(_state("contact id sc-9002, met with Alice from sales", intent="create_contact"))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["target_id"] == "sc-9002"
        assert result["record_label"] == "Alice"
        payload = from_json(result["sansan_payload"], {})
        record = payload["record_data"][0]
        assert {"name": "Department", "values": ["sales"]} in record["custom_fields"]

    def test_prose_wrapped_json_still_parses(self):
        node = InferSansanFieldsNode(
            llm=_FakeLLM('Here you go:\n```json\n{"record_label": "Alice", "custom_fields": []}\n```')
        )
        result = node(_state("contact id sc-1001, spoke with Alice", intent="update_contact"))
        assert result["record_label"] == "Alice"

    def test_malformed_response_falls_back_to_heuristic(self):
        node = InferSansanFieldsNode(llm=_FakeLLM("not json at all"))
        text = 'Register a new contact named "Alice"\nDepartment: sales'
        result = node(_state(text, intent="create_contact", record_hint="sc-9002"))
        assert result["record_label"] == "Alice"

    def test_wrong_shape_response_falls_back_to_heuristic(self):
        # Well-formed JSON but custom_fields is the wrong shape (not a list of {name, values}).
        node = InferSansanFieldsNode(llm=_FakeLLM('{"record_label": "Alice", "custom_fields": "sales"}'))
        text = 'Register a new contact named "Alice"\nDepartment: sales'
        result = node(_state(text, intent="create_contact", record_hint="sc-9002"))
        payload = from_json(result["sansan_payload"], {})
        record = payload["record_data"][0]
        assert {"name": "Department", "values": ["sales"]} in record["custom_fields"]

    def test_llm_raising_falls_back_to_heuristic(self):
        node = InferSansanFieldsNode(llm=_FakeLLM(raises=True))
        result = node(_state("Look up the deal with deal id d-42", intent="lookup_deal"))
        assert result["target_id"] == "d-42"

    def test_no_llm_injected_and_no_secret_bound_falls_back_to_heuristic(self):
        # The real production shape in any environment without a configured key.
        node = InferSansanFieldsNode()
        result = node(_state("Look up the business card contact with contact id sc-1001.", intent="lookup_contact"))
        assert result["target_id"] == "sc-1001"

    def test_empty_input_never_calls_the_llm(self):
        fake = _FakeLLM('{"record_label": "Alice", "custom_fields": []}')
        node = InferSansanFieldsNode(llm=fake)
        result = node(_state(""))
        assert result["status"] == AgentStatus.ERROR.value
        assert fake.calls == 0

    def test_llm_never_supplies_target_id_even_when_valid_shaped(self):
        # target_id resolution is regex+hint-only by design; an LLM cannot
        # inject or override it even indirectly, because the prompt/parser
        # never asks for or accepts one.
        node = InferSansanFieldsNode(llm=_FakeLLM('{"record_label": "Alice", "custom_fields": []}'))
        result = node(_state("met with Alice, no id mentioned", intent="create_contact"))
        assert result["target_id"] == ""

    def test_source_reported_on_audit_event(self, monkeypatch):
        events = []
        monkeypatch.setattr(
            "src.nodes.infer_sansan_fields_node.emit_trace_event",
            lambda *args, **kwargs: events.append(args),
        )
        node = InferSansanFieldsNode(llm=_FakeLLM('{"record_label": "Alice", "custom_fields": []}'))
        node(_state("contact id sc-1001, met with Alice", intent="update_contact"))
        payloads = {args[0]: args[1] for args in events}
        assert payloads["infer_sansan_fields_complete"]["source"] == "llm"
