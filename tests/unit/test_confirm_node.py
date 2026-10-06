# CMN-C2-274 - Unit tests: ConfirmNode (inner Step 5)
# Adapted from a sibling tool-calling golden suite (Kaonavi -> Sansan).
#
# Canon: invoked via node(state) (BaseNode.__call__ routes the full security
# pipeline); inner domain node -> caller_trust_level = TrustLevel.ANONYMOUS.value.

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.confirm_node import ConfirmNode


@pytest.fixture(autouse=True)
def _mute_audit(monkeypatch):
    monkeypatch.setattr("src.nodes.confirm_node.emit_trace_event", lambda *a, **k: None)


def _state(**overrides) -> dict:
    state = {
        "record_id": "sc-1001",
        "record_ref": "sansan://contacts/sc-1001",
        "record_label": "Contact sc-1001",
        "intent": "lookup_contact",
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "correlation_id": "confirm-test",
        "node_history": [],
        "error_log": [],
        "execution_time": {},
    }
    state.update(overrides)
    return state


class TestConfirmNode:
    def setup_method(self):
        self.node = ConfirmNode()

    def test_lookup_confirmation(self):
        result = self.node(_state())
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "Retrieved Sansan contact" in result["confirmation"]
        assert "Contact sc-1001" in result["confirmation"]
        assert "ref=sansan://contacts/sc-1001" in result["confirmation"]
        assert "id=sc-1001" in result["confirmation"]
        assert result["result"]["record_id"] == "sc-1001"
        assert result["result"]["record_ref"] == "sansan://contacts/sc-1001"

    def test_create_verb(self):
        result = self.node(
            _state(
                intent="create_contact",
                record_id="sc-9002",
                record_ref="sansan://contacts/sc-9002",
                record_label="Alice",
            )
        )
        assert "Created Sansan contact" in result["confirmation"]

    def test_update_verb(self):
        result = self.node(_state(intent="update_contact"))
        assert "Updated Sansan contact" in result["confirmation"]

    def test_deal_verbs(self):
        result = self.node(
            _state(intent="lookup_deal", record_id="d-42", record_ref="sansan://deals/d-42", record_label="Deal d-42")
        )
        assert "Retrieved Sansan deal" in result["confirmation"]
        result = self.node(
            _state(intent="update_deal", record_id="d-42", record_ref="sansan://deals/d-42", record_label="Deal d-42")
        )
        assert "Updated Sansan deal" in result["confirmation"]

    def test_unknown_intent_uses_generic_verb(self):
        result = self.node(_state(intent="mystery"))
        assert "Processed Sansan record" in result["confirmation"]

    def test_id_only_no_ref(self):
        result = self.node(_state(record_ref=""))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "id=sc-1001" in result["confirmation"]
        assert "ref=" not in result["confirmation"]

    def test_falls_back_to_record_id_when_label_missing(self):
        result = self.node(_state(record_label=""))
        assert "'sc-1001'" in result["confirmation"]

    def test_missing_record_evidence_errors(self):
        result = self.node(_state(record_id="", record_ref=""))
        assert result["status"] == AgentStatus.ERROR.value
        assert result["error_log"]
