# CMN-C2-274 - Unit tests: inner SansanWorkflowGraph (BaseGraph) contract.
# Adapted from a sibling tool-calling golden suite. The compiled outer path is
# exercised end-to-end by tests/proof_of_boundary/test_pb_invoke_order.py; this
# module unit-checks the inner graph's identity, config forwarding, routing,
# output contract, and a direct inner invoke on the network-free v1 stub.

from langgraph.graph import END

from framework.schemas.agent_status import AgentStatus

from src.graph.domain_workflow_graph import SansanWorkflowGraph
from src.schemas.state import State, from_json


def _graph(config=None):
    return SansanWorkflowGraph(config=config or {})


def test_inner_graph_identity():
    g = _graph()
    assert g.name == "sansan_contact_deal_workflow"
    assert g.state_schema is State


def test_extra_initial_state_injects_sansan_config_as_json():
    g = _graph({"configurable": {"sansan": {"base_url": "https://sansan.example.test/v3"}}})
    extra = g._extra_initial_state()
    # State contract: forwarded as a JSON string, not a native dict.
    assert isinstance(extra["sansan_config"], str)
    assert from_json(extra["sansan_config"], {}) == {"base_url": "https://sansan.example.test/v3"}


def test_extra_initial_state_without_sansan_section_omits_config():
    extra = _graph()._extra_initial_state()
    assert "sansan_config" not in extra


def test_extra_initial_state_seeds_caller_contract_from_bridge():
    """The framework does not forward input_context into a subgraph; the
    validated caller contract crosses on the bridge and is seeded here."""
    from src.graph.context_bridge import set_caller_input_context

    set_caller_input_context({"record_hint": "sc-4242"})
    try:
        extra = _graph()._extra_initial_state()
        assert extra["input_context"] == {"record_hint": "sc-4242"}
    finally:
        set_caller_input_context(None)


def test_extra_initial_state_defaults_to_empty_caller_contract():
    from src.graph.context_bridge import set_caller_input_context

    set_caller_input_context(None)
    assert _graph()._extra_initial_state()["input_context"] == {}


def test_route_error_ends_graph():
    g = _graph()
    assert g.route({"status": AgentStatus.ERROR.value}) == END
    assert g.route({"status": AgentStatus.SUCCESS.value}) == "confirm"


def test_get_output_surfaces_record_fields():
    g = _graph()
    out = g.get_output(
        {
            "result": {"record_id": "sc-1001", "record_ref": "sansan://contacts/sc-1001", "confirmation": "ok"},
            "status": AgentStatus.SUCCESS.value,
            "intent": "lookup_contact",
            "target_id": "sc-1001",
            "record_id": "sc-1001",
            "record_ref": "sansan://contacts/sc-1001",
            "record_label": "Contact sc-1001",
            "confirmation": "ok",
            "sansan_payload": "{}",
            "redaction_flags": "[]",
            "error_log": [],
            "trace_id": "tr",
            "correlation_id": "co",
            "node_history": ["ValidateInputNode", "ConfirmNode"],
        }
    )
    assert out["status"] == AgentStatus.SUCCESS.value
    assert out["intent"] == "lookup_contact"
    assert out["record_ref"] == "sansan://contacts/sc-1001"
    assert out["confirmation"] == "ok"
    assert out["output"] == {"record_id": "sc-1001", "record_ref": "sansan://contacts/sc-1001", "confirmation": "ok"}


def test_get_output_carries_error_log():
    g = _graph()
    out = g.get_output({"status": AgentStatus.ERROR.value, "error_log": ["boom"], "confirmation": ""})
    assert out["status"] == AgentStatus.ERROR.value
    assert out["error_log"] == ["boom"]


def test_inner_graph_compiles():
    g = _graph()
    g.compile()
    assert g._compiled is not None


def test_inner_invoke_lookup_on_v1_stub():
    """Direct inner invoke (default ANONYMOUS ctx - every inner node is
    ANONYMOUS): validate -> classify -> infer -> call(stub) -> confirm."""
    g = _graph({"configurable": {"sansan": {"base_url": "https://api.sansan.com/v3"}}})
    g.compile()
    result = g.invoke(
        user_input="Look up the business card contact with contact id sc-1001 and summarize the record on file."
    )
    assert result["status"] == AgentStatus.SUCCESS.value
    assert result["record_id"] == "sc-1001"
    assert result["record_ref"] == "sansan://contacts/sc-1001"
    assert result["intent"] == "lookup_contact"
    assert result["confirmation"]
    history = result.get("node_history", [])
    assert history == [
        "ValidateInputNode",
        "ClassifyIntentNode",
        "InferSansanFieldsNode",
        "CallSansanApiNode",
        "ConfirmNode",
    ]
