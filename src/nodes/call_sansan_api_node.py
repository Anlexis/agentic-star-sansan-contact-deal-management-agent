"""AgentCore Platform v1.0 - inner workflow Step 4: CallSansanApi (tool side-effect).

Performs the lookup/create/update call against the Sansan REST API contact and
deal collections via src/services/sansan_client.py.

Security posture:
  Trust: required_trust_level = ANONYMOUS. The single external trust gate lives on
       the OUTER backbone pre_process (VERIFIED_EXTERNAL), not on this inner node.
       GraphNode.execute() passes the caller's InvocationContext into the inner
       subgraph UNCHANGED (no trust elevation), so a real external caller runs
       this call under its own VERIFIED_EXTERNAL context; declaring INTERNAL here
       would deny that already-gated external caller before the call ever
       runs. The node therefore stays ANONYMOUS.
  Secrets: the integration key is read via ctx.secrets.get("SANSAN_API_KEY")
       (InvocationContext.from_state(state)) - never os.environ, never stored in
       state. v1 note: while the deterministic NETWORK-FREE stub transport is
       active a missing key is tolerated (a sentinel placeholder is used - it
       is never sent anywhere because no request leaves the process); with a
       LIVE transport injected, a missing key is a hard status=error - a real
       API is never called unauthenticated.
  Audit: emit_trace_event() is called on the success path - a side-effect against
       an external sales/contact system; HTTP 4xx/5xx surfaces as status=error +
       error_log (no silent pass).

Configuration: this node takes NO constructor arguments (SDK v1 nodes are
no-arg). Sansan settings (base_url) arrive as the JSON `sansan_config` state
field - injected by the inner graph's _extra_initial_state() from the
config/config.yaml section forwarded by SansanWorkflowGraphNode._parent_config() - or via the
optional `config["configurable"]["sansan"]` argument for direct invocation.
The client is constructed locally per call (no module-global mutation).
"""

from typing import Any

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import from_json
from src.services.sansan_client import SansanApiError, SansanClient

_SECRET_KEY = "SANSAN_API_KEY"
# Placeholder handed to the network-free stub transport when no secret is
# provisioned. Never sent over any network (the stub performs no I/O) and never
# written to state or logs.
_STUB_PLACEHOLDER = "stub-transport-no-credential"


class CallSansanApiNode(FunctionNode):
    """Look up / create / update a Sansan contact or deal record via the REST API."""

    # The external trust gate is enforced UPSTREAM on the outer backbone
    # pre_process (VERIFIED_EXTERNAL). This inner node runs under the caller's
    # UNELEVATED context (GraphNode does not elevate trust for the subgraph), so
    # it must stay ANONYMOUS - declaring INTERNAL would deny a real external
    # caller before the call runs.
    required_trust_level = TrustLevel.ANONYMOUS

    def execute(self, state: dict[str, Any], config: "dict[str, Any] | None" = None) -> dict[str, Any]:
        payload = from_json(state.get("sansan_payload"), None)
        if not payload:
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["CallSansanApiNode: missing sansan_payload"],
            }

        intent = state.get("intent", "lookup_contact") or "lookup_contact"
        op, _, record_type = intent.partition("_")
        if record_type not in ("contact", "deal"):
            record_type = "contact"

        # Settings: config section from state (graph-injected), overridable via
        # an explicit config["configurable"]["sansan"] for direct invocation.
        # Merged into a LOCAL dict - module globals are never mutated.
        settings = dict(from_json(state.get("sansan_config"), {}) or {})
        override = ((config or {}).get("configurable") or {}).get("sansan") or {}
        settings.update(override)

        # Client built locally per call; with no injected transport it uses the
        # deterministic NETWORK-FREE v1 stub (documented limitation, docs/02).
        base_url = str(settings.get("base_url", "") or "").strip()
        client = SansanClient(base_url=base_url) if base_url else SansanClient()

        # Key from the bound secret provider - never os.environ / state.
        ctx = InvocationContext.from_state(state)
        api_key = ctx.secrets.get(_SECRET_KEY)
        if api_key is None:
            if client.uses_stub_transport:
                # v1 stub limitation: no request leaves the process, so run with
                # a non-credential placeholder (see module docstring).
                api_key = _STUB_PLACEHOLDER
            else:
                return {
                    "status": AgentStatus.ERROR.value,
                    "error_log": [
                        f"CallSansanApiNode: secret {_SECRET_KEY} unavailable - "
                        "refusing to call a live transport unauthenticated"
                    ],
                }

        target_id = state.get("target_id", "") or str(payload.get("record_id", "") or "")

        try:
            if op == "lookup":
                if not target_id:
                    return {
                        "status": AgentStatus.ERROR.value,
                        "error_log": [f"CallSansanApiNode: unresolved record id - cannot look up {record_type}"],
                    }
                resp = client.find_record(record_type, target_id, api_key) or {}
                records = resp.get("data") or []
                if not records:
                    # The reason names the record TYPE (a closed-set label),
                    # never the id: error_log is operator-side (the caller
                    # sees only post_process's closed-set envelope), but it is
                    # the audit trail and must not carry the record evidence.
                    return {
                        "status": AgentStatus.ERROR.value,
                        "error_log": [f"CallSansanApiNode: no matching {record_type} record found"],
                    }
                record = records[0]
                record_id = str(record.get("id", "")) or target_id
                record_label = state.get("record_label", "") or str(record.get("name", ""))
            elif op in ("create", "update"):
                if op == "update" and not target_id:
                    return {
                        "status": AgentStatus.ERROR.value,
                        "error_log": [f"CallSansanApiNode: unresolved record id - cannot update {record_type}"],
                    }
                if op == "create":
                    resp = client.create_record(record_type, payload, api_key) or {}
                else:
                    resp = client.update_record(record_type, payload, api_key) or {}
                record_id = str(resp.get("record_id", "")) or target_id
                record_label = state.get("record_label", "")
            else:
                return {
                    "status": AgentStatus.ERROR.value,
                    "error_log": [f"CallSansanApiNode: unknown intent '{intent}'"],
                }
        except SansanApiError as exc:
            # HTTP status only. A live tenant's error body is unbounded
            # third-party text that can echo the record it refused (name,
            # company, email). error_log is operator-side, but it is the audit
            # trail - so the closed-set signal travels, the body does not.
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [f"CallSansanApiNode: Sansan API error {exc.status_code}"],
            }
        except Exception as exc:  # transport failure - no silent pass
            # Exception TYPE only, for the same reason: a transport error
            # string can carry the request URL and the record id.
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [f"CallSansanApiNode: Sansan call failed ({type(exc).__name__})"],
            }

        collection = "deals" if record_type == "deal" else "contacts"
        record_ref = f"sansan://{collection}/{record_id}" if record_id else ""

        # Audit the tool side-effect - intent + presence signals only,
        # never contact/deal content or credentials.
        emit_trace_event(
            "call_sansan_api_complete",
            {
                "intent": intent,
                "has_record_id": bool(record_id),
                "stub_transport": client.uses_stub_transport,
            },
            state,
        )

        return {
            "record_id": record_id,
            "record_ref": record_ref,
            "target_id": target_id or record_id,
            "record_label": record_label,
            "status": AgentStatus.SUCCESS.value,
        }
