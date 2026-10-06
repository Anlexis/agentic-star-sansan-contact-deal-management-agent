# Test Specification - CMN-C2-274 Sansan Contact & Deal Agent

## Test Strategy
- Test types: Unit (per node + service + config + inner graph) / Proof-of-Boundary
  (full outer-graph invoke, real-ASGI end-to-end, import isolation, state safety,
  server boot, HITL stub).
- Location: `tests/unit/`, `tests/proof_of_boundary/` (`tests/integration/` is an
  empty package; end-to-end backbone coverage lives in PB-6 and the `/invoke`
  E2E suite, which drive the real compiled outer graph).
- Adapted from a sibling **Kaonavi HR Profile** tool-calling golden suite,
  Kaonavi -> Sansan, updated to the current test canon.
- The Sansan call is exercised through the deterministic, network-free v1 stub
  transport (default) and through monkeypatched fake clients; no live Sansan call.
- **Trust-gate routing canon**: every per-node unit test invokes the node as
  `node(state)` - `BaseNode.__call__` routes the full security pipeline (trust
  gate -> PII mask -> `execute()` -> credential scan) - never bare
  `node.execute(state)`. State builders set `caller_trust_level =
  TrustLevel.VERIFIED_EXTERNAL.value` for PreProcessNode (the single external gate)
  and `TrustLevel.ANONYMOUS.value` for every other node. The documented
  exceptions: `CallSansanApiNode.execute(state, config=...)` config-override test
  (a 2nd argument `__call__` cannot forward), and the caller-contract /
  instruction-override screens, which are asserted through DIRECT `execute()`
  calls on purpose - the refusal proven must be the node's own, not a framework
  gate's. The trust-gate rejection test asserts on the RETURNED error dict
  (`status == AgentStatus.ERROR.value`, "trust gate denied" in `error_log`,
  execute-only keys absent) - `__call__` never raises for a trust-gate denial.
- Assertion contract (wheel `agenticstar-agentcore[anthropic]==1.0.1`): the invoke
  surface is `result["output"]` / `status` / `trace_id` / `correlation_id` /
  `node_history`, plus - on SUCCESS with record evidence only - the structured
  keys surfaced by the `get_output()` override (`intent`, `record_id`,
  `record_ref`, `record_label`, `confirmation`); status is compared to
  `AgentStatus.SUCCESS`/`.value` (lowercase `success`/`error`); the outer graph is
  called as `invoke(user_input=..., ctx=...)`; identifiers may be masked
  (`[MASKED]`) so record evidence is asserted by presence, not raw repr; audit
  spies assert on `call.args[1]` (the event payload), never the whole-call repr.
- Real-SDK pipeline behaviors encoded by the suite: `__call__` short-circuits on an
  incoming errored state (execute() is skipped; error status/error_log pass through);
  the framework mask rewrites Title-Case bigrams (across newlines), emails, and
  digit groups in `user_input`/`validated_input` to `[MASKED]` before `execute()`
  sees the text - positive payloads are PII-free, intentional-PII tests assert the
  `[MASKED]` path.
- Domain audit events are muted per module via an autouse fixture patching
  `src.nodes.<mod>.emit_trace_event` (never a `sys.modules` stub of `shared.*`).

## Unit Tests (`tests/unit/`)

| TC-ID | Test file | Focus | Expected |
|-------|-----------|-------|----------|
| U-01 | test_trust_gate.py | trust boundary: ANONYMOUS caller on the VERIFIED_EXTERNAL pre_process gate; inner nodes ANONYMOUS; trust-posture declarations | denial RETURNS error dict (`status == AgentStatus.ERROR.value`, "trust gate denied" in error_log, execute-only keys absent); VERIFIED_EXTERNAL passes; every inner node declares ANONYMOUS |
| U-02 | test_pre_process_node.py | caller-contract validation (type/shape bounds on the record-id aliases, refusals naming the field and never the value; non-finite floats refused as wrong-typed; undeclared fields dropped) + the template-owned instruction-override screen asserted through DIRECT `execute()` calls, both directions (chat-template control tokens, anchored directive phrases, markup splices, hostile field names and nested/`\u`-escaped payloads refused with nothing carried forward; ordinary CRM prose borrowing the same words accepted) | refusals behavioural (error status, no validated_input/record_hint/caller_fields); accepts pinned |
| U-02b | test_pre_process_node.py | serialize NL request + record hint into `validated_input` (JSON) and `caller_fields`; markup strip; hint priority record_id > record_hint > contact_id > deal_id | hint resolved by priority; `<script>` stripped; empty/missing -> `status=error`; the audit payload carries hint presence only |
| U-03 | test_validate_input_node.py | empty/short guard; JSON-shaped input; framework `[MASKED]` path for emails; node-level token flag-and-redact (`secret_*`) | email -> `[MASKED]` before execute; token -> `[REDACTED]` + `redaction_flags=["token"]` (JSON string); empty/short -> error; the audit payload carries flags only |
| U-04 | test_classify_intent_node.py | intent = {lookup,create,update} x {contact,deal} (keyword, writes-first priority, contact default, read-only default) | correct intent per keyword incl. deal-axis flip; no-signal defaults to lookup_contact with non-fatal note; empty -> error; the audit event carries the intent label only |
| U-04b | test_classify_intent_node.py (`TestClassifyIntentNodeLLMEnhancement`) | optional per-invocation LLM enhancement (test-double only, no real network call): well-formed JSON overrides the keyword result; prose/markdown-fence-wrapped JSON still parses; malformed / wrong-enum-shape response and a raising client fall back to the keyword result; no `llm=` injected + no secret bound falls back (the real production shape without a configured key); empty input never calls the LLM | override, fallback, and no-call-on-empty-input all behave as documented; audit event's `source` field is `"llm"` / `"heuristic"` correctly |
| U-05 | test_infer_sansan_fields_node.py | record-id resolution (explicit text id > id-shaped hint; never invented); quoted name; `Key: value` custom fields; Sansan `record_data` envelope per intent (JSON string) | lookup `{record_id}` (contact + deal); create/update `record_data[0]` with id/name/custom_fields (name-key aliases excluded); unresolved id left `""`; empty input -> error |
| U-05b | test_infer_sansan_fields_node.py (`TestInferSansanFieldsNodeLLMEnhancement`) | optional per-invocation LLM enhancement of `record_label`/`custom_fields` only (test-double only, no real network call); `target_id` NEVER routed through the LLM even when the double returns a well-formed response | well-formed JSON overrides label/fields; prose-wrapped JSON parses; malformed / wrong-shape response and a raising client fall back to regex; no `llm=` + no secret bound falls back; empty input never calls the LLM; `target_id` always comes from regex+hint regardless of the LLM response; audit event's `source` field is `"llm"` / `"heuristic"` correctly |
| U-06 | test_call_sansan_api_node.py | lookup/create/update via the network-free v1 stub (contacts + deals collections); `sansan_config` state field + `execute(state, config=...)` override (documented direct-execute exception); API error / unresolved id / unknown intent / missing payload; secret posture (live transport refuses to run unauthenticated; key read via `ctx.secrets`, never env/state) | record_id/record_ref on success (`sansan://contacts/...` / `sansan://deals/...`); 403 surfaces in error_log; live+no-secret -> error "unauthenticated"; live+bound secret -> key passed to client; the audit event carries presence signals with `stub_transport=True` |
| U-07 | test_confirm_node.py | human-readable confirmation per intent verb (contact + deal); ref/id formatting; label fallback | "Retrieved/Created/Updated Sansan contact/deal ... ref=... id=..."; missing evidence -> error |
| U-08 | test_post_process_node.py | `formatted_output` shaping (payload round-trip); errored state passes through `__call__` un-masked (short-circuit); record-evidence output gate (module-level `_security_gate_output()` helper, full-node path + direct function tests); the gate walks NESTED payload structures; every error return CLEARS every output-bearing field (result, confirmation, sansan_payload, record_label) AND the record evidence (record_id, record_ref, target_id), and the violation log names the location, never the value; the ERROR envelope is closed-set on every non-success path (inner-workflow error with and without evidence/answer still in state, credential in error_log, missing evidence, credential in confirmation, credential nested in payload — parametrised): `formatted_output` is exactly `{"reason": <module constant>}` and truthy, an `error_log` sentinel (name + token-shaped upstream text) reaches nowhere in the returned mapping, gate violations land in `error_log` never in the envelope, the errored path re-emits nothing, audit events carry the reason code and a count only | success shape with parsed `sansan_payload`; error status/error_log preserved, no success shape fabricated; SUCCESS without record_id/record_ref blocked; credential-shaped values (top-level, nested, error-shape) blocked AND cleared, the response replaced by the closed-set envelope |
| U-09 | test_sansan_client.py | Sansan Open API client: find/create/update record on `/contacts` + `/deals`; `X-Sansan-Api-Key` header; `SansanApiError` on non-2xx; stub shapes (`data` list / task receipt + record_id echo, `_stub` marker); `uses_stub_transport` | correct URLs/headers/bodies; 400 raises with joined `errors`; stub deterministic shapes |
| U-10 | test_config.py | flat `config/agent.yaml` manifest + `config/config.yaml` runtime parameters, AND the propagation of those values to their consumers | id CMN-C2-274, Cat 2, CMN, ToolCallingAgent, namespace `cmn`, dotted `class:` entry point, VERIFIED_EXTERNAL, `generation_mode: llm`, `requires.secrets` = the three `AZURE_OPENAI_*` keys / `requires.extras = ["openai"]`, no nested `agent:` block, no retired `timeout_seconds` key; `max_retry`/`timeout_s`/`sansan` reach `_parent_config()`, the inner graph's `sansan_config` state field, and the outer graph's own `config` |
| U-11 | test_domain_workflow_graph.py | inner `SansanWorkflowGraph`: identity, `_extra_initial_state()` sansan_config JSON injection + caller-context seeding from the bridge, `route()` error short-circuit, `get_output` contract, compile, direct inner invoke on the stub | name/state_schema correct; config forwarded as JSON string; bridge contract seeded as `input_context`; error -> END; inner invoke runs validate -> classify -> infer -> call -> confirm to SUCCESS with record evidence |
| U-12 | test_framework_compliance_tc06_tc07.py | framework-contract compliance: domain nodes may extend the input/output gates only through the `_extra_*` hooks | TC-06: overriding the default input gate raises at class definition; TC-07: overriding the default output gate raises at class definition |

## Proof-of-Boundary Tests (`tests/proof_of_boundary/`)

| PB-ID | Boundary | Test | Expected |
|-------|----------|------|----------|
| PB-4 | Import isolation | test_import_isolation.py | AST scan of `src/`: 0 platform-SDK imports |
| PB-2/PB-5 | State serialization | test_state_safety.py | `state.py`: no Pydantic/credential fields |
| PB-6 | Backbone invoke-order + external-trust | test_pb_invoke_order.py | `_VALID_PAYLOAD` byte-equal to `deploy/invoke_payload.json` "input" (asserted); VERIFIED_EXTERNAL caller yields `status=success` with `node_history == [InitializeNode, PreProcessNode, SansanWorkflowGraphNode, PostProcessNode, FinalizeNode]` and record evidence + confirmation in `result["output"]`; ANONYMOUS caller denied at pre_process (error, no post_process, no output); blank input -> error, not crash |
| PB-7 | HITL interrupt propagation *(conditional)* | test_pb7_hitl_interrupt_propagation.py | **Auto-waived - non-HITL** (`config/config.yaml` has no `hitl.enabled: true`): module-level skipif; stub bodies are real AssertionErrors so enabling HITL without implementing PB-7 fails loudly |
| PB (boot) | Server entry point | test_server_boot.py | importing `src.api.server` does not raise (construct + compile + provision_secrets at import); agent constructs + compiles via the supported path; `/invoke` + `/health` routes exposed |
| PB (containment) | ERROR envelope discloses no record evidence and carries closed-set labels only | test_error_envelope_no_record_evidence.py | the errored `post_process` branch returns a PRESENT and TRUTHY `formatted_output` (a falsy value re-opens the framework's `formatted_output or result` fallback) that is exactly `{"reason": "workflow_failed"}` — none of `record_id` / `record_ref` / `record_label` / `confirmation` / payload content / `error_log` text — and CLEARS all of those fields from state; through the agent's own `get_output()` over the merged state, the inner-workflow error (via `execute()`, the surface the pipeline short-circuits to) and both gate blocks (full node pipeline) hand the caller exactly the closed-set envelope, with an `error_log` sentinel (name + token-shaped upstream text) and the gate's own violation text reaching nowhere in it; the not-found, upstream-failure and transport-failure reasons written to `error_log` name a closed-set label (record type, HTTP status, exception class), never the record id, the URL or the personal data an upstream body can quote; success-path controls prove the clean path still returns the evidence and the structured surface |
| PB (E2E) | Caller contract through the real ASGI `/invoke` | test_invoke_e2e.py | Bearer auth enforced (401 without/with a wrong token); real record evidence derived from the request (record_id, `sansan://` reference) and output that tracks the input rather than a constant; a caller-only record id reaches the workflow AND the same instruction fails without it (the bridge, proven both ways); lookup/create paths and the deal collection reachable; every record-id alias fails CLOSED on a wrong type, a bad shape, an over-cap size and on bare `NaN`/`Infinity`/`-Infinity` sent over the wire; rejections never echo the value; adapter caps return 413; instruction-override content refused end-to-end on both channels while ordinary wording passes; no credential-shaped value, traceback or source path in the response; the structured surface withheld on error; record identifiers returned byte-identical |

> PB-1 (audit emission) is covered inside the unit suite via the emit-spy
> tests (validate / classify / call nodes assert on the event payload,
> `call.args[1]`). PB-3 (live external service) is exercised at first-invoke
> against a live tenant, not in this suite - the v1 transport is the documented
> network-free stub.

## Test Execution Summary
- Execution date: 2026-08-28
- Runner: `python -m pytest tests/ -v` against the real `agenticstar-agentcore[anthropic]==1.0.1` wheel
- Total tests: 204
- Pass: 202 / Fail: 0 / Skip: 2 (PB-7 A/B - auto-waived, non-HITL)
