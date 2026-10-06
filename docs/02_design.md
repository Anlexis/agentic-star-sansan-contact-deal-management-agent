# Template Design Specification — CMN-C2-274 Sansan Contact & Deal Agent

## Position in AgentCore Architecture

- **Agent Class**: `SansanContactDealAgent` (`src/graph/graph.py`)
| L1 Base (framework base class) | `AgentBaseGraph` — direct framework inheritance |
- **Category**: Cat 2 (multi-step domain workflow, ToolCallingAgent). Outer
  `AgentBaseGraph` 5-node backbone; the domain pipeline is encapsulated in a
  `GraphNode` (`main` slot) wrapping an inner `BaseGraph`
  (`src/graph/domain_workflow_graph.py`).
- **Agent Type**: ToolCallingAgent — classify intent -> extract contact/deal
  fields -> build a Sansan REST API request -> call the tool -> format the
  confirmation. No RAG retrieval, no autonomous ReAct loop.
- **Three-Layer Separation**:
  - State: flat TypedDict `State(AgentState)` (no Pydantic — msgpack incompatible)
  - Node: framework base-class inheritance (Template Method: `execute(self, state) -> dict` override only)
  - Graph: composition (`register_nodes()` + `super().register_nodes()`; `add_edges()`
    not overridden on the outer graph)

## Architecture Overview

### Outer graph — node configuration (`src/graph/graph.py`)

| Node | Responsibility | Input State | Output State | Trust | Inherits/Overrides |
|------|---------------|-------------|--------------|-------------|-------------------|
| initialize | framework setup (schema, session, trust) | user_input | session/trust fields | framework default | InitializeNode (default) |
| pre_process | validate the caller contract (`input_context`); refuse instruction-override content on BOTH caller channels; markup/length sanitize; serialize NL request into `validated_input` (JSON) | user_input, input_context | validated_input, record_hint, caller_fields | **VERIFIED_EXTERNAL** (the single external gate) | PreProcessNode (FunctionNode) |
| main | run the inner Sansan workflow subgraph | validated_input, caller_fields | result, intent, target_id, record_id, record_ref, record_label, confirmation, sansan_payload | GraphNode (caller ctx forwarded unchanged) | SansanWorkflowGraphNode (GraphNode) |
| post_process | shape caller-facing `formatted_output`; module-level `_security_gate_output()` scan; on violation, clear every output-bearing field | inner-result fields | formatted_output | ANONYMOUS | PostProcessNode (FunctionNode) |
| finalize | framework finalize (metadata, timing) | — | response_metadata | framework default | FinalizeNode (default) |

### Inner workflow — node configuration (`src/graph/domain_workflow_graph.py`)

Inner graph inherits `BaseGraph` (fully custom linear topology). The 5 pipeline
steps map 1:1 to inner nodes. **Every inner domain node declares
`required_trust_level = TrustLevel.ANONYMOUS`** — the caller's
`InvocationContext` is forwarded into the subgraph unchanged, so the single
external trust gate stays on the backbone `pre_process`.

| Inner node | Step | Responsibility | Output | Trust |
|------|------|---------------|--------|-------------|
| validate_input | 1 ValidateInput | empty/non-request guard; deterministic (regex) flag-and-redact of email/token-like strings before logging | validated_input, record_hint, redaction_flags | ANONYMOUS |
| classify_intent | 2 ClassifyIntent | keyword classification on two axes (operation x record type) -> lookup/create/update x contact/deal, optionally overridden by an LLM call (graceful degrade on failure); low-confidence operation -> lookup (read-only default — never a write) | intent | ANONYMOUS |
| infer_sansan_fields | 3 InferSansanFields | extract record id (regex+hint only, never LLM) / display name (or deal title) / custom fields — regex baseline, optionally overridden for label/fields by an LLM call (graceful degrade on failure); assemble the Sansan REST API request body per intent; an unresolved record id is left empty (never invented) | record_label, target_id, sansan_payload | ANONYMOUS |
| call_sansan_api | 4 CallSansanApi | GET (lookup) / POST (create) / PATCH (update) against `/contacts` or `/deals` via `SansanClient`; API key via ctx.secrets; 4xx/5xx -> status=error | record_id, record_ref, target_id, record_label | ANONYMOUS |
| confirm | 5 Confirm | format intent + record id + reference into a human-readable confirmation | confirmation, result | ANONYMOUS |

### Data Flow

```
Outer:  START -> initialize -> pre_process -> main -> {route} -> post_process -> finalize -> END
                                              | (RETRY, max 3) ^
Inner (inside main / SansanWorkflowGraphNode):
        START -> validate_input -> classify_intent -> infer_sansan_fields
              -> call_sansan_api -> confirm -> END
```

The instruction text travels as a JSON string: `pre_process` serializes
`{"text", "record_hint"}` into `validated_input`,
`SansanWorkflowGraphNode.extract_input()` hands that JSON to the subgraph, and
the first inner node (`validate_input`) parses it back. The VALIDATED caller
contract additionally crosses the graph boundary on a ContextVar bridge (see
"Crossing the graph boundary"), which is the authoritative copy: the framework
masks `validated_input` between hops, so values inside that JSON can be
rewritten in transit. Inner nodes read
`state.get("validated_input") or state.get("user_input", "")`.

### State Definition (`src/schemas/state.py`)

All domain fields are declared `NotRequired[...]` (state contract —
fields are absent until their producer node writes them). Dict/list payloads
are stored as JSON strings (`Optional[str]`) via the module helpers
`to_json` / `from_json`, used by every producer and consumer.

| Field | Type | Purpose | Producer |
|-------|------|---------|----------|
| record_hint | NotRequired[str] | caller-supplied contact/deal id hint; never inferred | pre_process / validate_input |
| target_id | NotRequired[str] | resolved Sansan record id (v1: pass-through when the hint/text already carries an id) | infer_sansan_fields |
| caller_fields | NotRequired[Optional[str]] | JSON — the caller contract after PreProcessNode validated it (record_hint); decoded by the graph node at the inner-graph boundary | pre_process |
| redaction_flags | NotRequired[Optional[str]] | JSON list of pattern categories redacted before logging | validate_input |
| record_label | NotRequired[str] | contact display name / deal title (record label) | infer_sansan_fields / call_sansan_api |
| sansan_payload | NotRequired[Optional[str]] | JSON — assembled Sansan REST API request body (msgpack-safe: stored as a JSON string via `to_json`/`from_json`) | infer_sansan_fields |
| sansan_config | NotRequired[Optional[str]] | JSON — the `sansan:` section of `config/config.yaml` forwarded by `_parent_config()` and injected via the inner graph's `_extra_initial_state()` | inner graph |
| record_id | NotRequired[str] | contact/deal record id returned by Sansan | call_sansan_api |
| record_ref | NotRequired[str] | human-readable record reference (`sansan://contacts/<id>` or `sansan://deals/<id>`) | call_sansan_api |
| confirmation | NotRequired[str] | human-readable confirmation | confirm |

`intent`, `result`, `validated_input`, `formatted_output` are inherited from
`AgentState` and are **not** re-declared.

**State Constraints (mandatory, satisfied):**
- Flat TypedDict only (primitives + JSON-serializable) — no Pydantic/dataclass.
- No JWT / API keys / credentials in State — the Sansan API key is accessed via `ctx.secrets`.
- InvocationContext read via `InvocationContext.from_state(state)`, never stored in State.

## Configuration Forwarding (nested Cat-2)

Nodes take **no constructor arguments** (SDK v1 nodes are no-arg; configuration
never rides on node instances).

Runtime parameters live in `config/config.yaml`; `config/agent.yaml` is the
static registry manifest and holds no runtime block.
`SansanWorkflowGraphNode._parent_config()` reads `config/config.yaml` and
forwards the `sansan:` section plus the runtime values (`max_retry`,
`timeout_s`) to the inner graph under `config["configurable"]` — never `{}`.
The inner graph's `_extra_initial_state()` then injects the `sansan` section
into State as a JSON string (`sansan_config`), where
`CallSansanApiNode.execute(state, config=None)` reads it (an explicit
`config["configurable"]["sansan"]` override is also honoured for direct/unit
invocation).

A reader aimed at the wrong file returns `{}` and every declared value goes
quietly dead while the tests stay green, so `tests/unit/test_config.py`
asserts the values ARRIVE — at `_parent_config()`, at the inner graph's state,
and on the outer graph's own `config` — rather than only that they are
declared.

`src/api/server.py` loads the same file via `load_runtime_config()` and passes
it as `config=`. Without that, the framework's own `max_retry` routing would
run on its built-in default no matter what the file said.

## Caller-Data Contract

Structured caller data arrives on `input_context` beside the free-text
instruction and is validated in `PreProcessNode`, the node that owns the
contract:

| Field | Type | Bound |
|---|---|---|
| `record_id` / `record_hint` / `contact_id` / `deal_id` | string | an inert record identifier `^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$` (first supplied alias wins) |

Rules: the value must be the declared TYPE and is never coerced —
`str(float("nan"))` is `"nan"`, which satisfies an identifier check, so a
coercing validator would aim a lookup or write at whatever fell out. A refusal
names the field and never echoes the value; a hostile field NAME is masked,
never echoed either. Undeclared fields are dropped, never forwarded. The
adapter additionally caps the envelope (16 keys, 256KB). An absent hint simply
degrades to whatever id the instruction text yields.

The template accepts **no caller-supplied numbers**, so there is no numeric
threshold to parse; the type guard is what keeps non-finite values out, and the
test suite drives bare `NaN` / `Infinity` / `-Infinity` through the real route
for every alias to prove it.

Instruction-override content (directives aimed at the model: role
reassignment, system-prompt manipulation, chat-template control tokens such as
`<|im_start|>`, `[INST]`, `<<SYS>>`) is REFUSED, fail closed, on BOTH caller
channels — the instruction text and every decoded string in `input_context`,
keys included, at any depth. The screen runs on the text both as received and
after the markup strip, so a control token is caught before the strip removes
it and a spliced directive (`ig<b>nore...`) is caught after the strip
re-assembles it. The screen is the template's own
(`src/services/security.py`), never delegated to the framework gate, and the
tests prove it by calling `execute()` directly with no framework wrapper in
front — in both directions: ordinary CRM prose that borrows the same words
("please ignore my previous request", "override approved by the account
owner") stays accepted.

### Crossing the graph boundary

`GraphNode.execute()` invokes the inner graph without forwarding
`input_context`, so an inner read of that field would always see `{}`. The
validated contract crosses on a ContextVar (`src/graph/context_bridge.py`):
`SansanWorkflowGraphNode.extract_input()` stashes it, and the inner graph's
`_extra_initial_state()` seeds it. The ContextVar is per thread/task, so
concurrent invocations cannot see each other's data.

The authoritative copy deliberately does NOT ride inside the
`validated_input` JSON: the framework masks that field at every node boundary,
and a hyphenated numeric record id is exactly the shape the mask rewrites, so
the caller's target could be corrupted between hops. The bridge channel is not
masked. `tests/proof_of_boundary/test_invoke_e2e.py` proves the crossing
end-to-end: a record id supplied ONLY on the caller channel resolves the
lookup, and the identical instruction without it fails.

## Security Design

- **Trust gate** — the single external trust gate is on the outer backbone
  `PreProcessNode.required_trust_level = TrustLevel.VERIFIED_EXTERNAL`; every inner
  domain node — **including the write-capable `CallSansanApiNode`** — declares
  `TrustLevel.ANONYMOUS`. `GraphNode.execute()` forwards the caller's
  `InvocationContext` into the subgraph **unchanged** (no elevation), and
  `VERIFIED_EXTERNAL (1) < INTERNAL (2)`, so declaring an inner node `INTERNAL`
  would deny a legitimate external caller before the call runs — the boundary
  is therefore enforced exactly once, at `pre_process`. Agent-level default trust
  `VERIFIED_EXTERNAL` is declared in `config/agent.yaml`. `src/api/server.py`
  enforces the standalone entry-point Bearer-token auth boundary
  (`INVOKE_AUTH_TOKEN` -> VERIFIED_EXTERNAL elevation).
- **Input sanitize + instruction-override screen** — the backbone `pre_process`
  refuses instruction-override content on both caller channels (see
  "Caller-Data Contract"), strips markup/control sequences and caps length
  before serialization (`src/services/security.py`).
  `ValidateInputNode.execute()` then runs a deterministic (regex, NOT LLM)
  scan for email addresses and access-token-like strings (`eyJ...`,
  `secret_...`, `sk-...`) and redacts them before any logging. A business-card
  contact / deal request legitimately names a person or company, so this scan
  is flag-and-redact for safe logging, not a hard reject; the framework PII
  mask in `BaseNode.__call__` additionally masks emails/phones/names in
  `user_input`/`validated_input`. The deterministic auto-rejects are the
  empty/non-request guard, the caller-contract bounds, and the
  instruction-override screen.
- **Secrets + output gate** — the integration key is read via
  `ctx.secrets.get("SANSAN_API_KEY")` (`InvocationContext.from_state(state)`),
  never `os.environ`, never stored in State. The manifest declares
  `requires.secrets: []` on purpose: the lookup is optional (`.get`, not a
  hard requirement), the shipped network-free stub transport runs without a
  credential, and declaring a secret the deployment does not provision would
  fail the agent at compile time. A missing key is tolerated **only** while
  the stub transport is active (no live call is made); with a live transport
  injected, a missing key is a hard `status=error` — a real API is never
  called unauthenticated. The domain output gate is the **module-level**
  `_security_gate_output()` in `src/nodes/post_process_node.py`, called from
  `PostProcessNode.execute()` on the success shape: it blocks
  any SUCCESS response that lacks record evidence (record_id/record_ref) and
  any credential-shaped string ANYWHERE in the caller-facing output — it
  walks nested mappings and lists, not just top-level strings. On a violation
  the node clears every output-bearing state field before returning the
  error: the response envelope falls back to `state["result"]` even on an
  error status, so blocked content left in state would still ship inside the
  error envelope. The caller-visible ERROR envelope is a **closed set**: on
  every non-success path (inner-workflow error, gate violation) the shared
  `_contain()` helper returns `formatted_output` as exactly
  `{"reason": <code>}` with `reason` one of the module constants
  `workflow_failed` / `output_withheld` — truthy, so the `or result` fallback
  never re-opens — and nothing is read from `error_log`, the violations list
  or an exception message, because those can carry identifiers, names and
  upstream response bodies that truncation or credential-only redaction does
  not bound. `error_log` stays operator-side (the audit channel); the base
  envelope never projects it. No node defines `_extra_security_gate_input/_output`
  instance methods (framework hooks are @final / auto-wrapped — domain checks
  live inline or in module-level helpers).
- **Audit** — every node's `execute()` emits exactly one positional
  `emit_trace_event("<node>_complete", {small non-PII payload}, state)` on its
  SUCCESS path (intent / presence signals only — never request text, contact or
  deal content, or credentials). `__call__()` is never overridden; `_invoke_impl`
  is never defined on any node. Event names (documented for operations):

  | Node | Audit event |
  |------|-----------|
  | pre_process | `pre_process_complete` |
  | validate_input | `validate_input_complete` |
  | classify_intent | `classify_intent_complete` |
  | infer_sansan_fields | `infer_sansan_fields_complete` |
  | call_sansan_api | `call_sansan_api_complete` |
  | confirm | `confirm_complete` |
  | post_process | `post_process_complete` |

## Structured product surface (`get_output()` override)

`SansanContactDealAgent.get_output()` **extends** `super().get_output()`:
on SUCCESS — and only when the gated `formatted_output` carries record
evidence (record_id/record_ref) — it additionally surfaces the structured keys
`intent`, `record_id`, `record_ref`, `record_label`, `confirmation` at the top
level of the invoke result, so callers can consume the outcome without parsing
the confirmation string. On any non-SUCCESS status (or missing evidence) the
structured keys are withheld — fail-closed, matching the output gate.

### Output invariant

This template renders **no monetary aggregates and no computed figures**, so the
fleet's round-to-the-nearest-1,000 precision grid does not apply here. Applying
one would be actively harmful: a Sansan record id such as `sc-1001` — or a
purely numeric one like `09012345678` — is an opaque identifier, and a
numeric-rewriting rule would corrupt the very record reference the response
exists to carry. Nothing in this template rewrites numbers on the way out, and
`tests/proof_of_boundary/test_invoke_e2e.py` pins that: record identifiers
must cross the output boundary byte-identical.

The invariant this template DOES enforce is its own: a SUCCESS response must
carry record evidence (`record_id` or `record_ref`), and no credential-shaped
string may reach the caller. The gate enforces that over **every
representation** of the output — it walks nested mappings and lists, not just
top-level strings, because `sansan_payload` is a nested mapping whose record
fields carry caller-derived text. A scan limited to top-level values steps
straight past a credential in a record field — and on a violation the gate
clears every output-bearing field, so nothing blocked can ride out inside the
error envelope either.

## v1 Implementation Note / LLM Enhancement Note

v1 shipped fully deterministic: intent classification (`ClassifyIntentNode`)
used a keyword heuristic and field inference (`InferSansanFieldsNode`) used
regex/line-structure extraction only, so the template ran and tested without
a live LLM. **That deterministic path is still the baseline and is still what
every unit test exercises without a secret bound** — nothing about it was
removed.

An optional per-invocation Azure OpenAI call now enhances both nodes
(`src/services/llm_enhance.py:resolve_llm()`, built fresh per invocation,
never cached on the node instance — cached secrets would leak across callers,
since node instances are reused via the registry's node cache):

- `ClassifyIntentNode` — the LLM classifies `{operation, record_type}`; a
  well-formed, enum-valid response overrides the keyword result.
- `InferSansanFieldsNode` — the LLM extracts `{record_label, custom_fields}`
  from freeform phrasing the "Key: value" regex cannot parse; a well-formed
  response overrides the regex result for those two fields only.

**`target_id` resolution never routes through the LLM, in either node** — it
stays regex-plus-caller-`record_hint`-only, unchanged from v1. That is the
safety-critical "never invent a target" path (see "Write target" in the
Design Decision Record below); `record_hint` is a caller-supplied side
channel the LLM prompt does not even see, and a freeform-extracted id is
exactly the kind of value that rule exists to keep out of a write.

Any failure on the LLM path — missing secret, network/API error, malformed or
wrong-shape JSON — is caught and degrades silently to the deterministic
result; neither node ever raises or sets `status=error` because of the LLM
call itself. `config/agent.yaml` declares `generation_mode: llm` and
`requires.secrets: [AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
AZURE_OPENAI_DEPLOYMENT]` / `requires.extras: [openai]` accordingly.

## v1 Limitation — Sansan client (documented)

`src/services/sansan_client.py` mirrors a sibling HR-profile template's
`kaonavi_client.py` service shape (injectable transport, `SansanApiError`, per-call API key, no
framework imports) but ships a **deterministic, network-free v1 stub** as its
default transport: it returns the documented connector response shapes (a
`data` list for lookups; a task/receipt shape with a synthetic `record_id`
echo for create/update, derived from the request) so the pipeline is runnable
and testable without a live Sansan tenant or the `requests` package. It does
**not** perform a live Sansan call. The rule it follows: never fake a live
call — document the limitation. To go live, inject real
`post`/`patch`/`get` transports at construction; the method contracts address
the Sansan Open API collections (`/contacts` for business-card contacts,
`/deals` for sales deals; `X-Sansan-Api-Key` auth header) with a normalized
`{"record_data": [...]}` envelope, so the live transport adapter maps the
envelope onto the exact wire shape — no business-logic change is required.
(The stub also runs without a live credential — see "Secrets + output gate"
above; a live transport requires `SANSAN_API_KEY`.)

## Framework Utilization

### Shared Components Used
- [x] InvocationContext — read in `CallSansanApiNode` via `InvocationContext.from_state(state)` (secrets + trust)
- [x] Trust gate — single external gate `PreProcessNode.required_trust_level = TrustLevel.VERIFIED_EXTERNAL`; inner domain nodes (incl. `CallSansanApiNode`) declare `TrustLevel.ANONYMOUS` (caller `InvocationContext` forwarded unchanged into the subgraph)
- [x] Secrets — `ctx.secrets.get("SANSAN_API_KEY")` (optional, stub default — see "Secrets + output gate"); `ctx.secrets.require("AZURE_OPENAI_API_KEY"/"AZURE_OPENAI_ENDPOINT"/"AZURE_OPENAI_DEPLOYMENT")` in `src/services/llm_enhance.py` (declared in `requires.secrets`, LLM enhancement — see "v1 Implementation Note / LLM Enhancement Note"); entry-point `bound_secrets` / `secrets_factory` / `provision_secrets` in `src/api/server.py`
- [x] Audit `emit_trace_event()` — one positional call per node on the SUCCESS path; framework lifecycle events (node_start/node_complete/node_error) NOT re-emitted

### Composition Pattern

- **Pattern**: GraphNode (subgraph) — Cat 2 outer/inner split.
- **Composition target**: inner `SansanWorkflowGraph` (`BaseGraph`) via `SansanWorkflowGraphNode.get_subgraph()`.
- **Config forwarding**: `SansanWorkflowGraphNode._parent_config()` reads
  `config/config.yaml` and forwards `{sansan, agent (max_retry, timeout_s)}`
  under `config["configurable"]` to the subgraph.
- **Error propagation strategy**: `propagate` (default) — inner errors re-raised as
  `SubgraphError`; per-step `status=error` + `error_log` for API/validation failures
  (no silent pass).

## Import Isolation Confirmation
- [x] Template imports `framework/` and `shared/` only; no platform-SDK import anywhere
- [x] `src/services/sansan_client.py` and `src/services/security.py` have no
      framework imports (pure service layer, stdlib only)

## Design Decision Record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| L1 base type | AgentBaseGraph | AutonomousBaseGraph | AgentBaseGraph | Fixed multi-step pipeline (Cat 2), not an autonomous loop |
| Composition pattern | flat Cat 1 (MainNode) | GraphNode + inner subgraph | GraphNode + inner subgraph | Cat 2 must not be flat; 5 domain steps live in the inner graph |
| LLM dependency | LLM client mandatory (no fallback) | deterministic baseline + optional per-invocation LLM enhancement, graceful degrade on any failure | deterministic baseline + optional LLM enhancement | every unit test still exercises the deterministic path without a secret bound; an LLM outage/error never surfaces as `status=error`; see "v1 Implementation Note / LLM Enhancement Note" |
| Sansan client | live `requests` call | injectable transport + documented v1 stub default | injectable + v1 stub default | no live network in v1; document the limitation; go-live is a transport injection, no logic change |
| Node configuration | ctor-arg dependency injection | no-arg nodes + runtime-config forwarding via `_parent_config()` -> `configurable` -> state | no-arg nodes | SDK v1 nodes are no-arg (ctor args TypeError at graph build); `config/config.yaml` stays the single runtime-config source |
| Record-type handling | separate contact/deal pipelines | one pipeline; intent encodes operation x record type | one pipeline, intent-encoded | contacts and deals share the identical 5-step shape; two pipelines would duplicate every node |
| Write target | infer record id from NL freely | caller-supplied/explicit id only; unresolved left empty | explicit only | never write to the wrong contact/deal record; unresolved id -> status=error, not invented |
| Default intent | create_contact | lookup_contact | lookup_contact | low-confidence classification must never default to a write |
| Caller-contract crossing | ride inside the `validated_input` JSON | ContextVar bridge (`src/graph/context_bridge.py`) | ContextVar bridge | the framework masks `validated_input` between hops; the bridge channel is unmasked and per-task |
