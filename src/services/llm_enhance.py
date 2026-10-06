"""Per-invocation Azure OpenAI client resolution for the optional LLM
enhancement path in classify_intent / infer_sansan_fields.

Built fresh per invocation, never cached on a node instance: node objects
are constructed once by register_nodes() and reused across every invocation
via the registry's node cache, so caching a client built from one caller's
secrets would leave it readable by the next caller. Any failure to resolve a
working client - missing secret, bad endpoint shape, whatever - returns None
so the caller falls back to its own deterministic/heuristic value; this
never raises.
"""

from typing import Any

from framework.schemas.invocation_context import InvocationContext
from shared.services.llm.azure_openai_client import AzureOpenAIClient


def resolve_llm(constructor_llm: Any, state: "dict[str, Any]") -> Any:
    """Return an LLM client for the optional enhancement path, or None.

    `constructor_llm` is the node's own `self._llm` - a test-double injection
    seam only. Production wiring (register_nodes()) never passes one, so in
    production this always builds fresh from ctx.secrets. A unit test injects
    a fake object exposing `complete(messages) -> {"content": ...}` instead.
    """
    if constructor_llm is not None:
        return constructor_llm
    try:
        ctx = InvocationContext.from_state(state)
        return AzureOpenAIClient(
            {
                "api_key": ctx.secrets.require("AZURE_OPENAI_API_KEY"),
                "azure_endpoint": ctx.secrets.require("AZURE_OPENAI_ENDPOINT"),
                "azure_deployment": ctx.secrets.require("AZURE_OPENAI_DEPLOYMENT"),
            }
        )
    except Exception:
        return None
