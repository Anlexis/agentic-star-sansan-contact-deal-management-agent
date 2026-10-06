"""AgentCore Platform v1.0 - Sansan REST API client.

Service layer: a thin wrapper around the Sansan (business-card contact & sales
deal SaaS) REST API contact/deal collections. Contains NO business logic, NO
routing, and NO credentials - the API key is passed in per call by the node
(which reads it via ctx.secrets). This module imports no framework/SDK
internals - pure stdlib (import-isolation, PB-4).

v1 LIMITATION (deliberate, documented):
    The DEFAULT transport is a deterministic, NETWORK-FREE stub. It returns the
    documented connector response shapes (a ``data`` list for lookups; a
    task/receipt shape with a synthetic ``record_id`` echo for create/update,
    derived from the request) so the pipeline is runnable and testable without
    a live Sansan tenant or the ``requests`` package - it does NOT perform a
    live Sansan call. This is a sibling template's kaonavi_client service shape
    adapted to Sansan. The rule it follows: never fake a live call;
    document the limitation.

    To perform real Sansan calls, inject live transports (requests-based
    ``post`` / ``patch`` / ``get``) at construction time; the method contracts
    address the Sansan Open API collections (``/contacts`` for business-card
    contacts, ``/deals`` for sales deals; ``X-Sansan-Api-Key`` auth header)
    with a normalized ``{"record_data": [...]}`` envelope, so the live
    transport adapter maps the envelope onto the exact wire shape - no
    business-logic change is needed to go live. A live transport also requires
    a real API key (see CallSansanApiNode - the stub runs without one because
    no request ever leaves the process).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

# A transport callable: (url, headers, json_body) -> (status_code, response_dict)
Transport = Callable[[str, "dict[str, Any]", "dict[str, Any]"], "tuple[int, dict[str, Any]]"]

_BASE_URL = "https://api.sansan.com/v3"

# Record-type -> REST collection path segment.
_COLLECTIONS = {"contact": "contacts", "deal": "deals"}


class SansanApiError(Exception):
    """Raised when the Sansan REST API returns a non-2xx status."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"Sansan API error {status_code}: {message}")


class SansanClient:
    """Sansan REST API contact/deal client.

    Args:
        base_url: Sansan API base URL (default https://api.sansan.com/v3).
        post/patch/get: optional injected transports (tests or a live client).
            When none is injected, a deterministic NETWORK-FREE v1 stub is used
            (see the module docstring - it returns the documented shape without
            a live Sansan call).
    """

    def __init__(
        self,
        base_url: str = _BASE_URL,
        *,
        post: Transport | None = None,
        patch: Transport | None = None,
        get: Transport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._post = post
        self._patch = patch
        self._get = get

    # -- transport mode --------------------------------------------------------

    @property
    def uses_stub_transport(self) -> bool:
        """True when NO live transport is injected (the network-free v1 default)."""
        return self._post is None and self._patch is None and self._get is None

    # -- auth ----------------------------------------------------------------

    def _headers(self, api_key: str) -> "dict[str, Any]":
        """Build the Sansan REST API auth headers.

        api_key is supplied per-call by the node (from ctx.secrets); it is
        never persisted on the instance or logged.
        """
        return {
            "Content-Type": "application/json",
            "X-Sansan-Api-Key": api_key,
        }

    # -- v1 deterministic stub transport (default; NO network) ----------------

    def _stub_transport(
        self, url: str, headers: "dict[str, Any]", json_body: "dict[str, Any]"
    ) -> "tuple[int, dict[str, Any]]":
        """Deterministic, network-free v1 stub - returns the documented connector shape.

        NOT a live call. Synthetic ids are derived from the request so the
        response is stable and inspectable. See the module docstring for the v1
        limitation and how to inject live transports.
        """
        seed = url + "|" + json.dumps(json_body, sort_keys=True, ensure_ascii=False, default=str)
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        label = "Deal" if url.rstrip("/").endswith("/deals") else "Contact"
        if json_body.get("_sansan_op") == "lookup":
            rid = str(json_body.get("record_id", "")) or f"r-{digest[:8]}"
            # Documented lookup shape: {"data": [...]} record list.
            return 200, {
                "data": [
                    {
                        "id": rid,
                        "name": f"{label} {rid}",
                        "tags": [],
                        "custom_fields": [],
                    }
                ],
                "_stub": True,  # marks the network-free v1 stub response
            }
        # POST (create) / PATCH (update) - documented task-receipt shape, plus a
        # synthetic record_id echo so the caller can reference the affected
        # record without a follow-up lookup.
        record_data = json_body.get("record_data") or [{}]
        first = record_data[0] if isinstance(record_data, list) and record_data else {}
        rid = str(first.get("id", "")) or f"r-{digest[:8]}"
        return 200, {
            "task_id": int(digest[:6], 16),
            "record_id": rid,
            "_stub": True,  # marks the network-free v1 stub response
        }

    def _resolve(self, injected: Transport | None) -> Transport:
        return injected or self._stub_transport

    def _collection_url(self, record_type: str) -> str:
        collection = _COLLECTIONS.get(record_type, "contacts")
        return f"{self._base_url}/{collection}"

    # -- public API ---------------------------------------------------------

    def find_record(self, record_type: str, record_id: str, api_key: str) -> "dict[str, Any]":
        """GET /contacts | /deals - look up a contact/deal record by id.

        The live Sansan endpoint returns a paged collection; a live ``get``
        transport adapter is expected to filter to ``record_id`` (the stub
        returns the matching record directly). Returns the parsed response dict
        (containing ``data``). Raises SansanApiError on non-2xx.
        """
        url = self._collection_url(record_type)
        transport = self._resolve(self._get)
        status, body = transport(url, self._headers(api_key), {"_sansan_op": "lookup", "record_id": record_id})
        if not (200 <= status < 300):
            raise SansanApiError(status, _err_message(body))
        return body

    def create_record(self, record_type: str, payload: "dict[str, Any]", api_key: str) -> "dict[str, Any]":
        """POST /contacts | /deals - register a new contact/deal record.

        ``payload`` is the normalized ``{"record_data": [...]}`` request
        envelope. Returns the parsed response dict (task receipt). Raises
        SansanApiError on a non-2xx status.
        """
        url = self._collection_url(record_type)
        transport = self._resolve(self._post)
        status, body = transport(url, self._headers(api_key), payload)
        if not (200 <= status < 300):
            raise SansanApiError(status, _err_message(body))
        return body

    def update_record(self, record_type: str, payload: "dict[str, Any]", api_key: str) -> "dict[str, Any]":
        """PATCH /contacts | /deals - partially update an existing contact/deal record.

        ``payload`` is the normalized ``{"record_data": [...]}`` request
        envelope (each entry keyed by ``id``). Returns the parsed response dict
        (task receipt). Raises SansanApiError on a non-2xx status.
        """
        url = self._collection_url(record_type)
        transport = self._resolve(self._patch)
        status, body = transport(url, self._headers(api_key), payload)
        if not (200 <= status < 300):
            raise SansanApiError(status, _err_message(body))
        return body


def _err_message(body: Any) -> str:
    """Extract a human-readable error message from a Sansan error body."""
    if isinstance(body, dict):
        errors = body.get("errors")
        if isinstance(errors, list) and errors:
            return "; ".join(str(e) for e in errors)
        msg = body.get("message")
        if msg:
            return str(msg)
    return str(body)
