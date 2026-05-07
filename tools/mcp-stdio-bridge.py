#!/usr/bin/env python3
"""Minimal stdio-to-HTTP MCP proxy. Bridges stdin/stdout JSON-RPC to an HTTP MCP server."""
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

import argparse
import contextlib
import json
import os
import ssl
import subprocess
import sys
import urllib.error
import urllib.request


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal stdio-to-HTTP MCP proxy")
    p.add_argument("url", help="HTTP(S) endpoint of the MCP server")
    p.add_argument("--header", action="append", default=[], help="HTTP header (e.g. 'Authorization: Bearer xxx')")
    p.add_argument(
        "--header-env",
        action="append",
        default=[],
        help="HTTP header from env var (e.g. 'Authorization=MY_TOKEN_VAR')",
    )
    p.add_argument(
        "--header-cmd",
        action="append",
        default=[],
        help=(
            "HTTP header from command output"
            " (e.g. 'Authorization=security find-generic-password -s obsidian-mcp -a bearer -w')"
        ),
    )
    p.add_argument("--ca-file", help="CA bundle path (overrides NODE_EXTRA_CA_CERTS)")
    p.add_argument("--bearer-env", metavar="ENV_VAR", help="Env var containing Bearer token for Authorization header")
    p.add_argument("--timeout", type=float, default=30.0, help="HTTP request timeout in seconds (default: 30)")
    p.add_argument("--debug", action="store_true", help="Log session changes and retries to stderr")
    p.add_argument(
        "--recover-stale-session",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Silently re-init when server returns -32001 recoverable-session error (default: enabled)",
    )
    return p.parse_args(argv)


def resolve_headers(args: argparse.Namespace) -> dict[str, str]:
    headers: dict[str, str] = {}
    for h in args.header:
        key, _, val = h.partition(":")
        headers[key.strip()] = val.strip()
    for h in args.header_env:
        key, _, var = h.partition("=")
        val = os.environ.get(var.strip())
        if not val:
            sys.exit(f"error: env var '{var.strip()}' not set")
        headers[key.strip()] = val
    for h in args.header_cmd:
        key, _, cmd = h.partition("=")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)  # noqa: S602
        if result.returncode != 0:
            sys.exit(f"error: command failed for --header-cmd '{key.strip()}'")
        val = result.stdout.strip()
        if not val:
            sys.exit(f"error: command returned empty output for --header-cmd '{key.strip()}'")
        headers[key.strip()] = val
    if args.bearer_env:
        bval = os.environ.get(args.bearer_env)
        if not bval:
            sys.exit(f"error: env var '{args.bearer_env}' not set")
        headers["Authorization"] = f"Bearer {bval}"
    return headers


def build_ssl_context(ca_file: str | None = None) -> ssl.SSLContext:
    resolved = ca_file or os.environ.get("NODE_EXTRA_CA_CERTS") or os.environ.get("SSL_CERT_FILE")
    if resolved and os.path.exists(resolved):
        return ssl.create_default_context(cafile=resolved)
    return ssl.create_default_context()


def parse_sse_or_json(body: str) -> object | None:
    for line in body.split("\n"):
        if line.startswith("data:"):
            result: object = json.loads(line[5:].strip())
            return result
    if body.strip():
        parsed: object = json.loads(body)
        return parsed
    return None


def write_jsonrpc_error(request_id: object, code: int, message: str) -> None:
    sys.stdout.write(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": code, "message": message},
            }
        )
        + "\n"
    )
    sys.stdout.flush()


class BridgeConfig:
    def __init__(
        self,
        url: str,
        headers: dict[str, str],
        ctx: ssl.SSLContext,
        timeout: float,
        debug: bool = False,
        recover_stale_session: bool = True,
    ) -> None:
        self.url = url
        self.headers = headers
        self.ctx = ctx
        self.timeout = timeout
        self.debug = debug
        self.recover_stale_session = recover_stale_session


class BridgeState:
    def __init__(self) -> None:
        self.session_id: str | None = None
        self.generation: int = 0
        self.cached_init: dict[str, object] | None = None

    def set_session(self, new_id: str | None, *, debug: bool = False) -> None:
        if new_id != self.session_id:
            self.session_id = new_id
            self.generation += 1
            if debug:
                print(
                    f"mcp-stdio-bridge: session updated generation={self.generation}",
                    file=sys.stderr,
                )

    def clear_session(self, *, debug: bool = False) -> None:
        self.set_session(None, debug=debug)


class HttpMcpError(Exception):
    def __init__(self, status: int, body: str, method: str) -> None:
        self.status = status
        self.body = body[:4096]
        self.method = method
        super().__init__(f"HTTP {status} on {method}")


class _RecoveryFailedError(Exception):
    """Internal: silent re-init flow could not complete."""


_SESSION_EXPIRED_CODE = -32001

_STALE_SESSION_TERMS = ("session", "expired", "unknown", "invalid", "not found", "stale", "mcp-session-id")


def is_stale_session_failure(err: HttpMcpError, session_was_sent: bool) -> bool:
    if not session_was_sent:
        return False
    if err.method == "initialize":
        return False
    if err.status in (401, 403):
        return False
    if err.status not in (400, 404):
        return False
    body_lower = err.body.strip().lower()
    if not body_lower:
        return True
    return any(term in body_lower for term in _STALE_SESSION_TERMS)


def _is_recoverable_session_error(parsed: object) -> bool:
    """True if a parsed JSON-RPC body is a -32001 recoverable-session error."""
    if not isinstance(parsed, dict):
        return False
    err = parsed.get("error")
    if not isinstance(err, dict):
        return False
    if err.get("code") != _SESSION_EXPIRED_CODE:
        return False
    data = err.get("data")
    return isinstance(data, dict) and bool(data.get("recoverable"))


def http_post_once(
    cfg: BridgeConfig,
    payload: dict[str, object],
    session_id: str | None,
) -> tuple[object | None, str | None]:
    data = json.dumps(payload).encode()
    req_headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    req_headers.update(cfg.headers)
    if session_id:
        req_headers["Mcp-Session-Id"] = session_id
    req = urllib.request.Request(cfg.url, data=data, headers=req_headers, method="POST")
    try:
        resp = urllib.request.urlopen(req, context=cfg.ctx, timeout=cfg.timeout)  # noqa: S310
    except urllib.error.HTTPError as e:
        body_bytes = e.read(4096)
        body_text = body_bytes.decode("utf-8", errors="replace")
        method = str(payload.get("method", "unknown"))
        raise HttpMcpError(e.code, body_text, method) from e
    resp_session: str | None = resp.headers.get("Mcp-Session-Id")
    body = resp.read().decode()
    return parse_sse_or_json(body), resp_session


def _recover_and_replay(cfg: BridgeConfig, payload: dict[str, object], state: BridgeState) -> object | None:
    """Re-initialize the HTTP session silently and replay the original payload.

    Raises _RecoveryFailedError on any sub-step failure. Caller surfaces the
    original -32001 error to the client when this raises.
    """
    if state.cached_init is None:
        raise _RecoveryFailedError("no cached initialize payload")

    state.clear_session(debug=cfg.debug)
    internal_init = dict(state.cached_init)
    internal_init["id"] = f"_bridge_reinit_{state.generation}"

    try:
        init_result, init_session = http_post_once(cfg, internal_init, None)
    except (HttpMcpError, OSError, ValueError) as e:
        raise _RecoveryFailedError(f"reinit POST failed: {e}") from e

    if isinstance(init_result, dict) and "error" in init_result:
        raise _RecoveryFailedError("reinit returned JSON-RPC error")

    if init_session:
        state.set_session(init_session, debug=cfg.debug)

    # notifications/initialized \u2014 non-fatal if it errors; some servers don't require it.
    with contextlib.suppress(HttpMcpError, OSError, ValueError):
        http_post_once(
            cfg,
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            state.session_id,
        )

    if cfg.debug:
        print(
            f"mcp-stdio-bridge: silent reinit succeeded generation={state.generation}",
            file=sys.stderr,
        )

    try:
        result, resp_session = http_post_once(cfg, payload, state.session_id)
    except (HttpMcpError, OSError, ValueError) as e:
        raise _RecoveryFailedError(f"replay POST failed: {e}") from e

    if resp_session:
        state.set_session(resp_session, debug=cfg.debug)
    return result


def http_post(
    cfg: BridgeConfig,
    payload: dict[str, object],
    state: BridgeState,
) -> object | None:
    method = str(payload.get("method", ""))
    if method == "initialize":
        state.clear_session(debug=cfg.debug)
        state.cached_init = dict(payload)

    session_sent = state.session_id is not None
    try:
        result, resp_session = http_post_once(cfg, payload, state.session_id)
        if resp_session:
            state.set_session(resp_session, debug=cfg.debug)
    except HttpMcpError as err:
        if not is_stale_session_failure(err, session_sent):
            raise
        if cfg.debug:
            print(
                f"mcp-stdio-bridge: stale session suspected status={err.status}"
                f" method={err.method} generation={state.generation}",
                file=sys.stderr,
            )
        state.clear_session(debug=cfg.debug)
        if cfg.debug:
            print("mcp-stdio-bridge: cleared session and retrying once", file=sys.stderr)
        result, resp_session = http_post_once(cfg, payload, None)
        if resp_session:
            state.set_session(resp_session, debug=cfg.debug)

    # In-band recovery: HTTP 200 envelope with JSON-RPC -32001 + recoverable=true.
    # Server returns this when its session has expired but the transport still
    # works; transparently re-initialize and replay so the client never sees it.
    if cfg.recover_stale_session and method != "initialize" and _is_recoverable_session_error(result):
        if cfg.debug:
            print(
                f"mcp-stdio-bridge: in-band -32001 recoverable error method={method} generation={state.generation}",
                file=sys.stderr,
            )
        try:
            return _recover_and_replay(cfg, payload, state)
        except _RecoveryFailedError as e:
            if cfg.debug:
                print(
                    f"mcp-stdio-bridge: silent reinit failed ({e}); surfacing original -32001",
                    file=sys.stderr,
                )

    return result


def run(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = BridgeConfig(
        url=args.url,
        headers=resolve_headers(args),
        ctx=build_ssl_context(args.ca_file),
        timeout=args.timeout,
        debug=args.debug,
        recover_stale_session=args.recover_stale_session,
    )
    state = BridgeState()

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        msg: dict[str, object] = json.loads(raw)
        is_notification = "id" not in msg

        if is_notification:
            try:
                http_post(cfg, msg, state)
            except HttpMcpError as e:
                print(f"notification forwarding failed: HTTP {e.status}", file=sys.stderr)
            except Exception as e:
                print(f"notification forwarding failed: {type(e).__name__}", file=sys.stderr)
            continue

        try:
            resp = http_post(cfg, msg, state)
        except HttpMcpError as e:
            write_jsonrpc_error(msg["id"], -32000, f"HTTP MCP request failed with status {e.status}")
            continue
        except RuntimeError as e:
            write_jsonrpc_error(msg["id"], -32000, f"HTTP MCP request failed: {e}")
            continue
        except Exception as e:
            write_jsonrpc_error(msg["id"], -32000, f"HTTP MCP request failed: {type(e).__name__}")
            continue
        if resp:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run()
