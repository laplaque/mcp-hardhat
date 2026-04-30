#!/usr/bin/env python3
"""Minimal stdio-to-HTTP MCP proxy. Bridges stdin/stdout JSON-RPC to an HTTP MCP server."""
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

import argparse
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


def run(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    url = args.url
    extra_headers = resolve_headers(args)
    ctx = build_ssl_context(args.ca_file)
    session: str | None = None

    def http_post(payload: dict[str, object]) -> object | None:
        nonlocal session
        if payload.get("method") == "initialize":
            session = None
        data = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        headers.update(extra_headers)
        if session:
            headers["Mcp-Session-Id"] = session
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        resp = urllib.request.urlopen(req, context=ctx)  # noqa: S310
        if not session:
            session = resp.headers.get("Mcp-Session-Id")
        body = resp.read().decode()
        return parse_sse_or_json(body)

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        msg: dict[str, object] = json.loads(raw)
        if "id" not in msg:
            try:
                http_post(msg)
            except urllib.error.HTTPError as e:
                print(f"notification forwarding failed: HTTP {e.code}", file=sys.stderr)
            except Exception as e:
                print(f"notification forwarding failed: {type(e).__name__}", file=sys.stderr)
            continue
        try:
            resp = http_post(msg)
        except urllib.error.HTTPError as e:
            write_jsonrpc_error(msg["id"], -32000, f"HTTP MCP request failed with status {e.code}")
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
