"""Tests for mcp-stdio-bridge."""

from __future__ import annotations

import importlib.util
import json
import ssl
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the bridge as a module from its file path.
_BRIDGE_PATH = Path(__file__).resolve().parent.parent / "tools" / "mcp-stdio-bridge.py"
_spec = importlib.util.spec_from_file_location("mcp_stdio_bridge", _BRIDGE_PATH)
assert _spec is not None and _spec.loader is not None
bridge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bridge)


class TestParseSSEOrJSON:
    def test_plain_json(self) -> None:
        body = '{"jsonrpc": "2.0", "id": 1, "result": {}}'
        result = bridge.parse_sse_or_json(body)
        assert result == {"jsonrpc": "2.0", "id": 1, "result": {}}

    def test_sse_data_line(self) -> None:
        body = 'event: message\ndata: {"jsonrpc": "2.0", "id": 1, "result": {}}\n\n'
        result = bridge.parse_sse_or_json(body)
        assert result == {"jsonrpc": "2.0", "id": 1, "result": {}}

    def test_empty_body(self) -> None:
        assert bridge.parse_sse_or_json("") is None
        assert bridge.parse_sse_or_json("   ") is None

    def test_sse_with_multiple_data_lines(self) -> None:
        body = 'data: {"first": true}\ndata: {"second": true}\n'
        result = bridge.parse_sse_or_json(body)
        assert result == {"first": True}


class TestBuildSSLContext:
    def test_default_context_validates(self) -> None:
        ctx = bridge.build_ssl_context()
        assert ctx.verify_mode == ssl.CERT_REQUIRED
        assert ctx.check_hostname is True

    def test_nonexistent_ca_file_falls_back_to_default(self) -> None:
        ctx = bridge.build_ssl_context(ca_file="/nonexistent/path.pem")
        assert ctx.verify_mode == ssl.CERT_REQUIRED

    def test_never_cert_none(self) -> None:
        ctx = bridge.build_ssl_context()
        assert ctx.verify_mode != ssl.CERT_NONE


class TestWriteJSONRPCError:
    def test_error_format(self, capsys: pytest.CaptureFixture[str]) -> None:
        bridge.write_jsonrpc_error(42, -32000, "something broke")
        captured = capsys.readouterr()
        parsed = json.loads(captured.out.strip())
        assert parsed == {
            "jsonrpc": "2.0",
            "id": 42,
            "error": {"code": -32000, "message": "something broke"},
        }

    def test_null_id(self, capsys: pytest.CaptureFixture[str]) -> None:
        bridge.write_jsonrpc_error(None, -32600, "invalid request")
        captured = capsys.readouterr()
        parsed = json.loads(captured.out.strip())
        assert parsed["id"] is None


class TestParseArgs:
    def test_minimal(self) -> None:
        args = bridge.parse_args(["https://localhost:3443/mcp"])
        assert args.url == "https://localhost:3443/mcp"
        assert args.header == []
        assert args.header_env == []
        assert args.header_cmd == []
        assert args.ca_file is None
        assert args.bearer_env is None

    def test_header(self) -> None:
        args = bridge.parse_args(["https://x", "--header", "Authorization: Bearer tok"])
        assert args.header == ["Authorization: Bearer tok"]

    def test_ca_file(self) -> None:
        args = bridge.parse_args(["https://x", "--ca-file", "/path/to/ca.pem"])
        assert args.ca_file == "/path/to/ca.pem"


class TestResolveHeaders:
    def test_literal_header(self) -> None:
        args = bridge.parse_args(["https://x", "--header", "X-Custom: value"])
        headers = bridge.resolve_headers(args)
        assert headers == {"X-Custom": "value"}

    def test_header_env(self) -> None:
        args = bridge.parse_args(["https://x", "--header-env", "Authorization=TEST_TOKEN"])
        with patch.dict("os.environ", {"TEST_TOKEN": "Bearer secret"}):
            headers = bridge.resolve_headers(args)
        assert headers == {"Authorization": "Bearer secret"}

    def test_header_env_missing_exits(self) -> None:
        args = bridge.parse_args(["https://x", "--header-env", "Authorization=NONEXISTENT_VAR"])
        with patch.dict("os.environ", {}, clear=False), pytest.raises(SystemExit):
            bridge.resolve_headers(args)

    def test_bearer_env(self) -> None:
        args = bridge.parse_args(["https://x", "--bearer-env", "MY_TOKEN"])
        with patch.dict("os.environ", {"MY_TOKEN": "tok123"}):
            headers = bridge.resolve_headers(args)
        assert headers == {"Authorization": "Bearer tok123"}

    def test_header_cmd(self) -> None:
        args = bridge.parse_args(["https://x", "--header-cmd", "X-Secret=echo hunter2"])
        headers = bridge.resolve_headers(args)
        assert headers == {"X-Secret": "hunter2"}

    def test_header_cmd_failure_exits(self) -> None:
        args = bridge.parse_args(["https://x", "--header-cmd", "X-Secret=false"])
        with pytest.raises(SystemExit):
            bridge.resolve_headers(args)
