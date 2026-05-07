"""Tests for mcp-stdio-bridge."""

from __future__ import annotations

import importlib.util
import io
import json
import ssl
from http.client import HTTPResponse
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Import the bridge as a module from its file path.
_BRIDGE_PATH = Path(__file__).resolve().parent.parent / "tools" / "mcp-stdio-bridge.py"
_spec = importlib.util.spec_from_file_location("mcp_stdio_bridge", _BRIDGE_PATH)
assert _spec is not None and _spec.loader is not None
bridge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bridge)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(body: str = "", session_id: str | None = None) -> MagicMock:
    """Create a mock urllib response with optional Mcp-Session-Id header."""
    resp = MagicMock(spec=HTTPResponse)
    resp.read.return_value = body.encode()
    headers: dict[str, str] = {}
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    resp.headers = MagicMock()
    resp.headers.get = lambda key, default=None: headers.get(key, default)
    return resp


def _make_http_error(code: int, body: str = "") -> Any:
    """Create a mock urllib.error.HTTPError."""
    import urllib.error

    err = urllib.error.HTTPError(
        url="https://test/mcp",
        code=code,
        msg="error",
        hdrs=MagicMock(),
        fp=io.BytesIO(body.encode()),
    )
    return err


# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------


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

    def test_timeout_default(self) -> None:
        args = bridge.parse_args(["https://x"])
        assert args.timeout == 30.0

    def test_timeout_custom(self) -> None:
        args = bridge.parse_args(["https://x", "--timeout", "60"])
        assert args.timeout == 60.0

    def test_debug_flag(self) -> None:
        args = bridge.parse_args(["https://x", "--debug"])
        assert args.debug is True

    def test_debug_default_off(self) -> None:
        args = bridge.parse_args(["https://x"])
        assert args.debug is False


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

    def test_header_cmd_empty_output_exits(self) -> None:
        args = bridge.parse_args(["https://x", "--header-cmd", "X-Secret=printf ''"])
        with pytest.raises(SystemExit, match="empty output"):
            bridge.resolve_headers(args)

    def test_bearer_env_missing_exits(self) -> None:
        args = bridge.parse_args(["https://x", "--bearer-env", "NONEXISTENT_TOKEN"])
        with patch.dict("os.environ", {}, clear=False), pytest.raises(SystemExit, match="not set"):
            bridge.resolve_headers(args)


# ---------------------------------------------------------------------------
# New tests: BridgeState
# ---------------------------------------------------------------------------


class TestBridgeState:
    def test_initial_state(self) -> None:
        state = bridge.BridgeState()
        assert state.session_id is None
        assert state.generation == 0

    def test_set_session_bumps_generation(self) -> None:
        state = bridge.BridgeState()
        state.set_session("abc-123")
        assert state.session_id == "abc-123"
        assert state.generation == 1

    def test_set_session_same_value_no_bump(self) -> None:
        state = bridge.BridgeState()
        state.set_session("abc-123")
        state.set_session("abc-123")
        assert state.generation == 1

    def test_clear_session(self) -> None:
        state = bridge.BridgeState()
        state.set_session("abc-123")
        state.clear_session()
        assert state.session_id is None
        assert state.generation == 2

    def test_debug_logs_to_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        state = bridge.BridgeState()
        state.set_session("abc", debug=True)
        captured = capsys.readouterr()
        assert "session updated" in captured.err
        assert "generation=1" in captured.err


# ---------------------------------------------------------------------------
# New tests: HttpMcpError
# ---------------------------------------------------------------------------


class TestHttpMcpError:
    def test_body_truncated_at_4kb(self) -> None:
        err = bridge.HttpMcpError(400, "x" * 8192, "tools/call")
        assert len(err.body) == 4096

    def test_str_format(self) -> None:
        err = bridge.HttpMcpError(404, "not found", "tools/list")
        assert "404" in str(err)
        assert "tools/list" in str(err)

    def test_short_body_unchanged(self) -> None:
        err = bridge.HttpMcpError(500, "short", "ping")
        assert err.body == "short"


# ---------------------------------------------------------------------------
# New tests: is_stale_session_failure
# ---------------------------------------------------------------------------


class TestIsStaleSessionFailure:
    def test_400_with_session_term(self) -> None:
        err = bridge.HttpMcpError(400, "invalid session id", "tools/call")
        assert bridge.is_stale_session_failure(err, session_was_sent=True) is True

    def test_400_with_expired(self) -> None:
        err = bridge.HttpMcpError(400, "token expired", "tools/call")
        assert bridge.is_stale_session_failure(err, session_was_sent=True) is True

    def test_404_empty_body(self) -> None:
        err = bridge.HttpMcpError(404, "", "tools/call")
        assert bridge.is_stale_session_failure(err, session_was_sent=True) is True

    def test_404_whitespace_body(self) -> None:
        err = bridge.HttpMcpError(404, "   \n  ", "tools/call")
        assert bridge.is_stale_session_failure(err, session_was_sent=True) is True

    def test_no_session_sent(self) -> None:
        err = bridge.HttpMcpError(400, "session expired", "tools/call")
        assert bridge.is_stale_session_failure(err, session_was_sent=False) is False

    def test_initialize_never_stale(self) -> None:
        err = bridge.HttpMcpError(400, "session expired", "initialize")
        assert bridge.is_stale_session_failure(err, session_was_sent=True) is False

    def test_401_never_stale(self) -> None:
        err = bridge.HttpMcpError(401, "unauthorized", "tools/call")
        assert bridge.is_stale_session_failure(err, session_was_sent=True) is False

    def test_403_never_stale(self) -> None:
        err = bridge.HttpMcpError(403, "forbidden", "tools/call")
        assert bridge.is_stale_session_failure(err, session_was_sent=True) is False

    def test_500_never_stale(self) -> None:
        err = bridge.HttpMcpError(500, "session error", "tools/call")
        assert bridge.is_stale_session_failure(err, session_was_sent=True) is False

    def test_400_unrelated_body(self) -> None:
        err = bridge.HttpMcpError(400, "bad json format in request payload", "tools/call")
        assert bridge.is_stale_session_failure(err, session_was_sent=True) is False

    def test_mcp_session_id_in_body(self) -> None:
        err = bridge.HttpMcpError(400, "Missing Mcp-Session-Id header", "tools/call")
        assert bridge.is_stale_session_failure(err, session_was_sent=True) is True


# ---------------------------------------------------------------------------
# New tests: http_post_once
# ---------------------------------------------------------------------------


class TestHttpPostOnce:
    CFG = bridge.BridgeConfig("https://test/mcp", {}, ssl.create_default_context(), 30.0)

    def test_success_returns_parsed_body_and_session(self) -> None:
        resp = _make_response('{"jsonrpc":"2.0","id":1,"result":{}}', session_id="sess-1")
        with patch("urllib.request.urlopen", return_value=resp):
            result, session = bridge.http_post_once(self.CFG, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, None)
        assert result == {"jsonrpc": "2.0", "id": 1, "result": {}}
        assert session == "sess-1"

    def test_sends_session_header_when_present(self) -> None:
        resp = _make_response("{}")
        with patch("urllib.request.urlopen", return_value=resp) as mock_open:
            bridge.http_post_once(self.CFG, {"jsonrpc": "2.0", "id": 1, "method": "ping"}, "my-session")
        req = mock_open.call_args[0][0]
        assert req.get_header("Mcp-session-id") == "my-session"

    def test_no_session_header_when_none(self) -> None:
        resp = _make_response("{}")
        with patch("urllib.request.urlopen", return_value=resp) as mock_open:
            bridge.http_post_once(self.CFG, {"jsonrpc": "2.0", "id": 1, "method": "ping"}, None)
        req = mock_open.call_args[0][0]
        assert req.get_header("Mcp-session-id") is None

    def test_http_error_raises_http_mcp_error(self) -> None:
        err = _make_http_error(400, "bad request")
        with patch("urllib.request.urlopen", side_effect=err), pytest.raises(bridge.HttpMcpError) as exc_info:
            bridge.http_post_once(self.CFG, {"jsonrpc": "2.0", "id": 1, "method": "tools/call"}, None)
        assert exc_info.value.status == 400
        assert exc_info.value.body == "bad request"
        assert exc_info.value.method == "tools/call"

    def test_timeout_passed_to_urlopen(self) -> None:
        cfg = bridge.BridgeConfig("https://test/mcp", {}, ssl.create_default_context(), 42.0)
        resp = _make_response("{}")
        with patch("urllib.request.urlopen", return_value=resp) as mock_open:
            bridge.http_post_once(cfg, {"jsonrpc": "2.0", "id": 1, "method": "ping"}, None)
        assert mock_open.call_args[1]["timeout"] == 42.0

    def test_no_response_session(self) -> None:
        resp = _make_response('{"result": "ok"}')
        with patch("urllib.request.urlopen", return_value=resp):
            _, session = bridge.http_post_once(self.CFG, {"jsonrpc": "2.0", "id": 1, "method": "ping"}, None)
        assert session is None


# ---------------------------------------------------------------------------
# New tests: http_post (retry logic)
# ---------------------------------------------------------------------------


class TestHttpPost:
    CFG = bridge.BridgeConfig("https://test/mcp", {}, ssl.create_default_context(), 30.0)
    CFG_DEBUG = bridge.BridgeConfig("https://test/mcp", {}, ssl.create_default_context(), 30.0, debug=True)

    def test_success_updates_session(self) -> None:
        state = bridge.BridgeState()
        resp = _make_response('{"result":"ok"}', session_id="sess-new")
        with patch("urllib.request.urlopen", return_value=resp):
            bridge.http_post(self.CFG, {"method": "tools/list", "id": 1}, state)
        assert state.session_id == "sess-new"

    def test_always_updates_session_not_just_first_time(self) -> None:
        """Regression: original code only stored session when `not session`."""
        state = bridge.BridgeState()
        state.set_session("old-session")

        resp = _make_response('{"result":"ok"}', session_id="new-session")
        with patch("urllib.request.urlopen", return_value=resp):
            bridge.http_post(self.CFG, {"method": "tools/list", "id": 1}, state)
        assert state.session_id == "new-session"

    def test_initialize_clears_session_first(self) -> None:
        state = bridge.BridgeState()
        state.set_session("old")

        resp = _make_response('{"result":"ok"}', session_id="fresh")
        with patch("urllib.request.urlopen", return_value=resp) as mock_open:
            bridge.http_post(self.CFG, {"method": "initialize", "id": 1}, state)
        req = mock_open.call_args[0][0]
        assert req.get_header("Mcp-session-id") is None
        assert state.session_id == "fresh"

    def test_stale_session_retries_without_session(self) -> None:
        state = bridge.BridgeState()
        state.set_session("stale-sess")

        stale_err = _make_http_error(400, "unknown session")
        retry_resp = _make_response('{"result":"recovered"}', session_id="new-sess")

        call_count = 0

        def mock_urlopen(req: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise stale_err
            return retry_resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen) as mock_open:
            result = bridge.http_post(self.CFG_DEBUG, {"method": "tools/call", "id": 1}, state)

        assert result == {"result": "recovered"}
        assert state.session_id == "new-sess"
        assert call_count == 2
        retry_req = mock_open.call_args_list[1][0][0]
        assert retry_req.get_header("Mcp-session-id") is None

    def test_stale_retry_logs_to_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        state = bridge.BridgeState()
        state.set_session("stale")

        stale_err = _make_http_error(400, "unknown session")
        retry_resp = _make_response('{"ok":true}', session_id="new")

        calls: list[Any] = [stale_err, retry_resp]

        def mock_urlopen(req: Any, **kwargs: Any) -> Any:
            val = calls.pop(0)
            if isinstance(val, Exception):
                raise val
            return val

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            bridge.http_post(self.CFG_DEBUG, {"method": "tools/call", "id": 1}, state)

        captured = capsys.readouterr()
        assert "stale session suspected" in captured.err
        assert "status=400" in captured.err
        assert "cleared session and retrying once" in captured.err

    def test_stale_retry_updates_session(self) -> None:
        state = bridge.BridgeState()
        state.set_session("old")

        stale_err = _make_http_error(400, "session expired")
        retry_resp = _make_response('{"ok":true}', session_id="recovered-sess")

        calls = [stale_err, retry_resp]

        def mock_urlopen(req: Any, **kwargs: Any) -> Any:
            val = calls.pop(0)
            if isinstance(val, Exception):
                raise val
            return val

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            bridge.http_post(self.CFG, {"method": "tools/call", "id": 1}, state)

        assert state.session_id == "recovered-sess"

    def test_non_stale_error_reraises(self) -> None:
        state = bridge.BridgeState()
        state.set_session("sess")

        err = _make_http_error(401, "unauthorized")
        with patch("urllib.request.urlopen", side_effect=err), pytest.raises(bridge.HttpMcpError) as exc_info:
            bridge.http_post(self.CFG, {"method": "tools/call", "id": 1}, state)
        assert exc_info.value.status == 401

    def test_no_retry_when_no_session_sent(self) -> None:
        state = bridge.BridgeState()  # no session
        err = _make_http_error(400, "session expired")
        with patch("urllib.request.urlopen", side_effect=err), pytest.raises(bridge.HttpMcpError):
            bridge.http_post(self.CFG, {"method": "tools/call", "id": 1}, state)

    def test_no_retry_on_initialize(self) -> None:
        state = bridge.BridgeState()
        state.set_session("old")
        err = _make_http_error(400, "session expired")
        with patch("urllib.request.urlopen", side_effect=err), pytest.raises(bridge.HttpMcpError):
            bridge.http_post(self.CFG, {"method": "initialize", "id": 1}, state)

    def test_exactly_one_retry(self) -> None:
        """Retry also fails \u2014 should raise, not loop."""
        state = bridge.BridgeState()
        state.set_session("stale")

        err1 = _make_http_error(400, "unknown session")
        err2 = _make_http_error(400, "still broken")

        calls = [err1, err2]

        def mock_urlopen(req: Any, **kwargs: Any) -> Any:
            raise calls.pop(0)

        with patch("urllib.request.urlopen", side_effect=mock_urlopen), pytest.raises(bridge.HttpMcpError) as exc_info:
            bridge.http_post(self.CFG, {"method": "tools/call", "id": 1}, state)
        assert exc_info.value.status == 400
        assert len(calls) == 0

    def test_notification_uses_same_retry(self) -> None:
        state = bridge.BridgeState()
        state.set_session("stale")

        stale_err = _make_http_error(404, "")
        retry_resp = _make_response("", session_id="new")

        calls: list[Any] = [stale_err, retry_resp]

        def mock_urlopen(req: Any, **kwargs: Any) -> Any:
            val = calls.pop(0)
            if isinstance(val, Exception):
                raise val
            return val

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            bridge.http_post(self.CFG, {"method": "notifications/progress"}, state)

        assert state.session_id == "new"


# ---------------------------------------------------------------------------
# New tests: run() integration
# ---------------------------------------------------------------------------


class TestRun:
    def _run_with_stdin(
        self, lines: list[str], urlopen_side_effect: Any, argv: list[str] | None = None
    ) -> tuple[str, str]:
        """Run the bridge with mocked stdin and urlopen, return (stdout, stderr)."""
        import io as _io

        stdin_text = "\n".join(lines) + "\n"
        if argv is None:
            argv = ["https://test/mcp"]

        with (
            patch("sys.stdin", _io.StringIO(stdin_text)),
            patch("urllib.request.urlopen", side_effect=urlopen_side_effect),
            patch("sys.stdout", new_callable=_io.StringIO) as mock_stdout,
            patch("sys.stderr", new_callable=_io.StringIO) as mock_stderr,
        ):
            bridge.run(argv)
        return mock_stdout.getvalue(), mock_stderr.getvalue()

    def test_request_response_roundtrip(self) -> None:
        resp = _make_response('{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}', session_id="s1")
        stdout, _ = self._run_with_stdin(
            [json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})],
            lambda req, **kw: resp,
        )
        parsed = json.loads(stdout.strip())
        assert parsed == {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}

    def test_notification_no_stdout(self) -> None:
        resp = _make_response("")
        stdout, _ = self._run_with_stdin(
            [json.dumps({"jsonrpc": "2.0", "method": "notifications/progress"})],
            lambda req, **kw: resp,
        )
        assert stdout == ""

    def test_http_error_returns_jsonrpc_error(self) -> None:
        err = _make_http_error(500, "internal error")
        stdout, _ = self._run_with_stdin(
            [json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call"})],
            err,
        )
        parsed = json.loads(stdout.strip())
        assert parsed["error"]["code"] == -32000
        assert "500" in parsed["error"]["message"]

    def test_notification_http_error_logs_stderr(self) -> None:
        err = _make_http_error(502, "bad gateway")
        _, stderr = self._run_with_stdin(
            [json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})],
            err,
        )
        assert "notification forwarding failed" in stderr
        assert "502" in stderr

    def test_notification_generic_error_logs_stderr(self) -> None:
        _, stderr = self._run_with_stdin(
            [json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})],
            ConnectionError("refused"),
        )
        assert "notification forwarding failed" in stderr
        assert "ConnectionError" in stderr

    def test_runtime_error_returns_jsonrpc_error(self) -> None:
        stdout, _ = self._run_with_stdin(
            [json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call"})],
            RuntimeError("broken pipe"),
        )
        parsed = json.loads(stdout.strip())
        assert parsed["error"]["code"] == -32000
        assert "broken pipe" in parsed["error"]["message"]

    def test_generic_exception_returns_jsonrpc_error(self) -> None:
        stdout, _ = self._run_with_stdin(
            [json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call"})],
            OSError("connection reset"),
        )
        parsed = json.loads(stdout.strip())
        assert parsed["error"]["code"] == -32000
        assert "OSError" in parsed["error"]["message"]

    def test_empty_lines_skipped(self) -> None:
        resp = _make_response('{"jsonrpc":"2.0","id":1,"result":{}}')
        stdout, _ = self._run_with_stdin(
            ["", "  ", json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}), ""],
            lambda req, **kw: resp,
        )
        lines = [line for line in stdout.strip().split("\n") if line]
        assert len(lines) == 1

    def test_stale_session_recovery_in_run(self) -> None:
        init_resp = _make_response('{"jsonrpc":"2.0","id":1,"result":{}}', session_id="sess-1")
        stale_err = _make_http_error(400, "unknown session")
        retry_resp = _make_response('{"jsonrpc":"2.0","id":2,"result":{"recovered":true}}', session_id="sess-2")

        responses: list[Any] = [init_resp, stale_err, retry_resp]

        def side_effect(req: Any, **kw: Any) -> Any:
            val = responses.pop(0)
            if isinstance(val, Exception):
                raise val
            return val

        stdout, _ = self._run_with_stdin(
            [
                json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
                json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call"}),
            ],
            side_effect,
        )
        lines = [json.loads(line) for line in stdout.strip().split("\n") if line]
        assert len(lines) == 2
        assert lines[1]["result"] == {"recovered": True}

    def test_no_response_body_emits_nothing(self) -> None:
        resp = _make_response("")
        stdout, _ = self._run_with_stdin(
            [json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"})],
            lambda req, **kw: resp,
        )
        assert stdout == ""

    def test_debug_flag_from_cli(self) -> None:
        init_resp = _make_response('{"jsonrpc":"2.0","id":1,"result":{}}', session_id="s1")
        _, stderr = self._run_with_stdin(
            [json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"})],
            lambda req, **kw: init_resp,
            argv=["https://test/mcp", "--debug"],
        )
        assert "session updated" in stderr


# ---------------------------------------------------------------------------
# New tests: BridgeState cached_init
# ---------------------------------------------------------------------------


class TestBridgeStateCachedInit:
    def test_default_none(self) -> None:
        state = bridge.BridgeState()
        assert state.cached_init is None


# ---------------------------------------------------------------------------
# New tests: BridgeConfig recover_stale_session
# ---------------------------------------------------------------------------


class TestBridgeConfigRecoverFlag:
    def test_default_true(self) -> None:
        cfg = bridge.BridgeConfig("https://x", {}, ssl.create_default_context(), 30.0)
        assert cfg.recover_stale_session is True

    def test_explicit_false(self) -> None:
        cfg = bridge.BridgeConfig(
            "https://x",
            {},
            ssl.create_default_context(),
            30.0,
            recover_stale_session=False,
        )
        assert cfg.recover_stale_session is False


# ---------------------------------------------------------------------------
# New tests: --recover-stale-session CLI flag
# ---------------------------------------------------------------------------


class TestParseArgsRecoverFlag:
    def test_default_true(self) -> None:
        args = bridge.parse_args(["https://x"])
        assert args.recover_stale_session is True

    def test_explicit_no(self) -> None:
        args = bridge.parse_args(["https://x", "--no-recover-stale-session"])
        assert args.recover_stale_session is False

    def test_explicit_yes(self) -> None:
        args = bridge.parse_args(["https://x", "--recover-stale-session"])
        assert args.recover_stale_session is True


# ---------------------------------------------------------------------------
# New tests: _is_recoverable_session_error
# ---------------------------------------------------------------------------


class TestIsRecoverableSessionError:
    def test_recoverable_true(self) -> None:
        body = {
            "jsonrpc": "2.0",
            "id": 3,
            "error": {
                "code": -32001,
                "message": "MCP session expired or unknown",
                "data": {"recoverable": True, "retry": "initialize"},
            },
        }
        assert bridge._is_recoverable_session_error(body) is True

    def test_recoverable_false(self) -> None:
        body = {
            "jsonrpc": "2.0",
            "id": 3,
            "error": {"code": -32001, "data": {"recoverable": False}},
        }
        assert bridge._is_recoverable_session_error(body) is False

    def test_no_data_field(self) -> None:
        body = {"error": {"code": -32001, "message": "x"}}
        assert bridge._is_recoverable_session_error(body) is False

    def test_wrong_code(self) -> None:
        body = {"error": {"code": -32000, "data": {"recoverable": True}}}
        assert bridge._is_recoverable_session_error(body) is False

    def test_no_error_field(self) -> None:
        body = {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}
        assert bridge._is_recoverable_session_error(body) is False

    def test_non_dict(self) -> None:
        assert bridge._is_recoverable_session_error(None) is False
        assert bridge._is_recoverable_session_error("string") is False
        assert bridge._is_recoverable_session_error([1, 2, 3]) is False

    def test_data_not_dict(self) -> None:
        body = {"error": {"code": -32001, "data": "string"}}
        assert bridge._is_recoverable_session_error(body) is False


# ---------------------------------------------------------------------------
# New tests: in-band recovery in http_post + run()
# ---------------------------------------------------------------------------


_RECOVERABLE_ERROR_BODY = json.dumps(
    {
        "jsonrpc": "2.0",
        "id": 99,
        "error": {
            "code": -32001,
            "message": "MCP session expired or unknown",
            "data": {"recoverable": True, "retry": "initialize", "sessionId": "old"},
        },
    }
)

_INIT_PAYLOAD: dict[str, object] = {
    "jsonrpc": "2.0",
    "id": 0,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-11-25",
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "0.1.0"},
    },
}


class TestInBandRecovery:
    CFG = bridge.BridgeConfig("https://test/mcp", {}, ssl.create_default_context(), 30.0)
    CFG_DEBUG = bridge.BridgeConfig("https://test/mcp", {}, ssl.create_default_context(), 30.0, debug=True)
    CFG_NO_RECOVER = bridge.BridgeConfig(
        "https://test/mcp",
        {},
        ssl.create_default_context(),
        30.0,
        recover_stale_session=False,
    )

    def _state_with_cached_init(self) -> Any:
        state = bridge.BridgeState()
        state.cached_init = dict(_INIT_PAYLOAD)
        state.set_session("stale-sess")
        return state

    def test_initialize_caches_payload(self) -> None:
        state = bridge.BridgeState()
        resp = _make_response('{"result":"ok"}', session_id="s1")
        with patch("urllib.request.urlopen", return_value=resp):
            bridge.http_post(self.CFG, dict(_INIT_PAYLOAD), state)
        assert state.cached_init is not None
        assert state.cached_init["method"] == "initialize"

    def test_recoverable_error_triggers_silent_reinit(self) -> None:
        state = self._state_with_cached_init()
        stale_resp = _make_response(_RECOVERABLE_ERROR_BODY)
        init_resp = _make_response('{"jsonrpc":"2.0","id":"_bridge_reinit_2","result":{}}', session_id="fresh")
        notif_resp = _make_response("")
        replay_resp = _make_response('{"jsonrpc":"2.0","id":99,"result":{"recovered":true}}', session_id="fresh")
        responses: list[Any] = [stale_resp, init_resp, notif_resp, replay_resp]

        def side_effect(req: Any, **kw: Any) -> Any:
            return responses.pop(0)

        with patch("urllib.request.urlopen", side_effect=side_effect):
            result = bridge.http_post(self.CFG, {"jsonrpc": "2.0", "id": 99, "method": "tools/call"}, state)

        assert result == {"jsonrpc": "2.0", "id": 99, "result": {"recovered": True}}
        assert state.session_id == "fresh"

    def test_recovery_disabled_passes_through(self) -> None:
        state = self._state_with_cached_init()
        stale_resp = _make_response(_RECOVERABLE_ERROR_BODY)
        with patch("urllib.request.urlopen", return_value=stale_resp) as mock_open:
            result = bridge.http_post(
                self.CFG_NO_RECOVER,
                {"jsonrpc": "2.0", "id": 99, "method": "tools/call"},
                state,
            )
        assert mock_open.call_count == 1
        assert isinstance(result, dict)
        assert result["error"]["code"] == -32001

    def test_initialize_method_does_not_trigger_recovery(self) -> None:
        state = bridge.BridgeState()
        stale_resp = _make_response(_RECOVERABLE_ERROR_BODY)
        with patch("urllib.request.urlopen", return_value=stale_resp) as mock_open:
            result = bridge.http_post(self.CFG, dict(_INIT_PAYLOAD), state)
        assert mock_open.call_count == 1
        assert isinstance(result, dict)
        assert result["error"]["code"] == -32001

    def test_reinit_post_failure_surfaces_original_error(self) -> None:
        state = self._state_with_cached_init()
        stale_resp = _make_response(_RECOVERABLE_ERROR_BODY)
        reinit_err = _make_http_error(503, "service unavailable")
        responses: list[Any] = [stale_resp, reinit_err]

        def side_effect(req: Any, **kw: Any) -> Any:
            val = responses.pop(0)
            if isinstance(val, Exception):
                raise val
            return val

        with patch("urllib.request.urlopen", side_effect=side_effect):
            result = bridge.http_post(self.CFG, {"jsonrpc": "2.0", "id": 99, "method": "tools/call"}, state)
        assert isinstance(result, dict)
        assert result["error"]["code"] == -32001

    def test_reinit_returns_jsonrpc_error_surfaces_original(self) -> None:
        state = self._state_with_cached_init()
        stale_resp = _make_response(_RECOVERABLE_ERROR_BODY)
        init_error_resp = _make_response(
            '{"jsonrpc":"2.0","id":"_bridge_reinit_2","error":{"code":-32600,"message":"invalid params"}}'
        )
        responses: list[Any] = [stale_resp, init_error_resp]

        def side_effect(req: Any, **kw: Any) -> Any:
            return responses.pop(0)

        with patch("urllib.request.urlopen", side_effect=side_effect):
            result = bridge.http_post(self.CFG, {"jsonrpc": "2.0", "id": 99, "method": "tools/call"}, state)
        assert isinstance(result, dict)
        assert result["error"]["code"] == -32001

    def test_replay_failure_surfaces_original(self) -> None:
        state = self._state_with_cached_init()
        stale_resp = _make_response(_RECOVERABLE_ERROR_BODY)
        init_resp = _make_response('{"jsonrpc":"2.0","id":"_bridge_reinit_2","result":{}}', session_id="fresh")
        notif_resp = _make_response("")
        replay_err = _make_http_error(500, "server error")
        responses: list[Any] = [stale_resp, init_resp, notif_resp, replay_err]

        def side_effect(req: Any, **kw: Any) -> Any:
            val = responses.pop(0)
            if isinstance(val, Exception):
                raise val
            return val

        with patch("urllib.request.urlopen", side_effect=side_effect):
            result = bridge.http_post(self.CFG, {"jsonrpc": "2.0", "id": 99, "method": "tools/call"}, state)
        assert isinstance(result, dict)
        assert result["error"]["code"] == -32001

    def test_notification_initialized_failure_does_not_block_replay(self) -> None:
        state = self._state_with_cached_init()
        stale_resp = _make_response(_RECOVERABLE_ERROR_BODY)
        init_resp = _make_response('{"jsonrpc":"2.0","id":"_bridge_reinit_2","result":{}}', session_id="fresh")
        notif_err = _make_http_error(404, "not found")
        replay_resp = _make_response('{"jsonrpc":"2.0","id":99,"result":{"ok":true}}', session_id="fresh")
        responses: list[Any] = [stale_resp, init_resp, notif_err, replay_resp]

        def side_effect(req: Any, **kw: Any) -> Any:
            val = responses.pop(0)
            if isinstance(val, Exception):
                raise val
            return val

        with patch("urllib.request.urlopen", side_effect=side_effect):
            result = bridge.http_post(self.CFG, {"jsonrpc": "2.0", "id": 99, "method": "tools/call"}, state)
        assert isinstance(result, dict)
        assert result["result"] == {"ok": True}

    def test_no_cached_init_surfaces_original(self) -> None:
        state = bridge.BridgeState()
        state.set_session("stale")
        # cached_init left as None
        stale_resp = _make_response(_RECOVERABLE_ERROR_BODY)
        with patch("urllib.request.urlopen", return_value=stale_resp) as mock_open:
            result = bridge.http_post(self.CFG, {"jsonrpc": "2.0", "id": 99, "method": "tools/call"}, state)
        assert mock_open.call_count == 1
        assert isinstance(result, dict)
        assert result["error"]["code"] == -32001

    def test_run_integration_recovers_silently(self) -> None:
        """Full stdin -> stdout integration: client never sees the -32001."""
        import io as _io

        init_resp = _make_response('{"jsonrpc":"2.0","id":0,"result":{}}', session_id="s1")
        stale_resp = _make_response(_RECOVERABLE_ERROR_BODY)
        reinit_resp = _make_response('{"jsonrpc":"2.0","id":"_bridge_reinit_2","result":{}}', session_id="s2")
        notif_resp = _make_response("")
        replay_resp = _make_response('{"jsonrpc":"2.0","id":99,"result":{"recovered":true}}', session_id="s2")
        responses: list[Any] = [init_resp, stale_resp, reinit_resp, notif_resp, replay_resp]

        def side_effect(req: Any, **kw: Any) -> Any:
            return responses.pop(0)

        stdin_text = (
            json.dumps(_INIT_PAYLOAD) + "\n" + json.dumps({"jsonrpc": "2.0", "id": 99, "method": "tools/call"}) + "\n"
        )
        with (
            patch("sys.stdin", _io.StringIO(stdin_text)),
            patch("urllib.request.urlopen", side_effect=side_effect),
            patch("sys.stdout", new_callable=_io.StringIO) as mock_stdout,
        ):
            bridge.run(["https://test/mcp"])

        lines = [json.loads(line) for line in mock_stdout.getvalue().strip().split("\n") if line]
        # Two visible responses: initialize result, and recovered tools/call result.
        # The -32001 is silently swallowed by recovery.
        assert len(lines) == 2
        assert lines[1] == {"jsonrpc": "2.0", "id": 99, "result": {"recovered": True}}
        for line in lines:
            assert "error" not in line or line.get("error", {}).get("code") != -32001

    def test_debug_logs_on_successful_recovery(self, capsys: pytest.CaptureFixture[str]) -> None:
        state = self._state_with_cached_init()
        stale_resp = _make_response(_RECOVERABLE_ERROR_BODY)
        init_resp = _make_response('{"jsonrpc":"2.0","id":"_bridge_reinit_2","result":{}}', session_id="fresh")
        notif_resp = _make_response("")
        replay_resp = _make_response('{"jsonrpc":"2.0","id":99,"result":{"ok":true}}', session_id="fresh")
        responses: list[Any] = [stale_resp, init_resp, notif_resp, replay_resp]

        def side_effect(req: Any, **kw: Any) -> Any:
            return responses.pop(0)

        with patch("urllib.request.urlopen", side_effect=side_effect):
            bridge.http_post(self.CFG_DEBUG, {"jsonrpc": "2.0", "id": 99, "method": "tools/call"}, state)

        captured = capsys.readouterr()
        assert "in-band -32001 recoverable error" in captured.err
        assert "silent reinit succeeded" in captured.err

    def test_debug_logs_on_failed_recovery(self, capsys: pytest.CaptureFixture[str]) -> None:
        state = self._state_with_cached_init()
        stale_resp = _make_response(_RECOVERABLE_ERROR_BODY)
        reinit_err = _make_http_error(503, "service unavailable")
        responses: list[Any] = [stale_resp, reinit_err]

        def side_effect(req: Any, **kw: Any) -> Any:
            val = responses.pop(0)
            if isinstance(val, Exception):
                raise val
            return val

        with patch("urllib.request.urlopen", side_effect=side_effect):
            result = bridge.http_post(self.CFG_DEBUG, {"jsonrpc": "2.0", "id": 99, "method": "tools/call"}, state)

        assert isinstance(result, dict)
        assert result["error"]["code"] == -32001
        captured = capsys.readouterr()
        assert "in-band -32001 recoverable error" in captured.err
        assert "silent reinit failed" in captured.err
