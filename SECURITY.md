# Security Policy

## Design principles

mcp-hardhat tools are built with a security-first posture:

### TLS certificate validation

All tools use strict TLS verification. `ssl.CERT_NONE` (or any equivalent that disables certificate validation) is never used. Custom CA bundles are supported via `--ca-file` or the `NODE_EXTRA_CA_CERTS` / `SSL_CERT_FILE` environment variables — this covers self-signed certificates without weakening the TLS posture.

### Secrets at runtime, never at rest

No secret is ever written to a config file or committed to the repository.

- `--header-cmd` resolves secrets from external commands (e.g. macOS Keychain `security find-generic-password`) at process startup
- `--header-env` and `--bearer-env` read from environment variables set by the calling process
- Secrets exist only in process memory for the lifetime of the bridge

### Zero supply-chain risk (Python tools)

Python tools use only the standard library. No PyPI dependencies means no transitive supply-chain exposure.

### Minimal attack surface

- No network listeners — the bridge is a stdin/stdout proxy, not a server
- No file I/O beyond reading the CA bundle
- No dynamic code execution or eval

### Logging

All diagnostic output goes to stderr, never stdout (stdout is the JSON-RPC channel). Log entries must never contain secrets, tokens, header values, or request/response bodies.

**Mandatory log format:**

```text
<timestamp> <service> <action> <status> <message>
```

| Field | Description |
|-------|-------------|
| `timestamp` | ISO 8601 with timezone (e.g. `2026-04-30T14:23:01+02:00`) |
| `service` | Tool name (e.g. `mcp-stdio-bridge`) |
| `action` | What was attempted (e.g. `http_post`, `resolve_header`, `tls_init`) |
| `status` | `ok`, `fail`, or `skip` |
| `message` | Human-readable detail — must not contain secrets |

**Rules:**

- Header values, bearer tokens, and command outputs from `--header-cmd` must never appear in logs
- Request and response bodies must never be logged (they may contain sensitive tool call parameters)
- Log the header *name* but never the header *value* (e.g. `resolve_header ok Authorization` — not the token itself)
- File paths (CA bundles, config) are safe to log
- Error messages from failed commands may be logged, but stderr output from `--header-cmd` must be truncated to avoid leaking partial secrets

## `--header-cmd` and `shell=True`

The `--header-cmd` flag passes user-provided command strings to `subprocess.run` with `shell=True`. This is intentional: the user controls the command, and it enables patterns like Keychain lookups that require shell features. The command runs once at startup with a 5-second timeout.

If you are wrapping `mcp-stdio-bridge.py` in an automated pipeline, ensure the `--header-cmd` value is not derived from untrusted input.

## Reporting vulnerabilities

If you find a security issue, please open a GitHub issue or contact the maintainer directly. We aim to respond within 48 hours.
