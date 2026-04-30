# mcp-hardhat

Security-first MCP tooling. Python + Go polyglot.

## Rules

- `tools/mcp-stdio-bridge.py` MUST remain a single file with zero external dependencies (stdlib only)
- Python: strict mypy, ruff, pytest. Dev setup: `uv sync --group dev`
- Go: `golangci-lint` with `gosec` enabled
- Never use `ssl.CERT_NONE` or equivalent — always validate TLS certificates
- Secrets are resolved at runtime (env vars, commands), never stored in config files
- New Python tools go in `tools/`, new Go tools go in `cmd/<name>/`
- Tests in `tests/` (Python) or alongside Go packages
