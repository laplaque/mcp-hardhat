# mcp-hardhat

Security-first tooling for the [Model Context Protocol](https://modelcontextprotocol.io/).

- **Strict TLS** — certificates are always validated; custom CAs supported via `--ca-file`
- **Runtime secrets** — tokens resolved from Keychain, env vars, or commands at startup; nothing at rest
- **Zero dependencies** — Python tools use stdlib only; no PyPI supply-chain risk

## Tools

### mcp-stdio-bridge

Minimal stdio-to-HTTP MCP proxy. Bridges a JSON-RPC stdin/stdout transport to an HTTP(S) MCP server.

```text
python3 tools/mcp-stdio-bridge.py https://localhost:3443/mcp \
  --header-cmd 'Authorization=security find-generic-password -s obsidian-mcp -a bearer -w' \
  --ca-file /etc/ssl/certs/extra-ca-certs.pem
```

#### Auth options

| Flag | Source | Example |
|------|--------|---------|
| `--header` | Literal value | `--header 'Authorization: Bearer xxx'` |
| `--header-env` | Environment variable | `--header-env 'Authorization=MY_TOKEN_VAR'` |
| `--header-cmd` | Command output | `--header-cmd 'Authorization=security find-generic-password ...'` |
| `--bearer-env` | Env var (auto-prefixes `Bearer`) | `--bearer-env MY_TOKEN_VAR` |

#### TLS options

| Flag | Description |
|------|-------------|
| `--ca-file` | Path to a CA bundle (PEM). Falls back to `NODE_EXTRA_CA_CERTS` or `SSL_CERT_FILE` env vars. |

See [SECURITY.md](SECURITY.md) for the full security posture.

## Development

```bash
uv sync --group dev
ruff check tools/ tests/
mypy tools/ tests/
pytest
```

## License

[MPL-2.0](LICENSE)
