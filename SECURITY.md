# Security policy

AtlasRV does not require credentials for its deterministic synthetic workflow.

## Sensitive data

- Never commit vendor credentials, API keys, proprietary datasets, or client data.
- Keep downloaded market data under ignored data/cache paths.
- Treat generated reports as potentially sensitive when they use licensed inputs.
- Prefer environment variables or an external secret manager for provider credentials.

## Reporting a vulnerability

Open a private GitHub security advisory for vulnerabilities that could expose
credentials, execute untrusted input, or corrupt research artefacts. For ordinary
correctness bugs, open a regular issue with a minimal reproducible example.

## Research integrity

A look-ahead leak, silent data mutation, or incorrect P&L attribution is treated
as a correctness and integrity defect. Such fixes require a regression test.
