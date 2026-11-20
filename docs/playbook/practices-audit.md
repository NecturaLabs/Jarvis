# Practices audit

Each area below is audited weekly against the sources listed, the vendor's documentation for the version
the project pins first. An audit reads the area's code, config and docs against those sources, fixes the
gaps or files them as issues, and only then sets the area's date (UTC, `YYYY-MM-DD`).

| Area | Held to | Last audit |
|---|---|---|
| Python code and packaging | Python 3.14 docs (docs.python.org/3.14), PEP 8, PEP 621, uv docs (docs.astral.sh/uv) | never |
| Lint, types and formatting | Ruff docs (docs.astral.sh/ruff, 0.17.3), pyright configuration docs (1.1.412), agentic-setup `lint/python` | never |
| Tests | pytest 9.2 docs (docs.pytest.org), agentic-setup `testing` skill | never |
| Audio pipeline | PipeWire docs (docs.pipewire.org), python-sounddevice docs, LiveKit WakeWord README 2.1.3, Silero VAD README | never |
| Voxtype integration | Voxtype `docs/INTEGRATIONS.md` and `CONFIGURATION.md` for the installed release | never |
| Local model serving | Strata docs for the pinned release (`docs/TERMINAL.md`, `DETAILS.md`, `AMD_HIP.md`) | never |
| Agent loop and tool calling | OpenAI Chat Completions function-calling guide, Strata API notes, OWASP Top 10 for LLM Applications | never |
| Policy, confirmation and prompt-injection defence | OWASP Top 10 for LLM Applications, OWASP ASVS 5.0, `docs/adr/0002-typed-operation-policy.md` | never |
| Spotify integration | Spotify Web API reference, Authorization Code with PKCE tutorial, redirect URI rules, Feb 2026 and Jul 2026 changes | never |
| Speech output | Piper (OHF-Voice/piper1-gpl) docs for the pinned release | never |
| Web research | SearXNG admin and search API docs, trafilatura docs | never |
| systemd user services | systemd.service, systemd.exec and systemd.resource-control man pages (systemd in Arch), Arch Wiki systemd/User | never |
| CI and supply chain | GitHub Actions security hardening guide, Dependabot docs, OpenSSF Scorecard checks | never |
| Repository basics and docs | GitHub community health docs, agentic-setup `project-setup` repository basics | never |
