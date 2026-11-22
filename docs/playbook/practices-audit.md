# Practices audit

Each area below is audited weekly against the sources listed, the vendor's documentation for the version
the project pins first. An audit reads the area's code, config and docs against those sources, fixes the
gaps or files them as issues, and only then sets the area's date (UTC, `YYYY-MM-DD`).

| Area | Held to | Last audit |
|---|---|---|
| Python code and packaging | Python 3.14 docs (docs.python.org/3.14), PEP 8, PEP 621, uv docs (docs.astral.sh/uv) | 2026-11-22 (practices audit; gaps fixed in 647d11f, rest in #32) |
| Lint, types and formatting | Ruff docs (docs.astral.sh/ruff, 0.17.5), pyright configuration docs (1.1.414), agentic-setup `lint/python` | 2026-11-22 (practices audit; strict pytest config fixed; suppressions and imports in #32) |
| Tests | pytest 9.2 docs (docs.pytest.org), agentic-setup `testing` skill | 2026-11-22 (practices audit; gaps in #32) |
| Audio pipeline | PipeWire docs (docs.pipewire.org), python-sounddevice docs, LiveKit WakeWord README 2.1.3, Silero VAD README | 2026-11-22 (practices audit; fixes in 884c638; stalled-microphone watchdog and the rest in #29) |
| Voxtype integration | Voxtype `docs/INTEGRATIONS.md` and `CONFIGURATION.md` for the installed release | 2026-11-22 (practices audit; `-q` and settings fixed; open in #29, owner decisions in #34) |
| Local model serving | Strata docs for the pinned release (`docs/TERMINAL.md`, `DETAILS.md`, `AMD_HIP.md`) | 2026-11-22 (practices audit; warm honours Retry-After; prefill and VRAM reserve in #33, idle unload decision in #34) |
| Agent loop and tool calling | OpenAI Chat Completions function-calling guide, Strata API notes, OWASP Top 10 for LLM Applications | 2026-11-22 (practices audit; PC-audio text marked as data; tool count in #31) |
| Policy, confirmation and prompt-injection defence | OWASP Top 10 for LLM Applications, OWASP ASVS 5.0, `docs/adr/0002-typed-operation-policy.md` | 2026-11-22 (practices audit; hidden home files shut, typos in classes refused; rest in #30, `strong` under bypass in #34) |
| Spotify integration | Spotify Web API reference, Authorization Code with PKCE tutorial, redirect URI rules, Feb 2026 and Jul 2026 changes | 2026-11-22 (practices audit; rate limits now wait out Retry-After; no gap left open) |
| Speech output | Kokoro (hexgrad/Kokoro-82M) and Piper (OHF-Voice/piper1-gpl) docs for the pinned releases | 2026-11-22 (practices audit; Kokoro chunking and inputs fixed; phonemes in #29, Piper licence in #34) |
| Web research | SearXNG admin and search API docs, trafilatura docs | 2026-11-22 (practices audit; hidden page text pruned, reads capped; SearXNG read-only in #33) |
| systemd user services | systemd.service, systemd.exec and systemd.resource-control man pages (systemd in Arch), Arch Wiki systemd/User | 2026-11-22 (practices audit; `Type=exec`, timeouts and ordering fixed; hardening and game-side limits in #33) |
| CI and supply chain | GitHub Actions security hardening guide, Dependabot docs, OpenSSF Scorecard checks | 2026-11-21 (open-source audit, then a comparison with the sibling project Canopy; gaps filed as #24, #25, #26) |
| Repository basics and docs | GitHub community health docs, opensource.guide, Contributor Covenant 3.0, Keep a Changelog, REUSE 3.3, agentic-setup `project-setup` repository basics | 2026-11-21 (open-source audit; owner decisions in #26) |
