# ADR: Where the model can run

**Status:** proposed
**Date:** 2026-11-26

## Context

Strata and its model hold most of Jarvis's RAM and VRAM. The owner wants to be able to put the backend
on another machine. Jarvis defaults to one PC under [ADR 0005](0005-local-only.md). Model inference
needs access to conversation text, tool results and pictures. Audio and actions belong on the PC
Jarvis controls.

## Decisions for the owner to confirm

1. **Backend scope.** Move only Strata and its model. Running the whole agent remotely would require
   sending audio across the network and exposing the client's PC tools through a new interface.
   Wake word, transcription, agent, tool policy, audit, PC actions and speech stay on the client.
   Backend-only setup writes the Strata profile and a headless user service.
2. **Transport.** Start with a loopback-bound backend and SSH forwarding. OpenSSH authenticates the
   host and encrypts requests, replies and key transfer. For direct network access, HTTPS verifies
   the host name and trust chain. The client can pin a private CA or self-signed server certificate
   with `strata.tls_ca`. A separate leaf fingerprint setting would require rotation and a second
   verification path. WireGuard and Tailscale can encrypt the link for a backend bound to a private
   IP. HTTP beyond loopback requires explicit opt-in and a doctor warning. A private IP alone does
   not make plaintext safe. The key lives in private state and is bound to the configured URL.
3. **Coding mode.** Apply the client overlay from [ADR 0017](0017-coding-mode.md) and keep serving the
   same remote model and context. Both the typed tool and the command say what changed. The local
   client contract documents chat completions, model listing and load/unload for the current model.
   It specifies no authenticated profile replacement or model switch. The backend owner manages the
   served model.
4. **Product rule.** Amend [AGENTS.md](../../AGENTS.md) to allow Strata on the owner's own machine on
   their network. Keep the ban on hosted model backends and paid APIs. All-local setup remains the
   default.

## Verification

Tests use a fake systemd and HTTP mock transport. They cover byte-for-byte compatibility of the
default local setup, connection retries, key handling and the client/backend service boundaries.
Live Strata authentication, optional capability fields and tunnel interoperability await a run
on the owner's machines. See [configuration.md](../configuration.md#model-on-another-machine).
