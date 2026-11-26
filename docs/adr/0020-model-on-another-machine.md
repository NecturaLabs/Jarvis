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
   A backend that listens beyond loopback speaks plain HTTP, so setup and the backend's doctor warn
   that its key and the conversations cross the network unencrypted.
   An address rule holds for host names as for IP literals: a name must resolve, and every address it
   resolves to must pass the check literals pass (loopback, `10/8`, `172.16/12`, `192.168/16`,
   `100.64/10` for Tailscale, `fc00::/7`). Setup, the doctor and each connection make it. It cannot
   stop a resolver that answers differently between the check and the connection, which is why the
   certificate chain, checked against the host name, stays the guard for HTTPS. A `tls_ca` file is
   the only trust for its endpoint; a self-signed server certificate must be usable as a CA.
   An SSH forward does not outlive a reboot or logout, so the doctor says so and gives the command.
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
default local setup, connection retries (a remote probe is one short attempt whose answer is reused
for a few seconds), key handling, name resolution, TLS trust against a local server and the
client/backend service boundaries.
Live Strata authentication, optional capability fields and tunnel interoperability await a run
on the owner's machines. See [configuration.md](../configuration.md#model-on-another-machine).
