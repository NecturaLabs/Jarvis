# Security policy

Jarvis is a private NecturaLabs project that acts on its owner's PC and accounts by voice, so security
issues matter even though it has no public users.

## Reporting a vulnerability

Open a GitHub security advisory on this repository (Security > Advisories > New draft advisory), or tell a
NecturaLabs maintainer directly. Do not file a public issue for a vulnerability, and never paste tokens,
recordings or personal data into a report.

## Supported versions

Only the latest release on `main` and the current `dev` branch are supported.

## Scope

Of particular interest: anything that lets speech, web content, files or Spotify metadata cause an action the
owner did not confirm; the policy engine and confirmation flow; token handling for Spotify; and the local
services Jarvis talks to (Strata, SearXNG), which must stay bound to localhost.
