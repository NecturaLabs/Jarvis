# Governance

Jarvis is a NecturaLabs project with a single maintainer, who decides what is merged and released and
settles disagreements. The maintainer explains decisions in the issue or pull request where they come up;
consequential ones are recorded as ADRs in [docs/adr/](docs/adr/).

## Roles

- **Contributors**: anyone who files an issue, takes part in a discussion or opens a pull request.
  Contributions follow [CONTRIBUTING.md](CONTRIBUTING.md) and the
  [code of conduct](CODE_OF_CONDUCT.md).
- **Maintainers**: people with write access, who triage issues, review and merge pull requests and cut
  releases. Today that is the NecturaLabs maintainer, [@Nectura](https://github.com/Nectura).

## Review

Every pull request into `dev` or `main` is reviewed by the maintainer, and its checks must pass before it
merges. Changes to the security-sensitive areas in [AGENTS.md](AGENTS.md) (the policy engine, the
confirmation flow, the Spotify login, the PC-operation tools and the web tools) also get a security review.
While there is one maintainer, the maintainer's own pull requests are reviewed by someone who did not write
the change, and the findings are answered in the pull request.

## Scope decisions

What Jarvis will and will not do is set by the product rules in [AGENTS.md](AGENTS.md#product-rules) and
[docs/roadmap.md](docs/roadmap.md). A change to those rules is a governance change: it is proposed in a
discussion and, if accepted, recorded as an ADR.

## Becoming a maintainer

A contributor with a record of sustained, careful contributions and reviews may be invited to become a
maintainer. They are added to `.github/CODEOWNERS` and this file is updated.

## Changes to this document

Changes to governance are made by pull request and approved by the maintainer.
