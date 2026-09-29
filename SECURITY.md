# Security policy

This SDK runs inside production applications, so we treat vulnerabilities in it as urgent.

## Reporting a vulnerability

Please **do not open a public issue**. Report privately through either:

- GitHub: **Security → Report a vulnerability** on this repository (private advisory), or
- Email: Founders@skaigroup.tech

Include the affected version, a description, and steps to reproduce. We acknowledge reports within 3 business
days and keep you updated until a fix ships. Our responsible disclosure policy, including safe harbor, is at
https://acp.skaigroup.tech/legal/responsible-disclosure.

## Supported versions

The latest minor release receives security fixes. Upgrade with `pip install --upgrade agent-control-panel`.

## How releases are protected

Releases are built by a GitHub Actions workflow and uploaded to PyPI with Trusted Publishing (short-lived
OIDC credentials, no long-lived PyPI tokens). The upload job waits for a maintainer's approval in a protected
GitHub environment, so a release never goes out without a person approving it.
