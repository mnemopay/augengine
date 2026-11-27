# Security policy

## Reporting a vulnerability

Please report security issues privately, not in a public issue:

- Email **augengine@mnemopay.com**, or
- use GitHub's private vulnerability reporting ("Report a vulnerability" on the
  [Security tab](https://github.com/mnemopay/augengine/security)).

Include what you found, how to reproduce it, and which package and version it affects.
We aim to acknowledge reports within 5 working days and to agree a disclosure date with you.

## Supported versions

AugEngine is alpha software. Fixes land on `main` and in the next alpha release of
`@mnemopay/augengine` / `@mnemopay/augengine-lod`; older alphas are not patched.

## Scope notes

- The detective game's model path keeps its API key server-side (`examples/detective/server`)
  and closes itself when spend limits are reached (`server/key-guard.ts`). Reports of ways to
  extract the key, bypass the spend guard or make the server call the model for free are in scope.
- Never include real API keys in reports or pull requests.
