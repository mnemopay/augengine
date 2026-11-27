# Contributing

Thanks for looking. AugEngine is alpha and small, so keep changes focused.

## How this repository works

This repository is a **one-way public mirror** of AugEngine's development monorepo.
Maintainers accept pull requests here, apply them in the development repository, and the
next sync brings them back to `main` here (each sync commit names the source revision).
Your authorship is kept in the upstream commit.

## Before you open a pull request

```bash
npm ci
npm run build
npm test
```

- One concern per pull request. Branch from `main`.
- Build and unit tests must pass (CI runs them on Node 20 and 22).
- Renderer or gameplay changes: attach a headed-Chrome capture or a scorecard line. Numbers
  in this repo are measured, not estimated, and say which GPU they came from.
- No new runtime dependency in `@mnemopay/augengine` without discussion first (see the
  contributor rules at the end of [AUGENGINE-ARCHITECTURE.md](AUGENGINE-ARCHITECTURE.md)).
- Never commit keys. Secrets come from the environment at runtime, never from tracked files.
- Paid model calls must stay behind an explicit click or flag with a spend estimate, as the
  detective game's spend guard does.

## Reporting bugs

Open an issue with the package and version, what you expected, what happened, and your
browser/GPU for anything rendering-related. Security issues: see [SECURITY.md](SECURITY.md).

By contributing you agree that your contribution is licensed under the Apache License 2.0.
