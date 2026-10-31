# Contributing

Thank you for helping. Issues and pull requests are welcome.

## Setup

```bash
git clone https://github.com/cmaftuleac/memtrails.git
cd memtrails
npm install
npm run typecheck
npm test                    # unit tests: no network, no model, no Qdrant
npm run test:integration    # the real model and a real Qdrant (fetched into the cache on first run, ~250 MB)
```

## What a change needs

- **Tests.** Unit tests sit beside the code (`*.test.ts`) and use the in-memory Qdrant and word-vector stand-ins in
  `src/core/memory.test.ts`. Anything that touches the index, the model or the public API also gets a case in
  `src/memory.integration.test.ts`.
- **Measurements for retrieval changes.** A change to what search returns or in what order is measured before it is
  merged: on the benchmark harness in `bench/`, by gold evidence and by the answer's words shown, with a per-question
  sign test against the current order (see docs/RESEARCH.md for the method). Say how many variants you tried. A gain
  found only after looking is a lead, not a result.
- **Any language.** No word lists, no language detection, no rules for one script. Test with at least one non-Latin
  script when text handling changes.
- **No new runtime dependencies** without a reason in the pull request.
- **Style.** TypeScript strict, ESM, `.js` import suffixes. Comments say why, in plain words.

## Reporting a bug

Include the version, Node and OS, what you called, what you expected and what happened. For search quality issues, a
small set of records and a query that shows it is the most useful thing you can give.

## Security

See [SECURITY.md](SECURITY.md). Please do not open public issues for vulnerabilities.

## Licence

By contributing you agree that your contribution is licensed under the MIT licence of this project.
