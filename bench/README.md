# Benchmark harness

Reproduces the LoCoMo results in [docs/BENCHMARKS.md](../docs/BENCHMARKS.md). Bring your own copy of `locomo10.json`
([LoCoMo](https://github.com/snap-research/locomo), CC BY-NC 4.0 — not redistributed here).

```bash
# 1. Build each conversation into its own memory and write each question's context (no model needed).
npx tsx bench/locomo.ts --data locomo10.json --out runs/memtrails --room 6000

#    The same with consolidation and sleep (a language model; about $0.07 a conversation with GPT-6 Luna):
MEMTRAILS_MODEL=gpt-6-luna OPENAI_API_KEY=… npx tsx bench/locomo.ts --data locomo10.json --out runs/memtrails-llm --consolidate

#    Each room from gather with notes instead of search (one model call a question, about $0.0015 with GPT-6 Luna):
MEMTRAILS_MODEL=gpt-6-luna OPENAI_API_KEY=… npx tsx bench/locomo.ts --data locomo10.json --out runs/memtrails-gather --consolidate --gather 40 --notes

# 2. Score with no model: do the gold answer's words reach the reader? Against another run, per question.
npx tsx bench/score.ts runs/memtrails/locomo.jsonl runs/memtrails-llm/locomo.jsonl

# 3. Judged answers: one reader, LongMemEval's yes/no judge.
OPENAI_API_KEY=… npx tsx bench/judge.ts runs/memtrails/locomo.jsonl --reader gpt-6-luna --judge gpt-5.5 --chars 6000
```

The published comparisons used the held-out conversations conv-41, 42, 43, 44, 47, 48, 49 and 50 (`--conversations`),
categories 1–4 (the adversarial category left out), GPT-6 Luna as the reader and GPT-5.5 as the judge, and Mem0's
contexts built by Mem0's own benchmark code, cut to the same room. To compare another system, write its contexts in the
same JSONL shape (`qid`, `question`, `answer`, `category`, `context`) and run steps 2 and 3 on both.

Without `--consolidate` memory holds the records as written: dates are what consolidation works out ("last week"
becomes a date), so a raw build shows far fewer temporal answers — the published numbers used consolidation.

One reading of 166 questions moves by ±3–4 points; of 1,307, by about ±1.2. Compare runs per question
(`score.ts` with two files), not by their totals alone.
