# Model eval

`model_eval.py` runs the whole core on 17 made-up code changes (12 with a problem, 5 without) with each model, as LangSmith
experiments on one dataset. See the file for how to run it.

25 November 2026, one run each, with the real FCA rules (the Handbook RAG) and the Consumer Credit Act over MCP:

| Model | Problem found | Right rule cited | No false alarm | Exact quotes | Median time | Price per million tokens, in / out |
|---|---|---|---|---|---|---|
| `gpt-6-luna` | 10/12 | 10/12 | 5/5 | 28/32 | 20 s | $0.10 / $0.50 |
| `gpt-6.1-terra` | 12/12 | 12/12 | 5/5 | 30/39 | 42 s | $1 / $5 |
| `gpt-6.2-sol` | 12/12 | 12/12 | 5/5 | 42/42 | 36 s | $2 / $10 |

- "Problem found" needs a finding marked as a breach or potential breach. In luna's two misses it cited rules in the right
  area, but marked the findings `needs_review` or lower.
- "Exact quotes" counts every FCA quote, checked word for word without a model. It needs an adapter with `check_citation`.
- The cases are few, fairly clear and written by the developer, so 10 against 12 is not strong evidence by itself. The
  quotes are the clearer difference.
- Prices are published rates from third-party pages, not measured cost.
- The scope includes SYSC. The core's default scope does not.
