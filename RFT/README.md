# RFT — SQL Correction via Reinforcement Fine-Tuning

Train and evaluate a model that **repairs broken SQL queries**. Each example gives the model an
*intent*, a *broken query*, and the *database schema*; the model must reason about the bug and
return a corrected query that produces the same rows as a known-good ground-truth query.

The project targets [Fireworks AI](https://fireworks.ai/) reinforcement fine-tuning (RFT) and uses
[`eval-protocol`](https://pypi.org/project/eval-protocol/) to define the reward function.

---

## How it works

```
 broken SQL + intent + schema
            │
            ▼
   ┌─────────────────┐      <reasoning> … </reasoning>
   │   LLM (RFT)      │ ───▶ <answer>    … </answer>
   └─────────────────┘
            │
            ▼
   ┌──────────────────────────────┐
   │  evaluator.py (reward fn)     │
   │  • run predicted SQL          │  ← in-memory SQLite, seeded schema
   │  • run ground-truth SQL       │
   │  • compare result row sets    │
   │  • score in [0.0, 1.0]        │
   └──────────────────────────────┘
            │
            ▼
   reward signal → RFT trainer
```

The model is contracted (via the system prompt) to answer in a fixed format:

```
<reasoning>
...step-by-step diagnosis of the bug...
</reasoning>
<answer>
...corrected SQL only...
</answer>
```

The evaluator runs **both** the predicted and ground-truth SQL against a real in-memory SQLite
database with seeded data, then compares the result **row sets** (order-independent). This rewards
semantically-correct fixes regardless of syntactic style.

---

## Evaluator rubric

The reward (`score`) is a single float in **`[0.0, 1.0]`**. There is a deliberate gap between
partial credit (max **0.5**) and a correct answer (**1.0**) so the model can never "talk its way"
to a passing reward without producing a query that actually returns the right rows.

### Top-level outcomes

| Outcome | Score | Where |
|---|---|---|
| Predicted SQL returns the **exact same rows** as ground truth | **1.0** | `evaluator.py` |
| No `<answer>` SQL extracted | reasoning credit (**≤ 0.5**) | partial path |
| Predicted SQL fails to execute | reasoning credit (**≤ 0.5**) | partial path |
| Predicted SQL runs but returns **wrong rows** | reasoning credit (**≤ 0.5**) | partial path |
| **Ground-truth** SQL fails to execute (dataset bug) | **0.0** | dataset guard |

Result rows are compared as a `set` of tuples, so ordering does not affect the score.

### Partial-credit breakdown (`score_reasoning_quality`, capped at 0.5)

When the answer is not exact, partial credit rewards the **type and groundedness** of the response —
not a specific bug category. Components are additive, then clamped to `[0.0, 0.5]`:

| Signal | Δ Score | What it checks |
|---|---|---|
| Has `<reasoning>` block | **+0.05** | Output-format compliance |
| Has `<answer>` block | **+0.05** | Output-format compliance |
| Reasoning ≥ 12 words | **+0.05** | Enough detail (`+0.02` if present but brief) |
| Answer is a `SELECT` query | **+0.05** | Produced an actual query |
| Diagnostic language (`wrong`, `bug`, `should`, `missing`, …) | **+0.07** | Actually diagnoses an issue |
| SQL/schema vocab (`join`, `having`, `status`, `user_id`, …) | **+0.08** | ≥ 2 signals (`+0.04` for exactly 1) |
| Mentions the **actual changed terms** between broken & correct SQL | **+0.12** | ≥ 2 terms (`+0.07` for exactly 1) |
| No irrelevant tangents | **+0.03** | On-topic bonus |
| Irrelevant tangent (`sql injection`, `postgres`, `migration`, …) | **−0.10** | Penalty for off-topic content |

> The strongest signal is **"mentions the changed terms."** The evaluator tokenizes the broken and
> correct SQL, takes the symmetric difference of the token sets, strips SQL keywords, and checks
> whether the model's reasoning names those specific tokens — i.e. it is rewarded for identifying the
> *real* diff, not for sounding plausible.

### Notes

- `score` is always clamped: `round(max(0.0, min(score, 0.5)), 3)` for partial credit.
- `passed_threshold=0.0` is a **reporting** bar (every row "passes"), not a clamp on the reward —
  RFT consumes the continuous score, not a pass/fail flag.

---

## Project layout

| Path | Purpose |
|---|---|
| `evaluator.py` | The reward function (`eval-protocol` test). Defines the scoring rubric, in-memory DB, and SQL extraction. |
| `generate_dataset.py` | Writes a small `dataset.jsonl` (5 simple examples) used by the evaluator's default input. |
| `hard_sql_rft_100_schema.jsonl` / `.csv` | 100 harder training examples (full schema in the system prompt). |
| `heldout_infer_sql_rft_30_test_schema_prompt.jsonl` | 30 held-out test prompts + ground truth for evaluation. |
| `rft_inference.ipynb` | Inference notebook: runs base vs. RFT model on held-out cases and compares execution scores. |
| `pyproject.toml` / `uv.lock` | Project metadata and pinned dependencies (managed with `uv`). |
| `.env.example` | Template for the required `FIREWORKS_API_KEY`. |

### Database schema (seeded in memory for scoring)

```
users       (id, name, email, created_at DATE, plan)
orders      (id, user_id, amount REAL, status, created_at DATE)
order_items (id, order_id, product_id, quantity INTEGER, unit_price REAL)
products    (id, name, category, price REAL, stock INTEGER)

orders.user_id         → users.id
order_items.order_id   → orders.id
order_items.product_id → products.id

status values: 'completed', 'pending'
```

---

## Setup

```bash
# 1. Install dependencies (uv recommended)
uv sync

# 2. Configure your Fireworks API key
cp .env.example .env
# then edit .env and set FIREWORKS_API_KEY=...
```

---

## Usage

### Generate the simple dataset

```bash
uv run python generate_dataset.py   # writes dataset.jsonl
```

### Run the evaluator

`evaluator.py` is an `eval-protocol` evaluation test (pytest-based). It runs rollouts against the
configured Fireworks model and scores them with the rubric above.

```bash
uv run pytest evaluator.py
```

Model and rollout settings live in the `@evaluation_test(...)` decorator
(`model`, `temperature`, `max_dataset_rows`, etc.).

### Compare base vs. RFT model

Open `rft_inference.ipynb` and run the cells. It loads the held-out test set, runs the baseline
and fine-tuned deployments, executes each generated query against the in-memory database, and
reports side-by-side execution scores. Set `FIREWORKS_BASE_MODEL` to enable the base-model comparison.

---

## Known limitations

- **Row-set comparison ignores `ORDER BY`.** Queries where ordering is the intended behavior can
  score `1.0` even if the order is wrong.
- **Broken-query parsing is string-anchored.** The evaluator extracts the broken SQL by splitting on
  the literal `"Broken query:\n"`; prompts phrased differently will degrade the changed-term signal.
