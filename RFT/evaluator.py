import os
import re
import sqlite3
import logging
from typing import Optional, Tuple

from eval_protocol.models import EvaluateResult, EvaluationRow
from eval_protocol.pytest import SingleTurnRolloutProcessor, evaluation_test

logger = logging.getLogger(__name__)

JSONL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../dataset.jsonl")
)


def extract_reasoning(text: str) -> str:
    match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def extract_sql(text: str) -> Optional[str]:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


SQL_STOPWORDS = {
    "select", "from", "where", "join", "left", "right", "inner", "outer",
    "on", "and", "or", "as", "group", "by", "order", "having", "limit",
    "count", "sum", "avg", "min", "max", "distinct", "case", "when",
    "then", "else", "end", "is", "not", "null", "in", "exists",
    "date", "now", "integer", "real", "text"
}


def extract_changed_terms(broken_sql: str, correct_sql: str) -> set[str]:
    """Return meaningful tokens that differ between broken and correct SQL."""
    token_re = re.compile(r"[A-Za-z_][A-Za-z0-9_\.]*|'[^']*'|>=|<=|!=|<>|=|>|<|\*")
    broken_tokens = {t.lower().strip("'") for t in token_re.findall(broken_sql)}
    correct_tokens = {t.lower().strip("'") for t in token_re.findall(correct_sql)}
    changed = (broken_tokens ^ correct_tokens) - SQL_STOPWORDS
    return {t for t in changed if len(t) > 1 or t in {">", "<", "=", "*"}}


def score_reasoning_quality(
    response_text: str,
    reasoning: str,
    broken_sql: str,
    correct_sql: str,
    predicted_sql: Optional[str],
) -> Tuple[float, str]:
    """
    Award partial credit for a useful response when the SQL answer is not correct.
    This does not identify a fixed bug category. It rewards the response type:
    tagged output, grounded reasoning, SQL-specific diagnosis, and no irrelevant tangents.
    Max score is 0.5 so an incorrect answer can never look correct.
    """
    score = 0.0
    reasons = []
    response_lower = response_text.lower()
    reasoning_lower = reasoning.lower()

    has_reasoning_tag = bool(re.search(r"<reasoning>.*?</reasoning>", response_text, re.DOTALL | re.IGNORECASE))
    has_answer_tag = bool(re.search(r"<answer>.*?</answer>", response_text, re.DOTALL | re.IGNORECASE))

    if has_reasoning_tag:
        score += 0.05
        reasons.append("has <reasoning> block")
    else:
        reasons.append("missing <reasoning> block")

    if has_answer_tag:
        score += 0.05
        reasons.append("has <answer> block")
    else:
        reasons.append("missing <answer> block")

    word_count = len(reasoning.split())
    if word_count >= 12:
        score += 0.05
        reasons.append("reasoning has enough detail")
    elif word_count > 0:
        score += 0.02
        reasons.append("reasoning is present but brief")
    else:
        reasons.append("reasoning is empty")

    if predicted_sql and predicted_sql.strip().lower().startswith("select"):
        score += 0.05
        reasons.append("answer contains a SELECT query")
    elif predicted_sql:
        reasons.append("answer is not a SELECT query")
    else:
        reasons.append("no SQL extracted")

    diagnostic_signals = [
        "wrong", "incorrect", "bug", "issue", "problem", "error",
        "instead of", "should", "needs to", "need to", "missing",
        "exclude", "include", "filters", "joins", "joined"
    ]
    if any(s in reasoning_lower for s in diagnostic_signals):
        score += 0.07
        reasons.append("diagnoses a query issue")
    else:
        reasons.append("does not clearly diagnose the issue")

    sql_signals = [
        "join", "where", "having", "group by", "status", "plan",
        "category", "price", "stock", "amount", "created_at",
        "user_id", "order_id", "product_id", "quantity", "unit_price",
        "count", "sum", "avg", "distinct", "left join"
    ]
    sql_signal_count = sum(1 for s in sql_signals if s in reasoning_lower)
    if sql_signal_count >= 2:
        score += 0.08
        reasons.append("uses SQL/schema-specific reasoning")
    elif sql_signal_count == 1:
        score += 0.04
        reasons.append("uses one SQL/schema-specific signal")
    else:
        reasons.append("not grounded in SQL/schema terms")

    changed_terms = extract_changed_terms(broken_sql, correct_sql)
    mentioned_changes = sorted(t for t in changed_terms if t in reasoning_lower)
    if len(mentioned_changes) >= 2:
        score += 0.12
        reasons.append(f"mentions changed terms: {mentioned_changes[:5]}")
    elif len(mentioned_changes) == 1:
        score += 0.07
        reasons.append(f"mentions one changed term: {mentioned_changes[0]}")
    else:
        reasons.append("does not mention the concrete SQL change")

    irrelevant_signals = [
        "sql injection", "security", "alter table", "schema change",
        "migration", "now()", "postgres", "mysql", "parameterized"
    ]
    irrelevant = [s for s in irrelevant_signals if s in response_lower]
    if irrelevant:
        score -= 0.10
        reasons.append(f"irrelevant tangent: {irrelevant[:3]}")
    else:
        score += 0.03
        reasons.append("no obvious irrelevant tangent")

    return round(max(0.0, min(score, 0.5)), 3), "; ".join(reasons)


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY, name TEXT NOT NULL,
            email TEXT NOT NULL, created_at DATE NOT NULL, plan TEXT NOT NULL
        );
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL,
            amount REAL NOT NULL, status TEXT NOT NULL, created_at DATE NOT NULL
        );
        CREATE TABLE order_items (
            id INTEGER PRIMARY KEY, order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL, quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL
        );
        CREATE TABLE products (
            id INTEGER PRIMARY KEY, name TEXT NOT NULL,
            category TEXT NOT NULL, price REAL NOT NULL, stock INTEGER NOT NULL
        );

        INSERT INTO users VALUES
            (1,'Alice','alice@ex.com', DATE('now','-5 days'),  'pro'),
            (2,'Bob',  'bob@ex.com',   DATE('now','-15 days'), 'free'),
            (3,'Carol','carol@ex.com', DATE('now','-30 days'), 'pro'),
            (4,'Dave', 'dave@ex.com',  DATE('now','-45 days'), 'free'),
            (5,'Eve',  'eve@ex.com',   DATE('now','-60 days'), 'pro');

        INSERT INTO orders VALUES
            (1,1,120.00,'completed', DATE('now','-4 days')),
            (2,1,85.50, 'completed', DATE('now','-3 days')),
            (3,2,200.00,'pending',   DATE('now','-14 days')),
            (4,3,340.00,'completed', DATE('now','-29 days')),
            (5,4,95.00, 'completed', DATE('now','-44 days')),
            (6,3,50.00, 'completed', DATE('now','-10 days')),
            (7,1,310.00,'completed', DATE('now','-2 days'));

        INSERT INTO products VALUES
            (1,'Laptop',   'electronics',999.00,10),
            (2,'Phone',    'electronics',699.00,25),
            (3,'Desk',     'furniture',  350.00,5),
            (4,'Chair',    'furniture',  199.00,0),
            (5,'Notebook', 'stationery', 4.99,  100),
            (6,'Pen',      'stationery', 1.99,  0);

        INSERT INTO order_items VALUES
            (1,1,1,1,999.00),
            (2,1,5,2,4.99),
            (3,2,2,1,699.00),
            (4,3,3,1,350.00),
            (5,4,1,1,999.00),
            (6,5,4,2,199.00),
            (7,6,5,5,4.99),
            (8,7,2,1,699.00),
            (9,7,1,1,999.00);
    """)
    return conn


@evaluation_test(
    input_dataset=[JSONL_PATH],
    completion_params=[{
        "model": "fireworks_ai/accounts/fireworks/models/llama-v3p1-8b-instruct",
        "temperature": 0.8
    }],
    max_dataset_rows=5,
    passed_threshold=0.0,
    rollout_processor=SingleTurnRolloutProcessor(),
    mode="pointwise",
)
def test_sql_evaluator(row: EvaluationRow, **kwargs) -> EvaluationRow:
    logger.info(f"Starting SQL rollout: {row.execution_metadata.rollout_id}")

    assistant = row.messages[-1]
    response_text = (
        assistant.get("content", "")
        if isinstance(assistant, dict)
        else (assistant.content or "")
    )

    correct_sql   = str(row.ground_truth)
    reasoning     = extract_reasoning(response_text)
    predicted_sql = extract_sql(response_text)
    user_message = row.messages[-2] if len(row.messages) >= 2 else {}
    user_content = (
        user_message.get("content", "")
        if isinstance(user_message, dict)
        else (user_message.content or "")
    )
    broken_sql = user_content.split("Broken query:\n", 1)[-1]

    # ── no answer tags: still allow reasoning/format partial ──
    if not predicted_sql:
        rsn_score, rsn_reason = score_reasoning_quality(
            response_text, reasoning, broken_sql, correct_sql, predicted_sql
        )
        row.evaluation_result = EvaluateResult(
            score=rsn_score,
            is_score_valid=True,
            reason=f"No <answer> tags found in model response | Reasoning partial credit: {rsn_reason}"
        )
        return row

    # ── execution score ──────────────────────────────────────
    conn = get_db()

    try:
        predicted_rows = set(map(tuple, conn.execute(predicted_sql).fetchall()))
    except Exception as e:
        conn.close()
        # still check reasoning even if SQL fails to execute
        rsn_score, rsn_reason = score_reasoning_quality(response_text, reasoning, broken_sql, correct_sql, predicted_sql)
        row.evaluation_result = EvaluateResult(
            score=rsn_score,
            is_score_valid=True,
            reason=f"SQL failed to execute: {str(e)} | Reasoning: {rsn_reason}"
        )
        return row

    try:
        correct_rows = set(map(tuple, conn.execute(correct_sql).fetchall()))
    except Exception as e:
        conn.close()
        row.evaluation_result = EvaluateResult(
            score=0.0,
            is_score_valid=True,
            reason=f"Ground truth SQL failed to execute: {str(e)}"
        )
        return row

    conn.close()

    # ── exact match — full score, no reasoning penalty ───────
    if predicted_rows == correct_rows:
        row.evaluation_result = EvaluateResult(
            score=1.0,
            is_score_valid=True,
            reason="Exact row match"
        )
        return row

    # ── wrong answer — reward reasoning quality only ─────────
    rsn_score, rsn_reason = score_reasoning_quality(response_text, reasoning, broken_sql, correct_sql, predicted_sql)

    row.evaluation_result = EvaluateResult(
        score=rsn_score,
        is_score_valid=True,
        reason=(
            f"Wrong SQL (got {len(predicted_rows)} rows, "
            f"expected {len(correct_rows)}) | "
            f"Reasoning partial credit: {rsn_reason}"
        )
    )
    return row