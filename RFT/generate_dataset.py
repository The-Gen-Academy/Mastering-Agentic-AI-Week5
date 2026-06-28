# generate_dataset.py
import json

SYSTEM_PROMPT = """You are a SQL correction assistant. You will be given a broken or \
semantically incorrect SQL query. Reason through what is wrong, then output a corrected \
version inside <answer> tags.

Format your response as:
<reasoning>
...your step by step reasoning...
</reasoning>
<answer>
...corrected SQL only, no explanation...
</answer>"""

dataset = [
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Fix this query:\nSELECT u.name, o.amount FROM users u JOIN orders o ON u.id = u.id WHERE o.status = 'completed'"}
        ],
        "answer": "SELECT u.name, o.amount FROM users u JOIN orders o ON u.id = o.user_id WHERE o.status = 'completed'"
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Fix this query:\nSELECT user_id, SUM(amount) FROM orders WHERE status = 'completed'"}
        ],
        "answer": "SELECT user_id, SUM(amount) FROM orders WHERE status = 'completed' GROUP BY user_id"
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Fix this query:\nSELECT * FROM users WHERE created_at > DATE('now', '-30 days')"}
        ],
        "answer": "SELECT * FROM users WHERE created_at >= DATE('now', '-30 days')"
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Fix this query:\nSELECT u.* FROM users u JOIN orders o ON u.id = o.user_id WHERE o.id = NULL"}
        ],
        "answer": "SELECT u.* FROM users u LEFT JOIN orders o ON u.id = o.user_id WHERE o.id IS NULL"
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Fix this query:\nSELECT category, COUNT(*) FROM products WHERE stock > 0"}
        ],
        "answer": "SELECT category, COUNT(*) FROM products WHERE stock > 0 GROUP BY category"
    },
]

with open("dataset.jsonl", "w") as f:
    for row in dataset:
        f.write(json.dumps(row) + "\n")

print("dataset.jsonl written")