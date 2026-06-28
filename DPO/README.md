# Mastering Agentic AI Bootcamp: Week 5

Week 5 demo materials for an LLM DPO workflow that aligns a model to The Gen Academy's brand voice.

## Contents

- `notebooks/brand_voice_huggingface_dpo_demo.ipynb`: executed Hugging Face DPO notebook with charts and before/after comparison.
- `data/brand_voice_preferences.jsonl`: human-readable source preference dataset.
- `data/brand_voice_trl_dpo.jsonl`: TRL-compatible DPO dataset.
- `data/brand_voice_trl_dpo_train.jsonl`: train split.
- `data/brand_voice_trl_dpo_eval.jsonl`: held-out eval split.
- `pyproject.toml`: Python dependencies and project metadata (managed with [uv](https://docs.astral.sh/uv/)).
- `requirements.txt`: legacy pip dependency list (kept for non-uv users).

## Dataset

The dataset has 80 preference pairs across 10 brand-voice content categories.

DPO training uses:

- `prompt`
- `chosen`
- `rejected`

The source dataset also includes `preference_reason`. That field is for humans reading the demo; it explains why the chosen response is better. It is not passed into DPO training.

## Run

```bash
uv sync
uv run jupyter notebook notebooks/brand_voice_huggingface_dpo_demo.ipynb
```

In the notebook, select the **Python (brand-voice-dpo)** kernel.

The notebook is already executed with the local demo results. To retrain from scratch, delete `outputs/gen-academy-brand-voice-hf-dpo` after running the notebook once locally.
