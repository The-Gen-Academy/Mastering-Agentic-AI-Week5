# Mastering Agentic AI - Week 5 Demo

Local inference demo for running a fine-tuned appointment intent router with Ollama.

## Contents

- `Appointment_Intent_Router_with_Ollama.ipynb` - Local Ollama API notebook for pulling, configuring, and running the fine-tuned appointment intent router.

## Setup (uv)

This project is managed with [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies (creates .venv automatically)
uv sync

# Launch the notebook
uv run jupyter lab Appointment_Intent_Router_with_Ollama.ipynb
```

In Jupyter, select the project's virtual environment kernel (`.venv`). The
in-notebook `pip install` cell is no longer needed — dependencies are pinned in
`pyproject.toml` / `uv.lock`.

> Requires a running [Ollama](https://ollama.com) instance (defaults to
> `http://localhost:11434`; override with the `OLLAMA_BASE_URL` env var).

## Appointment Intent Router

The Ollama notebook pulls the GGUF model:

```bash
hf.co/vidhyakshayakannan/appointment-intent-llama32-1b-sft-GGUF:Q4_K_M
```

It then creates a local Ollama model wrapper named:

```bash
appointment-intent-sft
```

The wrapper includes the routing system prompt, deterministic generation settings, and the Llama 3.2 chat template so the model returns appointment-routing intent labels reliably.
