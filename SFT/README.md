# Supervised Fine-Tuning (SFT) Lab: Intent Classification

A hands-on workshop notebook that walks through fine-tuning a pre-trained large language
model (**Llama 3.2-1B-Instruct**) for **intent classification** in a doctor appointment
booking system. You'll go from a raw base model to a LoRA fine-tuned model and measure the
improvement end-to-end.

> ⚠️ **This notebook is meant to be run on [Google Colab](https://colab.research.google.com/).**
> It requires a GPU and uses Colab-specific features (the Secrets manager for the Hugging
> Face token). Running locally is possible but not supported by these instructions — see
> [Running locally](#running-locally-optional) below.

**▶️ Open the notebook in Colab:** https://colab.research.google.com/drive/1N_hQUnsIIUbUdndBcpPvlTNMS3BsefX6

## What you'll learn

1. **Explore the data** — understand the intent classes and their distribution
2. **Tokenization** — how text is converted to token IDs the model understands
3. **Baseline inference** — evaluate the base model *before* fine-tuning
4. **SFT with LoRA** — parameter-efficient fine-tuning that trains ~0.5% of the weights
5. **Evaluation** — compare base vs. fine-tuned model with accuracy, F1, and confusion matrices

## Files

| File | Description |
|------|-------------|
| `Finetuning_Workshop_SFT_Demo.ipynb` | The main workshop notebook |
| `finetuning_preference_dataset.csv` | Labeled dataset (301 examples) of user messages → intent |
| `requirements.txt` | Python dependencies (for local runs; Colab installs these in-notebook) |

The dataset has three columns: `TEST_ID`, `USER_INPUT`, and `TARGET_INTENT`.

## Running on Google Colab (recommended)

1. **Open the notebook in Colab**
   - Open the hosted notebook directly: https://colab.research.google.com/drive/1N_hQUnsIIUbUdndBcpPvlTNMS3BsefX6
     (then **File → Save a copy in Drive** to get your own editable copy).
   - Alternatively, **File → Upload notebook** → select `Finetuning_Workshop_SFT_Demo.ipynb`.
   - Upload `finetuning_preference_dataset.csv` to the Colab session (drag it into the **Files** panel, or run the upload cell).

2. **Enable a GPU**
   - **Runtime → Change runtime type → Hardware accelerator → GPU** (the free **T4** is sufficient for this lab).

3. **Add your Hugging Face token**
   - Llama 3.2 is a **gated model**. Create a free account at [huggingface.co](https://huggingface.co), accept the Llama 3.2 license on the model page, and generate an access token.
   - In Colab, click the **🔑 key icon** (Secrets) in the left sidebar and add your token (e.g. as `HF_TOKEN`).

4. **Run the cells top to bottom.**
   - The first code cell installs all dependencies via `pip`.

## Running locally (optional)

You'll need a machine with an NVIDIA GPU and CUDA. `bitsandbytes` (4-bit quantization)
is Linux/CUDA-oriented and may not work on macOS or CPU-only machines.

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter lab   # or: jupyter notebook
```

Then open `Finetuning_Workshop_SFT_Demo.ipynb`. Set your Hugging Face token as an
environment variable (`export HF_TOKEN=...`) instead of using Colab Secrets, and adjust
the token-loading cell accordingly.

## Dependencies

Core libraries (see `requirements.txt` for the full list):

| Library | Purpose |
|---------|---------|
| `transformers` | Load pre-trained models and tokenizers |
| `trl` | `SFTTrainer` for supervised fine-tuning |
| `peft` | Parameter-efficient fine-tuning (LoRA) |
| `datasets` | Dataset handling |
| `bitsandbytes` | Quantization to reduce memory usage |
| `torch` | Deep learning framework |
