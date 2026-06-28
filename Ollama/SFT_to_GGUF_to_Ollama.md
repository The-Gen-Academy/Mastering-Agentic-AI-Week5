# From SFT Notebook to GGUF to Ollama

This guide explains how to take a model fine-tuned in an SFT notebook, convert it into a GGUF file, publish it to Hugging Face, and run it locally with Ollama.

The example use case is an appointment-intent classifier for a scheduling agent. The same workflow works for other small routing or classification models, such as support-ticket routing, tool selection, lead qualification, email triage, and document classification.

The pipeline is:

```text
SFT notebook -> LoRA adapter -> merged Hugging Face model -> GGUF -> quantized GGUF -> Ollama
```

Training teaches the behavior. Merging makes the model standalone. GGUF makes it local-runtime friendly. Quantization makes it laptop-friendly. The Ollama Modelfile keeps the local runtime aligned with the prompt format used during fine-tuning.

## How Big Was the Original Model?

The SFT notebook used:

```text
meta-llama/Llama-3.2-1B-Instruct
```

The notebook output showed:

```text
model.safetensors: 2.47G
Parameters: 1236M
```

So the original base model was approximately:

- **1.24 billion parameters**
- **2.47 GB** in the downloaded Hugging Face safetensors file

After LoRA was attached, the notebook printed:

```text
Total parameters:     1,247,086,592
Trainable parameters:    11,272,192 (0.90%)
Frozen parameters:    1,235,814,400
```

This means the fine-tuning run trained only about **11.3M parameters**, or **0.90%** of the model. The base model stayed frozen, and the LoRA adapter learned the task-specific behavior.

## Size Changes Across the Pipeline

| Artifact | Approximate size | What it means |
|---|---:|---|
| Base Hugging Face model | ~2.47 GB | Original Llama 3.2 1B Instruct model. |
| LoRA adapter | Usually tens of MB | Small task-specific fine-tuning delta. |
| Merged Hugging Face model | ~2.5 GB | Base model plus adapter merged into one standalone model. |
| F16 GGUF | ~2.5 GB | Local GGUF version before quantization. |
| Q4_K_M GGUF | ~800 MB | Smaller local model suitable for Ollama. |

The merged model has about the same parameter count as the original model. Quantization is what makes the local GGUF file much smaller.

## Replace These Values

Use your own values for these placeholders:

```bash
HF_USERNAME="<your-huggingface-username-or-org>"
BASE_MODEL_ID="meta-llama/Llama-3.2-1B-Instruct"
TASK_NAME="appointment-intent"
MERGED_REPO_NAME="${TASK_NAME}-llama32-1b-sft"
GGUF_REPO_NAME="${TASK_NAME}-llama32-1b-sft-GGUF"
LOCAL_MODEL_NAME="${TASK_NAME}-sft"
QUANTIZATION="Q4_K_M"
```

Example Hugging Face repos:

```text
<HF_USERNAME>/appointment-intent-llama32-1b-sft
<HF_USERNAME>/appointment-intent-llama32-1b-sft-GGUF
```

The first repo stores the merged Transformers model. The second repo stores the GGUF artifact that Ollama can pull.

## 1. Fine-Tune in the SFT Notebook

The SFT notebook fine-tunes the base model for one narrow behavior.

For the scheduling example, the model reads a user message and returns exactly one intent label.

Example input:

```text
Can you move my appointment from Monday to Friday?
```

Expected output:

```text
RESCHEDULE_APPOINTMENTS
```

The training examples are formatted as chat conversations:

```text
system: You are an intent classifier...
user: Can you move my appointment from Monday to Friday?
assistant: RESCHEDULE_APPOINTMENTS
```

This format matters because the Ollama Modelfile later recreates the same structure.

## 2. Save the LoRA Adapter

The notebook uses LoRA, so the training output is an adapter checkpoint. Common paths look like:

```text
./sft-intent-classifier/checkpoint-100
./sft-intent-classifier/checkpoint-200
./sft-intent-classifier/final-adapter
```

If needed, save a final adapter at the end of training:

```python
ADAPTER_DIR = "./sft-intent-classifier/final-adapter"

trainer.save_model(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
```

The adapter contains only the learned fine-tuning delta, not the full model.

## 3. Merge the Adapter into the Base Model

Ollama needs a standalone model artifact, so merge the LoRA adapter into the original base model.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_DIR = "./sft-intent-classifier/final-adapter"
MERGED_DIR = "./appointment-intent-llama32-1b-sft"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)

peft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
merged_model = peft_model.merge_and_unload()

merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)
```

After this step, `MERGED_DIR` contains a normal Hugging Face model with the fine-tuned behavior merged in.

## 4. Push the Merged Model to Hugging Face

```python
from huggingface_hub import login

login()

HF_USERNAME = "<your-huggingface-username-or-org>"
MERGED_REPO_NAME = "appointment-intent-llama32-1b-sft"
MERGED_REPO_ID = f"{HF_USERNAME}/{MERGED_REPO_NAME}"

merged_model.push_to_hub(MERGED_REPO_ID, safe_serialization=True)
tokenizer.push_to_hub(MERGED_REPO_ID)
```

This repo is useful for normal Hugging Face workflows, but it is not the final Ollama artifact yet.

## 5. Convert the Merged Model to GGUF

GGUF is the model format used by `llama.cpp` and supported by Ollama.

Clone and set up `llama.cpp`:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
python3 -m pip install -r requirements.txt
```

Download the merged model:

```bash
python3 - <<'PY'
from huggingface_hub import snapshot_download

HF_USERNAME = "<your-huggingface-username-or-org>"
MERGED_REPO_NAME = "appointment-intent-llama32-1b-sft"

snapshot_download(
    repo_id=f"{HF_USERNAME}/{MERGED_REPO_NAME}",
    local_dir=f"./{MERGED_REPO_NAME}",
    local_dir_use_symlinks=False,
)
PY
```

Convert it to F16 GGUF:

```bash
python3 convert_hf_to_gguf.py \
  ./appointment-intent-llama32-1b-sft \
  --outfile ./appointment-intent-llama32-1b-sft-f16.gguf \
  --outtype f16
```

The F16 GGUF is still large. For this 1B model, expect roughly 2.5 GB.

## 6. Quantize the GGUF

Quantization compresses the model for local inference.

Build the quantization tool:

```bash
cmake -B build
cmake --build build --config Release -j
```

Quantize to `Q4_K_M`:

```bash
./build/bin/llama-quantize \
  ./appointment-intent-llama32-1b-sft-f16.gguf \
  ./appointment-intent-llama32-1b-sft-Q4_K_M.gguf \
  Q4_K_M
```

`Q4_K_M` is a good default for small routing models because it is compact, fast, and usually preserves enough quality for classification.

## 7. Create an Ollama Modelfile

The GGUF file stores the model weights. The Modelfile tells Ollama how to prompt and run the model.

Create:

```text
appointment-intent-sft.Modelfile
```

Example:

```text
FROM hf.co/<HF_USERNAME>/<GGUF_REPO_NAME>:Q4_K_M

PARAMETER temperature 0
PARAMETER num_predict 16
PARAMETER stop <|start_header_id|>
PARAMETER stop <|end_header_id|>
PARAMETER stop <|eot_id|>

TEMPLATE """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

SYSTEM """You are an intent classifier for a doctor appointment booking system.
Classify the user message into exactly one of these following intents:
LIST_APPOINTMENTS, CANCEL_APPOINTMENTS, RESCHEDULE_APPOINTMENTS, UNSUPPORTED_REQUEST, BLOCK_SLOTS, UNBLOCK_SLOTS, UPDATE_SPECIAL_SLOTS, CONVERSATIONAL_GREETING, CLOSING_CONVERSATION
Reply with ONLY the EXACT intent label. Nothing else."""
```

For another classifier, replace the label list and system prompt with the labels used during training.

## 8. Push the GGUF and Modelfile to Hugging Face

Create a separate Hugging Face repo for the GGUF model:

```python
from huggingface_hub import HfApi

HF_USERNAME = "<your-huggingface-username-or-org>"
GGUF_REPO_NAME = "appointment-intent-llama32-1b-sft-GGUF"
GGUF_REPO_ID = f"{HF_USERNAME}/{GGUF_REPO_NAME}"

api = HfApi()

api.create_repo(
    repo_id=GGUF_REPO_ID,
    repo_type="model",
    exist_ok=True,
)
```

Upload the quantized GGUF:

```python
api.upload_file(
    path_or_fileobj="./appointment-intent-llama32-1b-sft-Q4_K_M.gguf",
    path_in_repo="appointment-intent-llama32-1b-sft-Q4_K_M.gguf",
    repo_id=GGUF_REPO_ID,
    repo_type="model",
)
```

Upload the Modelfile as `Modelfile`:

```python
api.upload_file(
    path_or_fileobj="./appointment-intent-sft.Modelfile",
    path_in_repo="Modelfile",
    repo_id=GGUF_REPO_ID,
    repo_type="model",
)
```

Ollama can pull directly from Hugging Face using the `hf.co/...` model reference.

## 9. Pull and Run with Ollama

Install Ollama from:

```text
https://ollama.com/download
```

Pull the GGUF model:

```bash
ollama pull hf.co/<HF_USERNAME>/<GGUF_REPO_NAME>:Q4_K_M
```

Create a local model alias:

```bash
ollama create appointment-intent-sft -f appointment-intent-sft.Modelfile
```

Run inference:

```bash
ollama run appointment-intent-sft "Can you move my appointment from Monday to Friday?"
```

Expected output:

```text
RESCHEDULE_APPOINTMENTS
```

## 10. Call Ollama from Python

A local notebook or app can call Ollama over HTTP.

```python
import requests

OLLAMA_URL = "http://localhost:11434"
LOCAL_MODEL_NAME = "appointment-intent-sft"

def classify_intent(user_message: str) -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LOCAL_MODEL_NAME,
            "prompt": user_message,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 16,
            },
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["response"].strip()

print(classify_intent("Please cancel my appointment tomorrow."))
```

Expected output:

```text
CANCEL_APPOINTMENTS
```

In an agent, this classifier is the routing step:

```text
user message -> local Ollama classifier -> intent label -> workflow/tool selection
```

## 11. What to Customize

| Field | What to change |
|---|---|
| `<HF_USERNAME>` | Hugging Face username or organization. |
| `<GGUF_REPO_NAME>` | Name of the GGUF model repo. |
| `LOCAL_MODEL_NAME` | Friendly local Ollama model name. |
| `SYSTEM` prompt | Exact labels and instructions for the classifier. |
| `num_predict` | Keep small for classifiers; increase only for longer outputs. |
| `temperature` | Usually `0` for routing and classification. |

For a support-ticket router, the labels might be:

```text
Active Directory
Computer-Services
EOL
Fileservice
O365
Software
Support general
```

The workflow stays the same. Only the dataset, labels, repo names, and system prompt change.

## 12. End-to-End Artifact Map

| Stage | Example artifact | Purpose |
|---|---|---|
| SFT notebook | `./sft-intent-classifier` | Training output folder. |
| LoRA adapter | `./sft-intent-classifier/final-adapter` | Small learned fine-tuning delta. |
| Merged model | `./appointment-intent-llama32-1b-sft` | Standalone Hugging Face model. |
| Merged HF repo | `<HF_USERNAME>/appointment-intent-llama32-1b-sft` | Shareable merged model. |
| F16 GGUF | `appointment-intent-llama32-1b-sft-f16.gguf` | Full precision local format. |
| Quantized GGUF | `appointment-intent-llama32-1b-sft-Q4_K_M.gguf` | Smaller local model file. |
| GGUF HF repo | `<HF_USERNAME>/appointment-intent-llama32-1b-sft-GGUF` | Ollama-pullable model source. |
| Modelfile | `appointment-intent-sft.Modelfile` | Runtime prompt and generation settings. |
| Ollama alias | `appointment-intent-sft` | Local model name used by apps. |

