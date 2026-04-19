# Gemma 4 E4B — Local GPU Setup

Setup guide for running the local Gemma 4 E4B backend (`bible_rag.py`).
This is **optional** — most users should use the Claude backend instead (see [README.md](README.md)).

## Prerequisites

| Requirement | Detail |
|-------------|--------|
| OS | Windows 10/11 (tested), Linux should also work |
| GPU | NVIDIA GPU with 11+ GB VRAM and CUDA 12.8 support (tested on RTX 5070 Ti Laptop) |
| Python | 3.11.x |
| uv | [Install uv](https://docs.astral.sh/uv/getting-started/installation/) |
| HuggingFace account | Gemma 4 is a gated model — accept the license first |

## Step 1 — Install dependencies

```bash
cd Gemma4

# Install uv if you don't have it
# Windows PowerShell:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/macOS:
# curl -LsSf https://astral.sh/uv/install.sh | sh

# Create .venv and install all packages (PyTorch cu128, transformers, bitsandbytes, etc.)
uv sync
```

> **Different CUDA version?**
> Edit `pyproject.toml` — change the `url` under `[[tool.uv.index]]` to match your CUDA version
> (e.g. `cu124` for CUDA 12.4). See https://download.pytorch.org/whl/ for available indexes.

## Step 2 — Download model weights (~16 GB)

```bash
# Activate the venv
# Windows PowerShell:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# Login to HuggingFace (you need a read token)
# Accept the license at: https://huggingface.co/google/gemma-4-E4B-it
huggingface-cli login

# (Optional) Enable fast parallel downloads
# Windows PowerShell:
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"
# Linux/macOS:
# export HF_HUB_ENABLE_HF_TRANSFER=1

# Download into the expected folder
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='google/gemma-4-E4B-it',
    local_dir='./gemma4_e4b_model',
)
"
```

This takes ~20 min on a typical connection. The scripts expect the model at `./gemma4_e4b_model/`.

## Step 3 — Verify

```bash
# Quick CUDA check
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')"

# Run the Bible RAG with local Gemma
python scripts/ask_bible.py --backend gemma
```

Expected: model loads in ~8.7 GB VRAM, generates responses with tool calls.

## Hardware Reference

| Item | Spec |
|------|------|
| GPU | NVIDIA RTX 5070 Ti Laptop (11 GB VRAM) |
| Model | google/gemma-4-E4B-it — 4.5B effective params |
| Quantization | 4-bit NF4, bfloat16 compute, double quantization |
| VRAM used | ~8.7 GB |
| Throughput | ~13-14 tok/s with SDPA attention |

## Known Issues

- **4-bit vision fix**: NF4 quantization breaks the vision tower's pixel value dtype cast. Scripts include a monkey-patch for this.
- **CJK script leakage**: The model occasionally produces Japanese kana or Thai characters when generating Traditional Chinese. Mitigated with explicit system prompt instructions.
- **OneDrive sync conflicts**: If `.venv` gets corrupted by OneDrive, pause syncing, delete `.venv`, and run `uv sync` again.

## Background: Wikipedia Tool-Calling Experiments

The Bible RAG architecture was developed from earlier experiments using Gemma 4 E4B with Wikipedia tool calling. Those scripts (`wiki_query.py`, `cot_translate.py`, `run_prompt.py`, etc.) are preserved in `scripts/local_llm_setup/` for reference but are not part of the Bible RAG project.

See `gemma_wiki_tool_setup.md` for details on those experiments.
