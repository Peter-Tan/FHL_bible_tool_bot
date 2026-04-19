# Bible RAG — Gemma 4 31B on NVIDIA RTX 6000 Pro (96 GB VRAM)

Setup guide for running `bible_rag.py` with Gemma 4 31B on a server GPU.
The current laptop setup uses Gemma 4 E4B (4B) with 4-bit NF4 on an 11 GB RTX 5070 Ti.
With 96 GB VRAM, we can run the full 31B model in BF16 (no quantization needed).

---

## 1. Hardware & VRAM Budget

| Item | Spec |
|------|------|
| GPU | NVIDIA RTX 6000 Pro (96 GB VRAM) |
| Model | `google/gemma-4-31B-it` (30.7B params + 550M vision encoder) |
| Architecture | Dense (not MoE), 60 layers, 262K vocab |
| Precision | BF16 (no quantization) |
| Context window | **256K tokens** |
| Estimated VRAM | ~62 GB model weights + ~15-25 GB KV cache = ~77-87 GB |
| Headroom | ~9-19 GB depending on context length used |

---

## 2. Server Environment Setup

```bash
# 1. System dependencies (Ubuntu 22.04 / 24.04)
sudo apt update && sudo apt install -y python3.11 python3.11-venv git

# 2. Verify NVIDIA driver and CUDA
nvidia-smi                    # should show RTX 6000 Pro, driver >= 550
nvcc --version                # CUDA 12.x required

# 3. Clone the repo
git clone https://github.com/Peter-Tan/Gemma4-E4b-zh-tw.git
cd Gemma4

# 4. Create venv
python3.11 -m venv .venv
source .venv/bin/activate

# 5. Install PyTorch (match your CUDA version — example for CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 6. Install dependencies
pip install -U transformers accelerate requests

# 7. (Optional) Flash Attention 2 for faster long-context inference
pip install flash-attn --no-build-isolation

# 8. Login to HuggingFace (Gemma requires license acceptance)
#    Accept license at: https://huggingface.co/google/gemma-4-31B-it
pip install huggingface-hub
huggingface-cli login

# 9. Download model weights (~62 GB for BF16)
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='google/gemma-4-31B-it',
    local_dir='./gemma4_31b_model',
)
"
```

---

## 3. Code Changes Required in `bible_rag.py`

### 3a. MODEL_ID — point to the 31B model

```python
# BEFORE (E4B):
MODEL_ID = "./gemma4_e4b_model"

# AFTER (31B):
MODEL_ID = "./gemma4_31b_model"
```

### 3b. Model class — switch from ImageTextToText to CausalLM

The 31B model uses `AutoModelForCausalLM` for text-only workflows
(our Bible RAG is text-only — no images). If you also need vision,
use `AutoModelForMultimodalLM` instead.

```python
# BEFORE:
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# AFTER:
from transformers import AutoProcessor, AutoModelForCausalLM
```

### 3c. Remove 4-bit quantization — run in BF16

96 GB VRAM can hold the full 31B model without quantization.

```python
# BEFORE (4-bit NF4 for 11 GB GPU):
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_cfg,
    device_map="cuda:0",
    attn_implementation="sdpa",
)

# AFTER (BF16, no quantization):
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="flash_attention_2",  # or "sdpa" if flash-attn not installed
)
```

### 3d. Generation config — increase output length

The 31B model has 256K context and stronger reasoning.

```python
# BEFORE:
GEN_CONFIG = dict(
    max_new_tokens     = 1024,
    do_sample          = True,
    temperature        = 0.1,
    top_p              = 0.95,
    repetition_penalty = 1.1,
)

# AFTER:
GEN_CONFIG = dict(
    max_new_tokens     = 4096,
    do_sample          = True,
    temperature        = 0.1,
    top_p              = 0.95,
    repetition_penalty = 1.1,
)
```

### 3e. Increase tool round limit (optional)

256K context supports much deeper multi-round tool chaining.

```python
# BEFORE:
MAX_TOOL_ROUNDS = 6

# AFTER:
MAX_TOOL_ROUNDS = 10
```

---

## 4. Bug Fix: Multi-Chapter Queries (E4B bug, mostly resolved by 31B)

### The problem (on E4B)

When a user asks "根據出埃及記7-12章，十災的發生順序為何？", the model calls
`get_bible_chapter` 6 times. Exodus 7-12 = ~180 verses. On the E4B with its
8K token context window, this immediately blows up — the tool results alone
exceed the entire context.

### Why this is mostly a non-issue on the 31B

The 31B has a **256K token** context window. Worst-case estimate for a
10-chapter query:

| Component | Tokens (est.) |
|-----------|---------------|
| System prompt | ~500 |
| 13 tool schemas | ~4,000 |
| 10 full chapters (~300 verses) | ~25,000-45,000 |
| Commentary + Strong's results | ~8,000 |
| `max_new_tokens` reserved | 4,096 |
| **Total** | **~42,000-62,000** |

That's **~20-25% of the 256K window**. No truncation needed.

**VRAM is the real constraint**, not context length:
- At 50K tokens → ~12 GB KV cache + 62 GB weights = **~74 GB** (fits)
- At 100K tokens → ~24 GB KV cache + 62 GB = **~86 GB** (fits, tight)
- At 256K tokens → ~61 GB KV cache + 62 GB = **~123 GB** (OOM)

For realistic Bible RAG queries, you'll stay around 50-80K tokens. Safe.

### Fix 1: Parallel tool execution (recommended)

The only real fix needed for the 31B: when the model requests multiple tool
calls in one round (e.g., 6 chapters), the current code runs them
sequentially. FHL API calls take ~0.5-1s each, so 6 chapters = ~3-6s of
serial waiting. Run them in parallel:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

# Inside bible_query(), replace the sequential tool execution loop with:

        # Execute every call the model requested this round — in parallel.
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(execute_tool, tc): tc for tc in tool_calls
            }
            for future in as_completed(futures):
                tc = futures[future]
                result_json = future.result()

                if verbose:
                    args_str = json.dumps(tc.get("arguments", {}), ensure_ascii=False)
                    print(f"[Round {round_num}] Tool call: {tc['name']}({args_str})")
                    preview = (result_json[:RESULT_PREVIEW] + "..."
                               if len(result_json) > RESULT_PREVIEW else result_json)
                    print(f"  → {preview}\n")

                messages.append({
                    "role": "tool",
                    "name": tc["name"],
                    "content": [{"type": "text", "text": result_json}],
                })
```

### Fix 2: VRAM safety guard (optional, defensive)

Not needed for typical Bible RAG queries, but prevents OOM if context
ever grows past ~100K tokens (e.g., extremely long multi-turn sessions).

```python
MAX_CONTEXT_TOKENS = 200000  # leave headroom for generation + KV cache

# Inside bible_query(), after apply_chat_template and before generate():
        input_len = inputs["input_ids"].shape[1]
        if input_len > MAX_CONTEXT_TOKENS:
            if verbose:
                print(f"[Warning] Context {input_len} tokens exceeds budget "
                      f"{MAX_CONTEXT_TOKENS}. Trimming old tool results.")
            # Drop oldest tool-result messages until under budget
            while input_len > MAX_CONTEXT_TOKENS:
                for i, m in enumerate(messages):
                    if m.get("role") == "tool":
                        messages.pop(i)
                        if i > 0 and messages[i-1].get("role") == "assistant":
                            messages.pop(i-1)
                        break
                else:
                    break  # no more tool messages to drop
                inputs = processor.apply_chat_template(
                    messages, tools=tools, add_generation_prompt=True,
                    tokenize=True, return_dict=True, return_tensors="pt",
                ).to(model.device)
                input_len = inputs["input_ids"].shape[1]
```

### Impact summary

| | Before | After |
|---|---|---|
| 6 API calls | ~3-6s serial | ~0.5-1s parallel |
| 10-chapter query on E4B (8K) | Blows up | Still blows up (E4B limitation) |
| 10-chapter query on 31B (256K) | Works but slow | Works and fast |
| VRAM safety | None | Optional guard at 200K tokens |

---

## 5. Code Changes Required in `fhl_tools.py`

No changes needed. The 13 FHL tools are model-agnostic.

---

## 5. Code Changes Required in `ask_bible.py`

No changes needed. It imports from `bible_rag.py`, so the model swap
propagates automatically.

---

## 6. Summary of All Changes

| File | What to change | Why |
|------|---------------|-----|
| `bible_rag.py` line 48 | `MODEL_ID` → `"./gemma4_31b_model"` | Point to 31B weights |
| `bible_rag.py` line 38 | `AutoModelForImageTextToText` → `AutoModelForCausalLM`, remove `BitsAndBytesConfig` | 31B uses CausalLM for text-only; no quantization needed |
| `bible_rag.py` lines 96-108 | Remove `bnb_cfg`, use `torch_dtype=torch.bfloat16`, use `flash_attention_2` | 96 GB fits BF16 natively; FA2 is faster for long context |
| `bible_rag.py` line 84 | `max_new_tokens` → `4096` | 31B can produce longer answers |
| `bible_rag.py` line 49 | `MAX_TOOL_ROUNDS` → `10` (optional) | Deeper tool chains with 256K context |
| `bible_rag.py` `bible_query()` | Parallel tool execution with `ThreadPoolExecutor` | 6 API calls in ~1s instead of ~6s |
| `bible_rag.py` `bible_query()` | VRAM safety guard at 200K tokens (optional) | Prevents OOM in extreme multi-turn sessions |
| `fhl_tools.py` | None | Model-agnostic |
| `ask_bible.py` | None | Inherits from bible_rag |

---

## 7. Verify After Setup

```bash
# Quick sanity check — model loads and GPU is used
python -c "
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    './gemma4_31b_model',
    torch_dtype=torch.bfloat16,
    device_map='cuda:0',
    attn_implementation='sdpa',
)
print(f'VRAM used: {torch.cuda.memory_allocated() / 1024**3:.1f} GB')
print(f'VRAM total: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"

# Run the Bible RAG
python scripts/bible_rag.py

# Test question (exercises multi-tool chain):
# 約翰福音3:16 的原文意義是什麼?
```

---

## 8. Expected Performance vs Current Setup

| | RTX 5070 Ti (current) | RTX 6000 Pro (server) |
|---|---|---|
| Model | Gemma 4 E4B (4B) | Gemma 4 31B (30.7B) |
| Precision | 4-bit NF4 | BF16 (full) |
| VRAM used | ~8.7 GB / 11 GB | ~62-80 GB / 96 GB |
| Context window | ~8K tokens | **256K tokens** |
| Model class | `AutoModelForImageTextToText` | `AutoModelForCausalLM` |
| Output quality | Acceptable, some CJK leakage | Significantly better reasoning and Chinese |
| Tool chaining | Limited by context | Deep multi-round chains feasible |
| Throughput | ~13-14 tok/s | ~10-20 tok/s (estimated, larger model but full precision) |

---

## 9. Known Considerations

1. **Model class difference**: The E4B uses `AutoModelForImageTextToText` but
   the 31B uses `AutoModelForCausalLM` for text-only. If you later add image
   input to the Bible RAG, switch to `AutoModelForMultimodalLM`.

2. **Flash Attention 2**: Strongly recommended for the 31B with 256K context.
   Without it, long multi-round tool chains will be noticeably slower.
   Requires: `pip install flash-attn --no-build-isolation`.

3. **VRAM is tight at full context**: 31B in BF16 uses ~62 GB for weights alone.
   At very long contexts (100K+ tokens), KV cache could push past 96 GB.
   For the Bible RAG pipeline this is unlikely (tool results are relatively
   small JSON), but monitor with `nvidia-smi -l 1` during initial testing.
   If VRAM runs out, options:
   - Use 8-bit quantization (~31 GB weights, plenty of headroom)
   - Cap effective context with a token budget guard in the pipeline

4. **CJK script leakage**: The 31B model should have significantly less
   Japanese/Thai character leakage than the 4B, but keep the system prompt
   instruction `嚴禁使用日文（片假名、平假名）、泰文、阿拉伯文` as a safeguard.

5. **bitsandbytes not needed**: After removing NF4 quantization, `bitsandbytes`
   is not required on the server. Skip installing it.

6. **History window**: `ask_bible.py` keeps the last 16 messages (8 turns).
   With 256K context, you could increase this to 32+ if multi-turn
   conversations are important.

7. **No audio support**: The 31B does not support audio (only E2B/E4B do).
   This is irrelevant for the Bible RAG pipeline.

8. **Sliding window attention**: The 31B uses a 1024-token sliding window.
   This is handled internally by the model — no code changes needed, but
   be aware that very distant context may receive less attention than recent
   context, even within the 256K window.
