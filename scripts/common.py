"""Shared utilities for experiment A1.

Model loading, KV-cache truncation (StreamingLLM style) and chunked NLL
computation used by both the heatmap and the PPL-comparison scripts.
"""
import os

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # experiment/
MODEL_DIR = os.path.join(ROOT, "models", "Qwen2.5-1.5B-Instruct")
DATA_FILE = os.path.join(ROOT, "data", "long_text.txt")
RESULTS = os.path.join(ROOT, "results")


def load_model(attn_implementation="eager"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    # bf16: Qwen2.5 activations overflow fp16 (observed NaN logits), bf16 is
    # the model's native dtype and is numerically stable here.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        device_map="cuda",
    )
    model.eval()
    return tok, model


def load_text(path=DATA_FILE):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def truncate_past(past_legacy, keep_start=0, window=None):
    """Keep the first `keep_start` and the last `window` KV positions.

    Operates on the legacy tuple-of-tuples cache format; callers convert
    to/from DynamicCache. Returns a new legacy cache.
    """
    if window is None:
        return past_legacy
    new_past = []
    for k, v in past_legacy:
        n = k.shape[2]
        idx = list(range(min(keep_start, n)))
        start = max(n - window, len(idx))
        idx += list(range(start, n))
        idx_t = torch.tensor(idx, device=k.device)
        new_past.append(
            (
                k.index_select(2, idx_t).contiguous(),
                v.index_select(2, idx_t).contiguous(),
            )
        )
    return tuple(new_past)


@torch.no_grad()
def chunked_nll(model, ids, chunk=1024, mode="full", window=1024, sink=4):
    """Token-level NLL sum over ids[0, :], evaluated chunk by chunk with a KV cache.

    Each chunk is scored with teacher forcing against the cached prefix.
    The last logit of a chunk (which would predict the next chunk's first
    token) is dropped; this is identical across modes, so the comparison is
    fair.

    mode: "full" (unbounded KV) | "window" (last `window`) |
          "sink_window" (first `sink` + last `window`).
    Returns (nll_sum, token_count, per_chunk) where per_chunk is a list of
    (start, nll, count).
    """
    from transformers.cache_utils import DynamicCache

    device = next(model.parameters()).device
    ids = ids.to(device)
    L = ids.shape[1]
    past = None
    nll_sum, tok_count = 0.0, 0
    per_chunk = []
    for start in range(0, L - 1, chunk):
        chunk_ids = ids[:, start : start + chunk]
        # Explicit true positions: RoPE attention is relative, so surviving
        # window keys keep correct distances to the new tokens and the sink
        # keys simply stay far away. (Rebasing positions to the post-trim
        # cache length collapses this geometry and badly distorts PPL.)
        position_ids = torch.arange(start, start + chunk_ids.shape[1],
                                    device=device).unsqueeze(0)
        out = model(input_ids=chunk_ids, past_key_values=past,
                    position_ids=position_ids, use_cache=True)
        legacy = out.past_key_values.to_legacy_cache()
        del out.past_key_values

        logits = out.logits[:, :-1, :].float()
        targets = chunk_ids[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="sum",
        )
        nll_sum += loss.item()
        tok_count += targets.numel()
        per_chunk.append((start, loss.item(), targets.numel()))
        del out, logits

        if mode == "window":
            legacy = truncate_past(legacy, keep_start=0, window=window)
        elif mode == "sink_window":
            legacy = truncate_past(legacy, keep_start=sink, window=window)
        past = DynamicCache.from_legacy_cache(legacy)
    return nll_sum, tok_count, per_chunk
