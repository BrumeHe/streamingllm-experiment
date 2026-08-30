"""PPL comparison of KV-cache strategies on a long text.

Three settings, all evaluated chunk-by-chunk over the same tokens:
  full        - unbounded KV cache
  window      - keep only the last `window` KV positions (pure sliding window)
  sink_window - keep the first `sink` + last `window` positions (StreamingLLM)

Writes results/ppl_compare.csv, results/ppl_compare.md and
results/ppl_per_chunk.csv.
"""
import argparse
import csv
import math
import os

import torch

from common import RESULTS, load_model, load_text, chunked_nll

MODES = [
    ("full", "full KV (baseline)"),
    ("window", "sliding window only"),
    ("sink_window", "sink (first 4) + sliding window"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=12288)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--sink", type=int, default=4)
    args = ap.parse_args()

    tok, model = load_model(attn_implementation="eager")
    text = load_text()
    ids = tok(text, return_tensors="pt").input_ids[:, : args.max_tokens]
    n = ids.shape[1]
    print(f"[ppl] evaluating {n} tokens, chunk={args.chunk}, "
          f"window={args.window}, sink={args.sink}")

    rows = []
    per_chunk_rows = []
    for mode, desc in MODES:
        nll, cnt, per_chunk = chunked_nll(
            model, ids, chunk=args.chunk, mode=mode,
            window=args.window, sink=args.sink,
        )
        ppl = math.exp(nll / cnt)
        print(f"[ppl] {mode:12s} nll={nll:.1f} tokens={cnt} ppl={ppl:.4f}")
        rows.append((mode, desc, cnt, ppl))
        for start, c_nll, c_cnt in per_chunk:
            per_chunk_rows.append((mode, start, math.exp(c_nll / c_cnt)))
        torch.cuda.empty_cache()

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "ppl_compare.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "description", "scored_tokens", "ppl"])
        for mode, desc, cnt, ppl in rows:
            w.writerow([mode, desc, cnt, f"{ppl:.4f}"])

    src = ""
    src_path = os.path.join(os.path.dirname(RESULTS), "data", "SOURCE.txt")
    if os.path.exists(src_path):
        src = open(src_path).read().strip()
    with open(os.path.join(RESULTS, "ppl_compare.md"), "w") as f:
        f.write("# PPL comparison: KV cache strategies\n\n")
        f.write(f"- model: Qwen2.5-1.5B-Instruct (bf16, eager attention)\n")
        f.write(f"- text source: {src}\n")
        f.write(f"- tokens scored: {rows[0][2]} (first {n} tokens of the text)\n")
        f.write(f"- chunk: {args.chunk}, window: {args.window}, sink tokens: {args.sink}\n\n")
        f.write("| setting | description | PPL |\n")
        f.write("|---|---|---|\n")
        for mode, desc, cnt, ppl in rows:
            f.write(f"| {mode} | {desc} | {ppl:.4f} |\n")

    with open(os.path.join(RESULTS, "ppl_per_chunk.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "chunk_start", "chunk_ppl"])
        w.writerows(per_chunk_rows)

    print(f"[ppl] tables written to {RESULTS}")


if __name__ == "__main__":
    main()
