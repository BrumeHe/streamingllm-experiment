"""Attention sink heatmaps for Qwen2.5-1.5B-Instruct (shallow vs deep layers).

Runs the model with attn_implementation="eager" and captures per-layer
attention weights through a forward hook that moves the selected heads to CPU
and nulls the GPU tensor, so memory stays flat even at 8K context.

For each (context length, layer) pair, saves one PNG with, per selected head:
  left:  attention heatmap (downsampled by mean-pooling for display)
  right: column-mean attention curve (attention received per key position),
         which makes the attention sink at the first tokens visible.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from common import RESULTS, load_model, load_text


@torch.no_grad()
def capture_attentions(model, ids, layers, heads):
    """Exact eager attention probs for the given layers/heads, on CPU.

    Returns {layer: tensor (n_heads_sel, L, L)}.
    """
    saved = {}
    handles = []

    def make_hook(i):
        def hook(module, args, output):
            # eager attention modules return (attn_output, attn_weights)
            if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                raise RuntimeError(
                    f"layer {i}: attention weights not returned; "
                    "attn_implementation must be 'eager'"
                )
            w = output[1]  # (batch, heads, L, L)
            saved[i] = w[0, list(heads)].float().cpu()
            return (output[0], None)  # free the GPU tensor immediately

        return hook

    for i in layers:
        handles.append(model.model.layers[i].self_attn.register_forward_hook(make_hook(i)))
    model(input_ids=ids)
    for h in handles:
        h.remove()
    return saved


def plot_layer(attn, layer, heads, ctx, out_path):
    n = len(heads)
    fig, axes = plt.subplots(n, 3, figsize=(16, 4.2 * n), squeeze=False,
                             gridspec_kw={"width_ratios": [1.3, 1.5, 0.8]})
    for row, h in enumerate(heads):
        w = attn[row]  # (L, L)
        # heatmap, mean-pooled to <=512x512 for display
        L = w.shape[-1]
        step = max(1, L // 512)
        if step > 1:
            wv = w[: L // step * step, : L // step * step]
            wv = wv.view(L // step, step, L // step, step).mean(dim=(1, 3))
        else:
            wv = w
        ax = axes[row][0]
        im = ax.imshow(wv.numpy(), origin="lower", aspect="auto",
                       norm=matplotlib.colors.LogNorm(vmin=1e-5),
                       cmap="viridis", interpolation="nearest")
        ax.set_title(f"layer {layer} head {h} - attention map (ctx {ctx})")
        ax.set_xlabel("key position (pooled)")
        ax.set_ylabel("query position (pooled)")
        fig.colorbar(im, ax=ax, fraction=0.03)

        ax = axes[row][1]
        colmean = w.mean(dim=0).numpy()  # attention received by each key position
        ax.plot(colmean, linewidth=0.6)
        ax.set_yscale("log")
        ax.set_title(f"layer {layer} head {h} - mean attention per key position")
        ax.set_xlabel("key position")
        ax.set_ylabel("mean attention (log)")
        ax.axvspan(0, 4, color="red", alpha=0.2, label="first 4 tokens")
        ax.legend()

        # third column: linear-scale zoom of the sink region, which the
        # log-scale main curve compresses into unreadability (first token
        # outweighs the rest by orders of magnitude)
        n_zoom = min(64, len(colmean))
        axz = axes[row][2]
        axz.plot(range(n_zoom), colmean[:n_zoom], linewidth=0.8,
                 marker=".", markersize=2.5)
        axz.axvspan(0, 4, color="red", alpha=0.2)
        axz.set_title(f"first {n_zoom} tokens (linear)")
        axz.set_xlabel("key position")
        axz.set_ylabel("mean attention (linear)")
        axz.grid(alpha=0.3)
        # when the token-0 sink dwarfs the rest, clip the zoom y-axis so the
        # remaining positions stay readable, and annotate the true peak value
        rest_max = colmean[1:n_zoom].max() if n_zoom > 1 else colmean[0]
        if colmean[0] > 4 * rest_max:
            top = rest_max * 1.6
            axz.set_ylim(0, top)
            axz.annotate(f"token 0 = {colmean[0]:.3f}",
                         xy=(0, top), xytext=(n_zoom * 0.12, top * 0.8),
                         fontsize=8, color="darkred",
                         arrowprops=dict(arrowstyle="->", color="darkred",
                                         lw=0.8))
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[heatmap] saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, nargs="+", default=[2048, 8192])
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="default: layer 1 (shallow) and num_layers-2 (deep)")
    ap.add_argument("--heads", type=int, nargs="+", default=[0])
    ap.add_argument("--max-chars", type=int, default=200_000)
    args = ap.parse_args()

    tok, model = load_model(attn_implementation="eager")
    n_layers = model.config.num_hidden_layers
    layers = args.layers or [1, n_layers - 2]
    print(f"[heatmap] model loaded, {n_layers} layers, layers={layers}, heads={args.heads}")

    text = load_text()[: args.max_chars]
    full_ids = tok(text, return_tensors="pt").input_ids
    print(f"[heatmap] text tokenized: {full_ids.shape[1]} tokens available")

    os.makedirs(RESULTS, exist_ok=True)
    for ctx in args.ctx:
        if full_ids.shape[1] < ctx:
            print(f"[heatmap] not enough tokens for ctx={ctx}, skipping")
            continue
        ids = full_ids[:, :ctx].to(model.device)
        maps = capture_attentions(model, ids, layers, args.heads)
        for layer, attn in maps.items():
            out = os.path.join(RESULTS, f"sink_heatmap_ctx{ctx}_layer{layer}.png")
            plot_layer(attn, layer, args.heads, ctx, out)
        del maps
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
