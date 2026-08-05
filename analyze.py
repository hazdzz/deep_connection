import os
import math
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model.gpt_base import GPTConfig, GPT


# ----------------------------------------------------------------------
# Load checkpoint
# ----------------------------------------------------------------------
def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd)
    model.eval().to(device)
    dataset = ckpt.get("config", {}).get("dataset", None)
    return model, dataset


# ----------------------------------------------------------------------
# Sample a val batch
# ----------------------------------------------------------------------
def make_get_batch(dataset, block_size, batch_size, device):
    data_dir = os.path.join("data", dataset)
    def get_batch():
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)
    return get_batch


# ----------------------------------------------------------------------
# Collect per-layer hidden states (replicate the first half of GPT.forward
# to avoid the last-position-only logits path)
# ----------------------------------------------------------------------
@torch.no_grad()
def collect_hidden_states(model, idx):
    b, t = idx.size()
    pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
    h = model.transformer.drop(model.transformer.wte(idx) + model.transformer.wpe(pos))
    hs = [h]                                   # h_0: embedding output
    for block in model.transformer.h:
        h = block(h)
        hs.append(h)                           # h_1 ... h_L
    return hs                                  # length L+1, each (B, T, C)


# ----------------------------------------------------------------------
# Angular Distance
# ----------------------------------------------------------------------
@torch.no_grad()
def angular_distance(h_l, h_lp, drop_first=True):
    if drop_first:                             # drop the first position (attention sink)
        h_l, h_lp = h_l[:, 1:], h_lp[:, 1:]
    cos = F.cosine_similarity(h_l.float(), h_lp.float(), dim=-1).clamp(-1.0, 1.0)
    return (torch.arccos(cos) / math.pi).mean().item()

@torch.no_grad()
def layerwise_angular_distance(model, get_batch, n=1, n_batches=20):
    L1 = len(model.transformer.h) + 1
    acc = np.zeros(L1 - n)
    for _ in range(n_batches):
        X, _ = get_batch()
        hs = collect_hidden_states(model, X)
        for l in range(L1 - n):
            acc[l] += angular_distance(hs[l], hs[l + n])
    return acc / n_batches                     # length L+1-n

@torch.no_grad()
def angular_distance_matrix(model, get_batch, n_batches=20):
    """M[n, l] = angular_distance(h_l, h_{l+n}); undefined entries stay NaN."""
    L1 = len(model.transformer.h) + 1          # number of hidden states = L+1 (h_0..h_L)
    acc = np.zeros((L1, L1))
    cnt = np.zeros((L1, L1))
    for _ in range(n_batches):
        X, _ = get_batch()
        hs = collect_hidden_states(model, X)
        for l in range(L1):
            for n in range(1, L1 - l):
                acc[n, l] += angular_distance(hs[l], hs[l + n])
                cnt[n, l] += 1
    M = np.full((L1, L1), np.nan)
    mask = cnt > 0
    M[mask] = acc[mask] / cnt[mask]
    return M                                   # shape (L1, L1), row index = gap n, col index = layer l


# ----------------------------------------------------------------------
# Per-layer hidden norm / variance
# ----------------------------------------------------------------------
@torch.no_grad()
def layerwise_hidden_stats(model, get_batch, n_batches=20):
    L1 = len(model.transformer.h) + 1
    norm_acc = np.zeros(L1)
    var_acc = np.zeros(L1)
    maxabs_acc = np.zeros(L1)
    for _ in range(n_batches):
        X, _ = get_batch()
        hs = collect_hidden_states(model, X)
        for l, h in enumerate(hs):
            hf = h.float()
            norm_acc[l] += hf.norm(dim=-1).mean().item()      # mean ||x||
            var_acc[l] += hf.var(dim=-1).mean().item()        # mean channel variance
            maxabs_acc[l] += hf.abs().max().item()            # max absolute value (massive activation)
    return norm_acc / n_batches, var_acc / n_batches, maxabs_acc / n_batches


# ----------------------------------------------------------------------
# Performance Drop (per-layer pruning; metric is val loss / PPL)
# ----------------------------------------------------------------------
@torch.no_grad()
def eval_loss(model, get_batch, n_batches=20):
    losses = []
    for _ in range(n_batches):
        X, Y = get_batch()
        _, loss = model(X, Y)                  # pass targets to trigger the full forward
        losses.append(loss.item())
    return float(np.mean(losses))

@torch.no_grad()
def performance_drop(model, get_batch, n_batches=20):
    P_org = eval_loss(model, get_batch, n_batches)
    drops = []
    for j in range(len(model.transformer.h)):
        orig = model.transformer.h[j]
        model.transformer.h[j] = nn.Identity()      # prune layer j (residual passes through)
        P_prn = eval_loss(model, get_batch, n_batches)
        model.transformer.h[j] = orig               # restore
        drops.append(P_prn - P_org)                 # larger = layer is more important
    return P_org, np.array(drops)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="path to ckpt.pt")
    ap.add_argument("--dataset", type=str, default=None, help="override the dataset stored in the ckpt")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_batches", type=int, default=20)
    ap.add_argument("--gap_n", type=int, default=1, help="layer gap for angular distance")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="analyze")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model, ds_in_ckpt = load_model(args.ckpt, args.device)
    dataset = args.dataset or ds_in_ckpt
    assert dataset is not None, "please specify the dataset with --dataset"
    block_size = model.config.block_size

    get_batch = make_get_batch(dataset, block_size, args.batch_size, args.device)

    print("computing angular distance ...")
    ang = layerwise_angular_distance(model, get_batch, n=args.gap_n, n_batches=args.n_batches)

    print("computing angular distance matrix ...")
    ang_mat = angular_distance_matrix(model, get_batch, n_batches=args.n_batches)

    print("computing hidden stats ...")
    norms, varis, maxabs = layerwise_hidden_stats(model, get_batch, n_batches=args.n_batches)

    print("computing performance drop ...")
    P_org, drops = performance_drop(model, get_batch, n_batches=args.n_batches)
    print(f"baseline val loss = {P_org:.4f}, PPL = {math.exp(P_org):.3f}")

    tag = os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))

    # save the angular distance matrix for later re-plotting
    np.save(os.path.join(args.out, f"angdist_matrix_{tag}.npy"), ang_mat)

    # save CSV
    with open(os.path.join(args.out, f"effdepth_{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "hidden_norm", "hidden_var", "max_abs",
                    f"angular_dist(n={args.gap_n})", "perf_drop"])
        L1 = len(norms)
        for l in range(L1):
            ang_l = ang[l] if l < len(ang) else ""
            drop_l = drops[l] if l < len(drops) else ""   # drops has length L (no h_0)
            w.writerow([l, norms[l], varis[l], maxabs[l], ang_l, drop_l])

    # plotting (optional)
    try:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].plot(range(len(norms)), norms, "o-"); axes[0, 0].set_title("hidden ||x||"); axes[0, 0].set_xlabel("layer")
        axes[0, 1].plot(range(len(varis)), varis, "o-"); axes[0, 1].set_title("hidden var"); axes[0, 1].set_xlabel("layer")
        axes[1, 0].plot(range(len(ang)), ang, "o-"); axes[1, 0].set_title(f"angular distance (n={args.gap_n})"); axes[1, 0].set_xlabel("layer l")
        axes[1, 1].bar(range(1, len(drops) + 1), drops); axes[1, 1].set_title("performance drop (delta loss)"); axes[1, 1].set_xlabel("pruned layer")
        fig.suptitle(tag)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"effdepth_{tag}.pdf"))
        print(f"saved {args.out}/effdepth_{tag}.pdf")
    except Exception as e:
        print(f"skip plotting: {e}")

    # angular distance heatmap (sweep over all gaps n), like the published figures
    try:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        im = ax2.imshow(ang_mat, origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
        ax2.set_xlabel(r"Layer Index $\ell$")
        ax2.set_ylabel(r"Subsequent $n^{th}$ Layer")
        ax2.set_title(f"Angular Distance ({tag})")
        fig2.colorbar(im, ax=ax2)
        fig2.tight_layout()
        fig2.savefig(os.path.join(args.out, f"angdist_heatmap_{tag}.pdf"))
        print(f"saved {args.out}/angdist_heatmap_{tag}.pdf")
    except Exception as e:
        print(f"skip heatmap plotting: {e}")
