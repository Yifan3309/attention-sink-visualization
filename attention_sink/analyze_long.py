"""长序列追踪的离线分析：末位稳定性 vs sink 质量。

读 long_trace.py 导出的 npz，算三组量：
  1. 逐层漂移：末位向量相邻层 1-cos / relL2（只用 pre-norm 层，post-norm 不参与）
  2. logit lens：各层过 norm + lm_head 的 top1 / p1 / 逐层 KL / 每步翻转次数 / 收敛深度
  3. sink：逐层首 token 注意力、归一化熵

用法:
  python analyze_long.py
  python attention_sink/analyze_long.py --npz attention_sink/results/x.npz --plot attention_sink/results/x.png
"""
import argparse
import os

import numpy as np
import torch
from safetensors import safe_open

from common import RESULTS_DIR, load_model_path, resolve_weight_files


def load_head(model_path=None):
    """直接从 safetensors 取 lm_head / final norm，不加载整个模型。"""
    keys = ["lm_head.weight", "model.norm.weight"]
    files = resolve_weight_files(load_model_path(model_path), keys)
    tensors = {}
    for key, path in files.items():
        with safe_open(str(path), "pt") as f:
            tensors[key] = f.get_tensor(key).float()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return tensors["lm_head.weight"].to(dev), tensors["model.norm.weight"].to(dev)


def logit_lens(H, W, g, eps=1e-6, chunk=128):
    """H: (T, L+1, D) numpy → top1(T,L), p1(T,L), kl(T,L-1)。

    只用到 pre-norm 的 [0..L-1]（[L] 是 post-norm，不参与逐层比较）；
    同一块内同时算相邻两层，KL 不必跨块保存整层分布。
    """
    T, L1, D = H.shape
    L = L1 - 1
    top1 = np.zeros((T, L), dtype=np.int32)
    p1 = np.zeros((T, L), dtype=np.float32)
    kl = np.zeros((T, L - 1), dtype=np.float32)

    X = torch.tensor(H[:, :L, :], device=W.device, dtype=torch.float32)
    X = X / torch.sqrt(X.pow(2).mean(-1, keepdim=True) + eps) * g
    for s in range(0, T, chunk):
        logps = []
        for layer in range(L):
            logps.append(torch.log_softmax(X[s:s + chunk, layer] @ W.T, dim=-1))
        for layer in range(L):
            top1[s:s + chunk, layer] = logps[layer].argmax(-1).cpu().numpy()
            p1[s:s + chunk, layer] = logps[layer].max(-1).values.exp().cpu().numpy()
            if layer > 0:
                kl[s:s + chunk, layer - 1] = (
                    (logps[layer].exp() * (logps[layer] - logps[layer - 1]))
                    .sum(-1).cpu().numpy()
                )
        del logps
        torch.cuda.empty_cache()
    return top1, p1, kl


def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(RESULTS_DIR / "long_trace.npz"))
    ap.add_argument("--plot", default=str(RESULTS_DIR / "long_trace.png"))
    ap.add_argument("--win", type=int, default=64, help="分段统计的窗口")
    ap.add_argument("--deep-start", type=int, default=12, help="“深层”起始层号")
    ap.add_argument("--model", default=None, help="本地模型目录或 HF repo id")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    H = d["hidden"].astype(np.float32)      # (T, L+1, D)
    sink = d["sink"]                        # (T, L)
    ent = d["ent"]
    p1_step = d["p1"]
    T, L1, D = H.shape
    L = L1 - 1
    prompt_len = len(d["prompt_ids"])
    deep_start = min(max(args.deep_start, 0), L)

    # ---------- 1. 逐层漂移 ----------
    u, v = H[:, :L, :], H[:, 1:L1, :]
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-8)
    drift = 1 - cos
    rel = np.linalg.norm(v - u, axis=-1) / (np.linalg.norm(u, axis=-1) + 1e-8)

    # ---------- 2. logit lens ----------
    W, g = load_head(args.model)
    top1, p1_layer, kl = logit_lens(H, W, g)
    flips = np.zeros(T, dtype=np.int32)
    for t in range(T):
        flips[t] = (np.diff(top1[t, :L]) != 0).sum()
    conv = np.full(T, L - 1, dtype=np.int32)
    for t in range(T):
        for layer in range(L - 1, -1, -1):
            if layer > 0 and top1[t, layer] == top1[t, L - 1]:
                conv[t] = layer
            else:
                break

    # ---------- 3. sink ----------
    sink_deep = sink[:, deep_start:]
    sink_mean = sink.mean(1)
    seq_len = prompt_len + np.arange(T)
    ent_norm = ent / np.log(seq_len)[:, None]

    nw = T // args.win
    print(f"T={T}  L={L}  窗口={args.win}  prompt={prompt_len} token")
    print(f"\n{'窗口':>10} | {'drift':>6} | {'relL2':>6} | {'sink':>6} | {'sink深':>6} | "
          f"{'熵/lnS':>6} | {'翻转':>5} | {'收敛层':>6} | {'p1':>5}")
    for i in range(nw):
        a, b = i * args.win, (i + 1) * args.win
        print(f"{a:>4}-{b - 1:<5} | {drift[a:b, :L-1].mean():6.3f} | "
              f"{rel[a:b, :L-1].mean():6.3f} | {sink_mean[a:b].mean():6.3f} | "
              f"{sink_deep[a:b].mean():6.3f} | {ent_norm[a:b].mean():6.3f} | "
              f"{flips[a:b].mean():5.1f} | {conv[a:b].mean():6.1f} | "
              f"{p1_step[a:b].mean():5.3f}")

    print("\n末步逐层 (l: sink / 1-cos / lens-p1):")
    for layer in range(0, L, 2):
        print(f"  L{layer:02d}: {sink[-1, layer]:5.3f} / {drift[-1, layer]:6.4f} / "
              f"{p1_layer[-1, layer]:5.3f}")

    print(f"\n相关 (n={T} 步，逐层平均后):")
    pairs = [
        ("sink深 vs drift", sink_deep.mean(1), drift[:, :L-1].mean(1)),
        ("sink深 vs relL2", sink_deep.mean(1), rel[:, :L-1].mean(1)),
        ("sink深 vs 翻转数", sink_deep.mean(1), flips.astype(float)),
        ("sink深 vs 收敛层", sink_deep.mean(1), conv.astype(float)),
        ("sink深 vs p1", sink_deep.mean(1), p1_step),
        ("drift vs 翻转数", drift[:, :L-1].mean(1), flips.astype(float)),
        ("drift vs 步号", drift[:, :L-1].mean(1), np.arange(T, dtype=float)),
        ("sink深 vs 步号", sink_deep.mean(1), np.arange(T, dtype=float)),
    ]
    for name, x, y in pairs:
        print(f"  {name:<18} pearson={pearson(x, y):+.3f}  spearman={spearman(x, y):+.3f}")

    # ---------- 4. 画图 ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 9))

    ax = axes[0]
    ax.plot(drift[:, :L-1].mean(1), color="crimson", lw=1, label=f"mean 1-cos (L0-{L-2})")
    ax.plot(sink_deep.mean(1), color="steelblue", lw=1,
            label=f"sink mass (L{deep_start}-{L-1})")
    ax.set_ylabel("value")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("long trace: representation drift vs sink mass", fontsize=11)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(p1_step, color="darkgreen", lw=1, label="step top-1 prob")
    ma = np.convolve(flips, np.ones(32) / 32, mode="valid")
    ax.plot(np.arange(len(ma)) + (32 - 1) / 2, ma, color="orange",
            lw=1.5, label="logit-lens flips (32-step MA)")
    ax.set_ylabel("value")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[2]
    w = args.win
    ax.plot(np.arange(L), sink[-w:].mean(0), "o-", ms=3, color="steelblue",
            label=f"sink mass per layer (last {w} steps)")
    ax.plot(np.arange(L - 1), drift[-w:, :L - 1].mean(0), "s-", ms=3, color="crimson",
            label=f"layer write 1-cos (last {w} steps)")
    ax.set_xlabel("layer")
    ax.set_ylabel("value")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.plot)), exist_ok=True)
    plt.savefig(args.plot, dpi=150)
    print(f"\n✓ {args.plot}")


if __name__ == "__main__":
    main()
