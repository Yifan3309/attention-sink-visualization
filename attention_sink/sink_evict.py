"""中途驱逐实验：推到一半舍弃头四个 token，PPL 会怎样

teacher-forced 固定文本（pg1342 前 N token），KV cache 逐 token 前向，
在 K = T/2 处对 cache 做一次干预后继续。三条件（模型只加载一次）：

  baseline : 无干预
  first4   : 删除 KV 位置 0-3（头四个 token，即 attention sink 载体）
  after4   : 删除 KV 位置 4-7（同样删 4 个 token、丢失等量上下文，但不是 sink 的对照）

关键细节：驱逐后显式传 position_ids = 原绝对位置，否则 HF 按 cache 长度重编
位置（整体 -4 平移），会把「失去 sink」和「位置平移」混在一起。
PPL 口径：分段 exp(mean NLL)；曲线 = 128-token 滚动 PPL。

用法:
  python sink_evict.py                       # 1024 token, K=512
  python sink_evict.py --max-tokens 2048 --evict-at 1024
"""
import argparse
import os

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from common import load_model, sliced_cache, RESULTS_DIR

TEXT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pg1342.txt")


@torch.no_grad()
def run_condition(model, ids, K, mode, n_layers):
    """逐 token teacher-forced。mode: baseline | first4 | after4。
    干预发生在喂入第 K 个 token 之前（cache 此刻含 [0..K-1]），
    因此 nll[K] 仍是干净的，nll[K+1] 起受影响。"""
    T = len(ids)
    nll = np.zeros(T, dtype=np.float32)
    p1 = np.zeros(T, dtype=np.float32)

    cache = DynamicCache()
    for t in range(T):
        if t == K and mode != "baseline":
            lo, hi = (0, 4) if mode == "first4" else (4, 8)
            expected = cache.get_seq_length() - (hi - lo)
            cache = sliced_cache(cache, lo, hi, n_layers)
            assert cache.get_seq_length() == expected, "cache 长度与预期不符"
            print(f"    [{mode}] step {t}: 驱逐 KV 位置 [{lo},{hi})，cache {expected + (hi - lo)} -> {expected}")

        out = model(
            input_ids=torch.tensor([[ids[t]]], device=model.device),
            past_key_values=cache,
            position_ids=torch.tensor([[t]], device=model.device),
            use_cache=True,
        )
        if t + 1 < T:
            logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
            nll[t + 1] = -logp[ids[t + 1]].item()
            p1[t + 1] = logp.max().exp().item()
    return nll, p1


def rolling_ppl(nll, win=128):
    T = len(nll)
    out = np.full(T, np.nan, dtype=np.float32)
    for t in range(1, T):
        a = max(1, t - win + 1)
        out[t] = np.exp(nll[a:t + 1].mean())
    return out


def seg_ppl(nll, a, b):
    seg = nll[a:b]
    return np.exp(seg.mean()) if len(seg) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--evict-at", type=int, default=None, help="干预步 K，默认 T/2")
    ap.add_argument("--text", default=TEXT_PATH, help="纯文本语料路径")
    ap.add_argument("--model", default=None, help="本地模型目录或 HF repo id")
    ap.add_argument("--out", default=str(RESULTS_DIR / "sink_evict.npz"))
    ap.add_argument("--plot", default=str(RESULTS_DIR / "sink_evict.png"))
    args = ap.parse_args()

    tok, model = load_model(eager=True, model=args.model)
    n_layers = model.config.num_hidden_layers

    # 先按字符粗截断，避免整本书一次性 tokenize 触发超长警告
    with open(args.text, encoding="utf-8") as f:
        text = f.read()[: args.max_tokens * 8]
    ids = tok(text, add_special_tokens=False).input_ids[:args.max_tokens]
    T = len(ids)
    K = args.evict_at if args.evict_at is not None else T // 2
    print(f"文本: pg1342.txt -> {T} token，干预步 K={K}")
    print(f"被驱逐 token: first4={[tok.decode([i]) for i in ids[:4]]!r}  "
          f"after4={[tok.decode([i]) for i in ids[4:8]]!r}")

    results = {}
    for mode in ["baseline", "first4", "after4"]:
        print(f"运行条件: {mode} ...")
        results[mode] = run_condition(model, ids, K, mode, n_layers)

    # 内在校验：干预前 (t <= K) 三条件 NLL 应完全一致
    for mode in ["first4", "after4"]:
        assert np.allclose(results["baseline"][0][:K + 1], results[mode][0][:K + 1]), \
            f"{mode} 在干预前的 NLL 与 baseline 不一致！"
    print("✓ 校验通过：干预前 (t<=K) 三条件 NLL 完全一致")

    # ---------- 分段 PPL 表 ----------
    nll_b, nll_f4, nll_a4 = results["baseline"][0], results["first4"][0], results["after4"][0]
    segs = [(1, K, f"[1,{K}) 干预前"), (K, K + 64, f"[{K},{K + 64})"),
            (K + 64, K + 128, f"[{K + 64},{K + 128})"), (K + 128, T, f"[{K + 128},{T})"),
            (1, T, "全程")]
    print(f"\n=== 分段 PPL（K={K}）===")
    print(f"{'区间':>16} | {'baseline':>8} | {'first4':>8} | {'after4':>8}")
    print("-" * 54)
    for a, b, name in segs:
        print(f"{name:>16} | {seg_ppl(nll_b, a, b):8.2f} | {seg_ppl(nll_f4, a, b):8.2f} | "
              f"{seg_ppl(nll_a4, a, b):8.2f}")

    # ---------- 画图 ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))

    for nll, name, color in [
        (nll_b, "baseline", "tab:green"),
        (nll_f4, "evict first-4 (sink) at K", "tab:red"),
        (nll_a4, "evict positions 4-7 (control)", "tab:blue"),
    ]:
        ax1.plot(rolling_ppl(nll), color=color, lw=1.2, label=name)
        ax2.plot(rolling_ppl(nll, win=16), color=color, lw=1.2, label=name)
    for ax in (ax1, ax2):
        ax.axvline(K, color="gray", ls="--", lw=1, alpha=0.8)
        ax.set_yscale("log")
        ax.set_ylabel("PPL")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which="both")
    ax1.set_title(f"mid-stream KV eviction at K={K} (pg1342, {T} tokens)", fontsize=11)
    ax2.set_xlim(max(1, K - 128), min(T - 1, K + 128))
    ax2.set_xlabel("token position t (zoom: K +/- 128, rolling window 16)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.plot)), exist_ok=True)
    fig.savefig(args.plot, dpi=150)
    print(f"\n✓ {args.plot}")

    # ---------- 存档 ----------
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out,
             nll_baseline=nll_b, nll_first4=nll_f4, nll_after4=nll_a4,
             p1_baseline=results["baseline"][1], p1_first4=results["first4"][1],
             p1_after4=results["after4"][1],
             ids=np.array(ids, dtype=np.int32), K=np.array(K),
             max_tokens=np.array(T))
    print(f"✓ {args.out}")


if __name__ == "__main__":
    main()
