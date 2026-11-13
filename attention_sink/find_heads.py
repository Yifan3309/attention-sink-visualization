"""Attention Sink 发现：扫描每个 (layer, head) 的 sink / mid-content 权重。

读取 extract_attn.py 导出的全层 attention (L, H, S, S)，
固定 query（默认最后一个 token），把每个 head 分成两类：
  worker：对“中间内容”注意力总和高；中间的统计范围是
          位置 2 到 S-4（排除首 token、位置 1 和最后 3 个 token）。
  sink  ：几乎把全部注意力倾倒在首 token 上。

用法:
  python attention_sink/find_heads.py
  python attention_sink/find_heads.py --npz attention_sink/results/attn.npz --query -1
"""
import argparse

import numpy as np

from common import RESULTS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(RESULTS_DIR / "attn.npz"))
    ap.add_argument("--query", type=int, default=-1, help="query 行；-1 = 最后一个 token")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    attn = d["attn"]            # (L, H, S, S)
    tokens = d["tokens"].tolist()
    L, H, S, _ = attn.shape

    q = args.query if args.query >= 0 else S + args.query
    if not 0 <= q < S:
        raise ValueError(f"query 越界: {args.query}（序列长度 {S}）")

    # “中间内容”：位置 2..S-4；不按标点过滤，只固定排除首 token、位置 1 和最后 3 个 token。
    mid_indices = list(range(2, max(2, S - 3)))
    first_token = tokens[0]
    print(f"npz: {args.npz}")
    print(f"query: {q} ({tokens[q]!r})")
    print(f"正在扫描关注的中间 tokens: {[tokens[i] for i in mid_indices]}\n")

    results = []
    for layer in range(L):
        for head in range(H):
            row = attn[layer, head, q, :]
            sink_weight = row[0]
            mid_weight = row[mid_indices].sum()
            results.append((layer, head, sink_weight, mid_weight))

    results_mid = sorted(results, key=lambda x: x[3], reverse=True)
    print("=== 👑 最关注中间内容的 Head Top 10 ===")
    print(f"{'Layer':<6} | {'Head':<5} | {'Sink(' + first_token + ')':<14} | {'Mid-Content Sum':<15}")
    print("-" * 55)
    for layer, head, sink, mid in results_mid[:10]:
        print(f"L{layer:<5} | H{head:<4} | {sink:.4f}         | {mid:.4f}")

    results_sink = sorted(results, key=lambda x: x[2], reverse=True)
    print("\n=== 🗑️ 最强 Attention Sink Head Top 5 ===")
    print(f"{'Layer':<6} | {'Head':<5} | {'Sink(' + first_token + ')':<14} | {'Mid-Content Sum':<15}")
    print("-" * 55)
    for layer, head, sink, mid in results_sink[:5]:
        print(f"L{layer:<5} | H{head:<4} | {sink:.4f}         | {mid:.4f}")


if __name__ == "__main__":
    main()
