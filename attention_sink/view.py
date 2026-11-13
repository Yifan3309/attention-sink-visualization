"""view_attn.py (view.py) — 查看 extract_attn.py 导出的注意力。

用法:
  python view.py single attn.npz out.png --layer 15 --head 7
  python view.py sweep attn.npz out.png --head 7 --query -1
  ...
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def ensure_parent(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def load_attn(npz_path):
    """统一读取逻辑，兼容只有 1 层（3D）和全层（4D）的 npz"""
    d = np.load(npz_path, allow_pickle=True)
    attn = d["attn"]
    if attn.ndim == 3:
        attn = attn[np.newaxis, ...]
        layer_indices = np.array([-1], dtype=np.int32)
        print("⚠ 警告：检测到旧格式 npz（仅 1 层）。建议重跑 extract_attn.py 导出全层。")
    else:
        layer_indices = d["layer_indices"]
    return attn, layer_indices, d["tokens"].tolist()


def resolve_layer(layer_indices, spec):
    """将 'last' / '-1' / 真实层号 (如 15) 转为 attn 数组的下标"""
    if spec is None or spec in ("last", "-1"):
        return len(layer_indices) - 1
    value = int(spec)
    hits = np.where(layer_indices == value)[0]
    if len(hits) > 0:
        return int(hits[0])
    if 0 <= value < len(layer_indices):
        return value
    raise ValueError(f"层 {value} 不在 {layer_indices.tolist()} 中")


# ---------- single: 单层 × 单 head 的完整 (S x S) 静态热图 ----------
def cmd_single(args):
    attn, layer_indices, tokens = load_attn(args.npz)
    L, H, S, _ = attn.shape
    li = resolve_layer(layer_indices, args.layer)
    real_layer = int(layer_indices[li])

    if args.head in ("mean", "avg"):
        mat = attn[li].mean(0)
        head_txt = "mean over heads"
    else:
        mat = attn[li][int(args.head)]
        head_txt = f"head {args.head}"

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap="viridis", aspect="auto", interpolation="nearest")
    ax.set_xticks(range(S)); ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticks(range(S)); ax.set_yticklabels(tokens, fontsize=8)
    ax.set_xlabel("key"); ax.set_ylabel("query")
    ax.set_title(f"layer L{real_layer} | {head_txt} (full query x key)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    ensure_parent(args.out)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"✓ {args.out}  layer=L{real_layer}  head={args.head}")


# ---------- sweep: (层 × token) 静态热图 ----------
def cmd_sweep(args):
    attn, layer_indices, tokens = load_attn(args.npz)
    L, H, S, _ = attn.shape

    if args.head in ("mean", "avg"):
        m = attn.mean(1)                     # (L, S, S)
        head_txt = "mean over heads"
    else:
        m = attn[:, int(args.head)]          # (L, S, S)
        head_txt = f"head {args.head}"

    q = args.query if args.query >= 0 else S + args.query
    data = m[:, q, :]                        # (L, S)

    fig_w = max(10, S * 0.32)
    fig_h = max(4, L * 0.38)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(data, cmap="viridis", aspect="auto", interpolation="nearest")
    ax.set_xticks(range(S)); ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticks(range(L))
    ax.set_yticklabels([f"L{layer}" for layer in layer_indices], fontsize=8)
    ax.set_xlabel(f"key  (query = {tokens[q]!r})")
    ax.set_ylabel("layer")
    ax.set_title(f"head={head_txt}  query row {q}")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    ensure_parent(args.out)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"✓ {args.out}  shape={data.shape}  (layer × token)")


# ---------- animate: 单层 × 单 head 的逐 query 动画 ----------
def cmd_animate(args):
    attn, layer_indices, tokens = load_attn(args.npz)
    L, H, S, _ = attn.shape

    li = resolve_layer(layer_indices, args.layer)
    real_layer = int(layer_indices[li])

    if args.head in ("mean", "avg"):
        mat = attn[li].mean(0)
        head_txt = "mean over heads"
    else:
        mat = attn[li][int(args.head)]
        head_txt = f"head {args.head}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    im = ax1.imshow(mat, cmap="viridis", aspect="auto", interpolation="nearest")
    ax1.set_xticks(range(S)); ax1.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax1.set_yticks(range(S)); ax1.set_yticklabels(tokens, fontsize=8)
    ax1.set_xlabel("key"); ax1.set_ylabel("query")
    ax1.set_title(f"layer L{real_layer} | {head_txt}")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    rect = mpatches.Rectangle((-0.5, -0.5), S, 1, fill=False,
                              edgecolor="red", linewidth=2)
    ax1.add_patch(rect)

    vmax = float(mat.max()) * 1.1 or 1e-6
    bars = ax2.bar(range(S), mat[0], color="steelblue")
    ax2.set_xticks(range(S)); ax2.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax2.set_ylim(0, vmax); ax2.set_ylabel("attention weight")

    def update(f):
        rect.set_xy((-0.5, f - 0.5))
        ax2.set_title(f"query = {tokens[f]!r}  (row {f}/{S-1})")
        for bar, value in zip(bars, mat[f]):
            bar.set_height(value)
        return [rect, *bars]

    ani = FuncAnimation(fig, update, frames=S, interval=1000 // args.fps, blit=False)
    ensure_parent(args.out)
    ani.save(args.out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"✓ {args.out}  layer=L{real_layer}  head={args.head}  frames={S}")


# ---------- layer-animate: 沿层动画 ----------
def cmd_layer_animate(args):
    attn, layer_indices, tokens = load_attn(args.npz)
    L, H, S, _ = attn.shape

    if args.head in ("mean", "avg"):
        m = attn.mean(1)
        head_txt = "mean over heads"
    else:
        m = attn[:, int(args.head)]
        head_txt = f"head {args.head}"

    q = args.query if args.query >= 0 else S + args.query
    vmax = float(m[:, q, :].max()) * 1.1 or 1e-6

    fig, ax = plt.subplots(figsize=(max(10, S * 0.4), 5))
    bars = ax.bar(range(S), m[0, q], color="steelblue")
    ax.set_xticks(range(S)); ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_ylim(0, vmax); ax.set_ylabel("attention weight")

    def update(f):
        ax.set_title(f"layer L{int(layer_indices[f])}  query={tokens[q]!r}  head={head_txt}")
        for bar, value in zip(bars, m[f, q]):
            bar.set_height(value)
        return bars

    ani = FuncAnimation(fig, update, frames=L, interval=1000 // args.fps, blit=False)
    ensure_parent(args.out)
    ani.save(args.out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"✓ {args.out}  layers={layer_indices.tolist()}  head={args.head}")


# ---------- CLI ----------
def build_cli():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp, *, need_layer=False, need_query=False):
        sp.add_argument("npz")
        sp.add_argument("out")
        sp.add_argument("--head", default="mean", help="mean / 0 / 5 / 7")
        sp.add_argument("--fps", type=int, default=2)
        if need_layer:
            sp.add_argument("--layer", default="last",
                            help="last / 真实层号(如16) / 位置(-1 也 = last)")
        if need_query:
            sp.add_argument("--query", type=int, default=-1,
                            help="query token 行；-1 = 最后一个")

    s1 = sub.add_parser("animate", help="单层 × 单 head 的逐 query 动画")
    add_common(s1, need_layer=True)
    s1.set_defaults(func=cmd_animate)

    s2 = sub.add_parser("sweep", help="层 × token 的静态热图（一个 query）")
    add_common(s2, need_query=True)
    s2.set_defaults(func=cmd_sweep)

    s3 = sub.add_parser("layer-animate", help="同一 query 沿层变化的动画")
    add_common(s3, need_query=True)
    s3.set_defaults(func=cmd_layer_animate)

    s4 = sub.add_parser("single", help="单层 × 单 head 的完整 (query x key) 静态热图")
    add_common(s4, need_layer=True)
    s4.set_defaults(func=cmd_single)

    return p


if __name__ == "__main__":
    args = build_cli().parse_args()
    args.func(args)
