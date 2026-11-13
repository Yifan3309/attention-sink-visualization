"""滑窗 PPL：不看头几个 token，PPL 会怎样（重算 vs 流式驱逐两种协议）

对 pg1342（傲慢与偏见）前 N 个 token 计算逐 token NLL，五个模式：

  full        : 上下文 [0, t)，整段一次前向 —— 基线
  win_nosink  : 滑窗重算 [t-w, t)，窗口头自建局部 sink（RoPE 相对位置下
                等价于把窗口当新序列），预测：不爆炸
  win_sink4   : 头 4 token ∪ [t-w+4, t) 重算 —— StreamingLLM 缓解（重算协议）
  stream_evict: 流式 KV cache，超长即驱逐最旧（**含头 4 个 sink token**）
                —— 真正的「不看头几个 token」，预测：t>w 后 PPL 爆炸
  stream_sink4: 流式 KV cache，永久保留头 4 个 + 最近 w-4 —— StreamingLLM 本尊

PPL 曲线 = 128-token 滚动均值 NLL 取 exp。

用法:
  python window_ppl.py                       # 默认 2048 token, w=512
  python window_ppl.py --max-tokens 4096 --window 256 --stride 2
"""
import argparse
import os

import numpy as np
import torch

from common import load_model, sliced_cache, RESULTS_DIR

TEXT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pg1342.txt")


def load_tokens(tok, max_tokens, text_path=TEXT_PATH):
    # 先按字符粗截断，避免整本书一次性 tokenize 触发超长警告
    with open(text_path, encoding="utf-8") as f:
        text = f.read()[: max_tokens * 8]
    ids = tok(text, add_special_tokens=False).input_ids[:max_tokens]
    print(f"文本: {os.path.basename(text_path)} -> {len(ids)} token")
    return ids


@torch.no_grad()
def nll_full(model, ids):
    """整段一次前向，返回逐 token NLL（nll[t] = -log p(x_t | x_<t)，t 从 1 起）"""
    x = torch.tensor([ids], device=model.device)
    logits = model(input_ids=x).logits[0].float()
    logp = torch.log_softmax(logits[:-1], dim=-1)
    tgt = torch.tensor(ids[1:], device=model.device)
    nll = -logp[torch.arange(len(ids) - 1), tgt]
    out = np.zeros(len(ids), dtype=np.float32)
    out[1:] = nll.cpu().numpy()
    return out


@torch.no_grad()
def nll_windows(model, ids, positions, w, with_sink4):
    """批量重算：positions 里每个 t 用窗口 [t-w, t)（或 sink4 版）预测 x_t。
    窗口内位置从 0 重编；对 Qwen3 的 RoPE，整体平移不改变相对位置，
    因此与保留绝对位置等价。"""
    nlls = np.zeros(len(positions), dtype=np.float32)
    B = 16
    for s in range(0, len(positions), B):
        chunk = positions[s:s + B]
        rows, tgts = [], []
        for t in chunk:
            if with_sink4:
                win = ids[:4] + ids[t - w + 4:t]
            else:
                win = ids[t - w:t]
            rows.append(win)
            tgts.append(ids[t])
        x = torch.tensor(rows, device=model.device)          # (B, w) 等长，无需 padding
        logits = model(input_ids=x).logits[:, -1].float()    # (B, V)
        logp = torch.log_softmax(logits, dim=-1)
        tgt = torch.tensor(tgts, device=model.device)
        nlls[s:s + len(chunk)] = (-logp[torch.arange(len(chunk)), tgt]).cpu().numpy()
    return nlls


@torch.no_grad()
def nll_stream(model, ids, w, keep_sink4, n_layers):
    """流式 KV cache 逐 token 前向；cache 超过 w 时驱逐最旧
    （keep_sink4=False 连头 4 个 sink 一起驱逐；True 则永久保留头 4 个）。
    位置用原始绝对位置；对 Qwen3 的 RoPE，与从 0 重编等价。"""
    T = len(ids)
    nll = np.zeros(T, dtype=np.float32)
    cache = None
    for t in range(T):
        if cache is not None and cache.get_seq_length() >= w:
            lo = 4 if keep_sink4 else 0     # sink4 版驱逐最旧的非 sink；否则连 sink 一起
            cache = sliced_cache(cache, lo, lo + 1, n_layers)
        kwargs = dict(
            input_ids=torch.tensor([[ids[t]]], device=model.device),
            position_ids=torch.tensor([[t]], device=model.device),
            use_cache=True,
        )
        if cache is not None:
            kwargs["past_key_values"] = cache
        out = model(**kwargs)
        cache = out.past_key_values
        if t + 1 < T:
            logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
            nll[t + 1] = -logp[ids[t + 1]].item()
    return nll


def rolling_ppl(nll, win=128):
    """PPL = exp(滚动均值 NLL)；前面不足窗口用可得前缀"""
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
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--window", type=int, default=512, help="滑窗宽度 w")
    ap.add_argument("--stride", type=int, default=4, help="重算模式 t>=w 后每 stride 个位置算一次")
    ap.add_argument("--text", default=TEXT_PATH, help="纯文本语料路径")
    ap.add_argument("--model", default=None, help="本地模型目录或 HF repo id")
    ap.add_argument("--out", default=str(RESULTS_DIR / "window_ppl.npz"))
    ap.add_argument("--plot", default=str(RESULTS_DIR / "window_ppl.png"))
    args = ap.parse_args()

    tok, model = load_model(eager=True, model=args.model)
    n_layers = model.config.num_hidden_layers
    ids = load_tokens(tok, args.max_tokens, args.text)
    T, w = len(ids), args.window

    print("full: 整段前向 ...")
    nll_f = nll_full(model, ids)

    positions = list(range(w, T, args.stride))
    print(f"win_nosink: {len(positions)} 个窗口重算 ...")
    nll_ns_sparse = nll_windows(model, ids, positions, w, with_sink4=False)
    print(f"win_sink4 : {len(positions)} 个窗口重算 ...")
    nll_s4_sparse = nll_windows(model, ids, positions, w, with_sink4=True)

    nll_ns, nll_s4 = nll_f.copy(), nll_f.copy()
    for i, t in enumerate(positions):
        nll_ns[t] = nll_ns_sparse[i]
        nll_s4[t] = nll_s4_sparse[i]

    print(f"stream_evict: 流式驱逐（含 sink），{T} 步 ...")
    nll_se = nll_stream(model, ids, w, keep_sink4=False, n_layers=n_layers)
    print(f"stream_sink4: 流式保留头 4 个（StreamingLLM），{T} 步 ...")
    nll_ss = nll_stream(model, ids, w, keep_sink4=True, n_layers=n_layers)

    # ---------- 分段 PPL 表 ----------
    print(f"\n=== 分段 PPL（w={w}）===")
    print(f"{'区间':>12} | {'full':>8} | {'win重算':>8} | {'win+sink4':>9} | "
          f"{'流式驱逐':>9} | {'流式+sink4':>9}")
    print("-" * 72)
    segs = [(1, w, "<w 同full")] + [
        (a, min(a + 512, T), f"{a}-{min(a + 512, T)}") for a in range(w, T, 512)
    ]
    for a, b, name in segs:
        print(f"{name:>12} | {seg_ppl(nll_f, a, b):8.2f} | {seg_ppl(nll_ns, a, b):8.2f} | "
              f"{seg_ppl(nll_s4, a, b):9.2f} | {seg_ppl(nll_se, a, b):9.2f} | "
              f"{seg_ppl(nll_ss, a, b):9.2f}")

    # ---------- 画图 ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5.5))
    curves = [
        (nll_f, "full context", "tab:green", "-", 1.2),
        (nll_ns, f"recompute window w={w} (no sink)", "gray", "--", 1.0),
        (nll_s4, f"recompute w/ first-4", "gray", ":", 1.0),
        (nll_se, f"streaming cache, evict oldest incl. sink", "tab:red", "-", 1.6),
        (nll_ss, f"streaming + keep first-4 (StreamingLLM)", "tab:blue", "-", 1.6),
    ]
    for nll, name, color, ls, lw in curves:
        ax.plot(rolling_ppl(nll), color=color, ls=ls, lw=lw, label=name)
    ax.axvline(w, color="gray", ls="--", lw=1, alpha=0.7, label=f"t = w = {w}")
    ax.set_yscale("log")
    ax.set_xlabel("token position t")
    ax.set_ylabel("PPL (128-token rolling)")
    ax.set_title("PPL anomaly: what if we don't look at the first few tokens (pg1342)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.plot)), exist_ok=True)
    fig.savefig(args.plot, dpi=150)
    print(f"\n✓ {args.plot}")

    # ---------- 存档 ----------
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, nll_full=nll_f, nll_win_nosink=nll_ns, nll_win_sink4=nll_s4,
             nll_stream_evict=nll_se, nll_stream_sink4=nll_ss,
             ids=np.array(ids, dtype=np.int32),
             window=np.array(w), stride=np.array(args.stride), max_tokens=np.array(T))
    print(f"✓ {args.out}")


if __name__ == "__main__":
    main()
