"""把末位的 residual 增量拆成 attention / MLP 两部分

对一段给定前缀做一次前向，在末位统计每层的：
  ‖attn‖  ‖mlp‖  ‖Δh‖=‖attn+mlp‖  ‖attn‖/(‖attn‖+‖mlp‖)  cos(attn, mlp)
对照同层的 sink 质量，回答："sink 高的层，attention 到底是空转，还是仍在做事"。

用法:
  python attention_sink/sublayer_writes.py
  python attention_sink/sublayer_writes.py --prompt "任意文本"
"""
import argparse
import os

import numpy as np
import torch

from common import load_model, RESULTS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(RESULTS_DIR / "long_trace.npz"))
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--model", default=None, help="本地模型目录或 HF repo id")
    ap.add_argument("--out", default=str(RESULTS_DIR / "sublayer_writes.npz"))
    args = ap.parse_args()

    sink = None
    if args.prompt:
        prompt = args.prompt
        ids = None
    else:
        d = np.load(args.npz, allow_pickle=True)
        prompt = str(d["prompt"])
        ids = list(d["prompt_ids"].tolist()) + list(d["gen_ids"].tolist())
        if "sink" in d.files:
            sink = d["sink"][-1]          # 末步逐层 sink
        print(f"复用前缀: {len(ids)} token")

    tok, model = load_model(eager=True, model=args.model)
    if ids is None:
        ids = tok(prompt, add_special_tokens=False).input_ids
    L = model.config.num_hidden_layers

    rec = {}
    hooks = []
    for i, layer in enumerate(model.model.layers):
        def mk(i, key):
            def hook(m, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                rec.setdefault(i, {})[key] = t[0, -1].detach().float().cpu()
            return hook
        hooks.append(layer.self_attn.register_forward_hook(mk(i, "attn")))
        hooks.append(layer.mlp.register_forward_hook(mk(i, "mlp")))

    with torch.no_grad():
        o = model(input_ids=torch.tensor([ids], device=model.device),
                  output_hidden_states=True)
    for h in hooks:
        h.remove()

    hs = [x[0, -1].float().cpu() for x in o.hidden_states]  # 各层末位向量（L+1 个）
    rows = []
    for i in range(L):
        a, m = rec[i]["attn"], rec[i]["mlp"]
        dh = a + m
        na, nm, nd = float(a.norm()), float(m.norm()), float(dh.norm())
        cos = float(a @ m / (a.norm() * m.norm() + 1e-8))
        rows.append((i, na, nm, nd, na / (na + nm + 1e-8), cos,
                     float(hs[i].norm()), float(hs[i + 1].norm())))
    R = np.array(rows)

    hdr = f"{'layer':>5} | {'|attn|':>7} | {'|mlp|':>7} | {'|dh|':>7} | {'attn%':>5} | {'cos':>5} | {'|h_l|':>7} | {'sink':>5}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for i, r in enumerate(rows):
        s = f"{sink[i]:5.3f}" if sink is not None else "  -  "
        print(f"{i:5d} | {r[1]:7.1f} | {r[2]:7.1f} | {r[3]:7.1f} | {r[4]:5.2f} | "
              f"{r[5]:+5.2f} | {r[6]:7.1f} | {s}")

    if sink is not None:
        def sp(x, y):
            return float(np.corrcoef(np.argsort(np.argsort(x)),
                                     np.argsort(np.argsort(y)))[0, 1])
        print(f"\nspearman(sink, attn份额)   = {sp(sink, R[:, 4]):+.3f}")
        print(f"spearman(sink, |attn|/|h|) = {sp(sink, R[:, 1] / R[:, 6]):+.3f}")
        print(f"spearman(sink, |mlp|/|h|)  = {sp(sink, R[:, 2] / R[:, 6]):+.3f}")
        hi = sink > 0.5
        print(f"sink>0.5 的层 ({hi.sum()} 层): attn 份额 {R[hi, 4].mean():.2f}，"
              f"|mlp| 均值 {R[hi, 2].mean():.1f}")
        print(f"sink<=0.5 的层 ({(~hi).sum()} 层): attn 份额 {R[~hi, 4].mean():.2f}，"
              f"|mlp| 均值 {R[~hi, 2].mean():.1f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, writes=R, sink=sink if sink is not None else np.array([]),
             tokens=np.array([tok.decode([i]) for i in ids[-8:]], dtype=object))
    print(f"\n✓ {args.out}")


if __name__ == "__main__":
    main()
