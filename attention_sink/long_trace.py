"""长序列逐 token 追踪：sink 质量 × 末位 hidden 稳定性

用 KV cache 贪心生成 --max-new 个 token，每一步只前向 1 个新位置，
在该位置上记录：
  hidden : (L+1, D)  各层输入向量（[0]=embedding，[L]=post-norm）
  sink   : (L,)      所有头平均、落回首 token 的注意力
  ent    : (L,)      注意力熵（所有头平均）
  top1   : (L,)      每头最大注意力（头间平均）
  p1     : 该步 top-1 概率
每 --ckpt 步再做一次整段前向，记录各位置范数，
用于看首 token 的 massive activation 随序列变长怎么走。

用法:
  python attention_sink/long_trace.py --max-new 512
  python attention_sink/long_trace.py --max-new 128 --prompt "自定义" --out attention_sink/results/lt.npz
"""
import argparse
import os

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from common import load_model, RESULTS_DIR

# 内容本身不重要，变量是"长度"；选一段能持续生成、不易退化的说明文
DEFAULT_PROMPT = (
    "Explain, step by step and in detail, how a bill becomes a law. "
    "Start with the first draft, then committee review, then the floor vote, "
    "and finish with the final signature. Describe what happens at each stage "
    "and why each stage exists."
)


def attn_stats(a: torch.Tensor):
    """a: (H, S) 某一个 layer、某一个 query 的注意力行。

    返回 (sink, ent, top1)，都在 16 个头上是平均。
    """
    sink = a[:, 0].mean()
    ent = (-(a * torch.log(a + 1e-12)).sum(-1)).mean()
    top1 = a.max(-1).values.mean()
    return float(sink), float(ent), float(top1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--ckpt", type=int, default=64,
                    help="每 N 步做一次整段前向记录位置范数；0 关闭")
    ap.add_argument("--model", default=None, help="本地模型目录或 HF repo id")
    ap.add_argument("--out", default=str(RESULTS_DIR / "long_trace.npz"))
    args = ap.parse_args()

    tok, model = load_model(eager=True, model=args.model)
    dev = model.device
    n_layer = model.config.num_hidden_layers

    ids = tok(args.prompt, add_special_tokens=False).input_ids
    print(f"prompt tokens: {len(ids)}")

    cache = DynamicCache()
    H, SINK, ENT, TOP1, P1, GEN = [], [], [], [], [], []
    ckpt_norms, ckpt_pos = {}, []

    with torch.no_grad():
        out = model(
            input_ids=torch.tensor([ids], device=dev),
            past_key_values=cache,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
        )

        for step in range(args.max_new):
            lg = out.logits[0, -1].float()
            nxt = int(lg.argmax())
            GEN.append(nxt)
            P1.append(float(torch.softmax(lg, -1).max()))

            # 当前这一步的"新位置"的 hidden（[0..L-1] 各层输入，[L] post-norm）
            H.append(
                torch.stack([h[0, -1] for h in out.hidden_states])
                .float().cpu().numpy().astype(np.float16)
            )

            s_l, e_l, t_l = [], [], []
            for l in range(n_layer):
                a = out.attentions[l][0][:, -1, :]  # (H, S)，q 维取最后一个
                s, e, t = attn_stats(a)
                s_l.append(s); e_l.append(e); t_l.append(t)
            SINK.append(s_l); ENT.append(e_l); TOP1.append(t_l)

            if args.ckpt and (step + 1) % args.ckpt == 0:
                prefix = torch.tensor([ids + GEN], device=dev)
                o2 = model(input_ids=prefix, output_hidden_states=True)
                n = torch.stack(
                    [h[0].float().norm(dim=-1) for h in o2.hidden_states]
                ).cpu().numpy()  # (L+1, S)
                ckpt_norms[f"norm_{len(ckpt_pos)}"] = n.astype(np.float32)
                ckpt_pos.append(len(ids) + step)
                print(f"  step {step+1:4d}  ctx={n.shape[1]:4d}  "
                      f"|h0|={n[:, 0].max():8.1f}  "
                      f"|h_mid|={np.median(n[:, 1:-1]):7.1f}  p1={P1[-1]:.3f}")

            if step == args.max_new - 1:
                break

            out = model(
                input_ids=torch.tensor([[nxt]], device=dev),
                past_key_values=cache,
                use_cache=True,
                output_attentions=True,
                output_hidden_states=True,
            )

    hidden = np.stack(H)                    # (T, L+1, D)
    payload = {
        "hidden": hidden,
        "sink": np.array(SINK, dtype=np.float32),
        "ent": np.array(ENT, dtype=np.float32),
        "top1mass": np.array(TOP1, dtype=np.float32),
        "p1": np.array(P1, dtype=np.float32),
        "gen_ids": np.array(GEN, dtype=np.int32),
        "prompt": args.prompt,
        "prompt_ids": np.array(ids, dtype=np.int32),
        "ckpt_pos": np.array(ckpt_pos, dtype=np.int32),
        "text": tok.decode(GEN),
        "num_model_layers": np.array(n_layer),
    }
    payload.update(ckpt_norms)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **payload)
    print(f"\n✓ {args.out}")
    print(f"  hidden {hidden.shape}  sink {payload['sink'].shape}")
    print(f"  生成片段: {payload['text'][:160]!r}")


if __name__ == "__main__":
    main()
