"""导出注意力 / hidden states 到 npz。

一次前向，保存所选层的 attention (selected, H, S, S) 与可选的
hidden_states (selected, S, D)，附带 tokens / prompt 元信息，
供 find_heads.py、view.py、analyze_hidden.py 离线分析。

用法（仓库根目录）:
  python attention_sink/extract_attn.py --out attention_sink/results/attn.npz
  python attention_sink/extract_attn.py --hidden --out attention_sink/results/attn_hidden.npz
  python attention_sink/extract_attn.py --prompt "自定义文本" --layers 0,4,8
"""
import argparse
import os

import numpy as np
import torch

from common import load_model, RESULTS_DIR

# 历史实验用的首句 prompt。默认加尾空格 = 21 token，
# 此时末 token（空格）就是预测位置，供 find_heads.py 的 q=-1 使用；
# --no-tail-space 则为旧版 20 token，末位是标点。
PROMPT_BASE = (
    "Once upon a time, in a small village by the sea, there lived an old fisherman."
)
DEFAULT_PROMPT = PROMPT_BASE + " "
# 完整三句渔夫故事（55 token）——旧 attn.gif（末层 head-mean 动画）用的 prompt
STORY_PROMPT = (
    "Once upon a time, in a small village by the sea, there lived an old fisherman. "
    "Every morning he rowed his boat out before sunrise, and every evening he came back "
    "with a story to tell. One day, he found something strange tangled in his net:"
)


def parse_layers(spec: str, n_layers: int) -> list[int]:
    """-1 / all / 0,4,8 / 2:10 / -3,-1  ->  list[int]"""
    if spec.strip().lower() == "all":
        return list(range(n_layers))
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            a, b = part.split(":")
            out.extend(range(int(a), int(b)))
        else:
            out.append(int(part))
    out = [layer if layer >= 0 else n_layers + layer for layer in out]
    out = sorted(set(out))
    for layer in out:
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"层 {layer} 越界，模型共 {n_layers} 层")
    return out


def pretty(token: str) -> str:
    return token.replace("Ġ", "␣").replace("▁", "␣")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help="默认为首句（带尾空格，21 token）；传 'story' 用完整三句渔夫故事")
    ap.add_argument("--no-tail-space", action="store_true",
                    help="旧版 20 token 导出：末位是标点，不是预测槽")
    ap.add_argument("--model", default=None, help="本地模型目录或 HF repo id")
    ap.add_argument("--out", default=str(RESULTS_DIR / "attn.npz"))
    ap.add_argument(
        "--layers", default="all",
        help="要导出的层：all(默认) / -1 / 0,4,8,16 / 2:10 / -3,-1",
    )
    ap.add_argument(
        "--hidden", action="store_true",
        help="同时导出 hidden_states（含 embedding，共 n_layers+1 个）",
    )
    ap.add_argument(
        "--hidden-layers", default=None,
        help="只导出部分 hidden_states 层，语法同 --layers；默认全部",
    )
    args = ap.parse_args()

    if args.prompt == "story":
        prompt = STORY_PROMPT
    elif args.no_tail_space:
        prompt = DEFAULT_PROMPT.rstrip()
    else:
        prompt = args.prompt

    tokenizer, model = load_model(eager=True, model=args.model)

    n_model_layers = model.config.num_hidden_layers

    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    toks = [pretty(t) for t in tokenizer.convert_ids_to_tokens(ids)]

    with torch.no_grad():
        out = model(
            input_ids=torch.tensor([ids], device=model.device),
            attention_mask=torch.ones(1, len(ids), device=model.device),
            output_attentions=True,
            output_hidden_states=args.hidden,
        )

    layers = parse_layers(args.layers, n_model_layers)
    attn = torch.stack(
        [out.attentions[layer][0] for layer in layers], dim=0
    ).float().cpu().numpy()

    payload = {
        "attn": attn,
        "layer_indices": np.array(layers, dtype=np.int32),
        "ids": np.array(ids),
        "tokens": np.array(toks, dtype=object),
        "prompt": prompt,
        "num_model_layers": np.array(n_model_layers),
    }

    if args.hidden:
        hs = out.hidden_states  # tuple, 长度 n_layers+1
        if args.hidden_layers:
            hidden_layers = parse_layers(args.hidden_layers, len(hs))
        else:
            hidden_layers = list(range(len(hs)))
        payload["hidden_states"] = torch.stack(
            [hs[layer][0] for layer in hidden_layers], dim=0
        ).float().cpu().numpy()
        payload["hidden_layer_indices"] = np.array(hidden_layers, dtype=np.int32)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **payload)

    print(f"✓ {args.out}")
    print(f"  model layers : {n_model_layers}")
    print(f"  attn shape   : {attn.shape}  (selected, heads, seq, seq)")
    print(f"  layers       : {layers}")
    print(f"  row_sum≈     : {attn.mean(0).sum(-1).mean():.3f}")
    if args.hidden:
        print(f"  hidden shape : {payload['hidden_states'].shape}")
        print(f"  hidden layers: {hidden_layers}")
    print("  tokens       :", toks)


if __name__ == "__main__":
    main()
