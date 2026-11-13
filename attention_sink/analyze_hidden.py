"""Hidden State 演化 + Logit Lens。

读取 extract_attn.py --hidden 导出的 hidden_states，追踪目标词：
  1. 每层向量与参考位置（默认 21-token 导出的末位空格）向量的余弦相似度曲线；
  2. Logit Lens：各层向量过 norm + lm_head，看该层把它“理解成”什么；
  3. 参考位置向量与所有 token 最终向量的相似度 Top10。

注意：hidden_states[-1] 是 model.norm 之后的 post-norm 向量，
hidden_states[l] 是第 l 层输入的 pre-norm 向量，两者约定不同；
本脚本只把它们当“向量像不像”看。

如果沿用旧的 20-token hidden 导出（--no-tail-space），末位是标点而不是
预测槽，脚本会提示；要做“预测位置”比较请用默认的 21-token 导出。

用法:
  python attention_sink/extract_attn.py --hidden --out attention_sink/results/attn_hidden.npz
  python attention_sink/analyze_hidden.py
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import torch

from common import load_model, RESULTS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(RESULTS_DIR / "attn_hidden.npz"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "hidden_evolution.png"))
    ap.add_argument("--target", default="␣village", help="要追踪的目标 token")
    ap.add_argument("--model", default=None, help="本地模型目录或 HF repo id")
    args = ap.parse_args()

    tokenizer, model = load_model(eager=False, model=args.model)

    d = np.load(args.npz, allow_pickle=True)
    hidden_states = d["hidden_states"]
    tokens = d["tokens"].tolist()
    layer_indices = d["hidden_layer_indices"]

    sorted_idx = np.argsort(layer_indices)
    hidden_states = hidden_states[sorted_idx]
    layer_indices = layer_indices[sorted_idx]

    print(f"✓ Hidden states shape: {hidden_states.shape}")

    target_word = args.target
    try:
        target_pos = tokens.index(target_word)
    except ValueError:
        fallback = target_word.lstrip("␣▁")
        print(f"⚠ 找不到 {target_word}，尝试 {fallback} ...")
        target_word = fallback
        target_pos = tokens.index(target_word)

    last_pos = len(tokens) - 1
    last_token = tokens[last_pos]
    is_prediction = last_token in ("␣", "▁")
    ref_label = "预测位置" if is_prediction else "末位 token"
    if not is_prediction:
        print("⚠ 末位不是空格 token：旧 20-token 导出在这里是标点，不是预测槽。"
              "要做预测位置比较，请用默认 21-token 导出。")
    print(f"✓ 追踪目标: {target_word!r} (位置 {target_pos}) -> "
          f"参考{ref_label} {last_token!r} (位置 {last_pos})")

    final_last_vec = hidden_states[-1, last_pos, :]
    cos_sims = []
    for layer in range(hidden_states.shape[0]):
        target_vec = hidden_states[layer, target_pos, :]
        sim = np.dot(target_vec, final_last_vec) / (
            np.linalg.norm(target_vec) * np.linalg.norm(final_last_vec) + 1e-8
        )
        cos_sims.append(sim)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(layer_indices, cos_sims, marker="o", color="crimson", linewidth=2)
    ref_en = "Prediction-Slot" if is_prediction else "Last-Token"
    ax1.set_title(f"Cosine Similarity: {target_word!r} vs {ref_en} Vector", fontsize=12)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Cosine Similarity")
    ax1.grid(True, alpha=0.3)

    device = model.device
    W = model.lm_head.weight.detach().float().to(device)

    preferred = [0, 2, 4, 8, 16, 27]
    available = layer_indices.tolist()
    show_idx = [i for i, layer in enumerate(available) if layer in preferred]
    if not show_idx:
        show_idx = list(range(0, len(available), max(1, len(available) // 6)))

    ax2.axis("off")
    text_lines = [f"Logit Lens: {target_word!r}\n----------------------------"]

    print(f"\n=== Logit Lens: {target_word!r} 在每层的 Top-3 映射 ===")
    for i in show_idx:
        layer = int(layer_indices[i])
        vec = torch.tensor(hidden_states[i, target_pos, :], dtype=torch.float32, device=device)
        if hasattr(model.model, "norm"):
            vec = model.model.norm(vec.unsqueeze(0)).squeeze(0)

        logits = torch.matmul(vec, W.T)
        topk = torch.topk(logits, 3)
        top_tokens = [tokenizer.decode([idx.item()]).strip() for idx in topk.indices]
        line = f"L{layer:02d}: {top_tokens}"
        text_lines.append(line)
        print(line)

    ax2.text(0.05, 0.5, "\n".join(text_lines), fontsize=12, va="center", family="monospace")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"\n✓ 已保存 {args.out}")

    print(f"\n=== 参考{ref_label}最终向量与各个 Token 向量的相似度 ===")
    final_vec = hidden_states[-1, last_pos, :]
    sim_results = []
    for i, token in enumerate(tokens):
        if i == last_pos:
            continue
        vec = hidden_states[-1, i, :]
        sim = np.dot(vec, final_vec) / (
            np.linalg.norm(vec) * np.linalg.norm(final_vec) + 1e-8
        )
        sim_results.append((token, sim))

    sim_results.sort(key=lambda x: x[1], reverse=True)
    for token, sim in sim_results[:10]:
        print(f"  {token:>10}: {sim:.4f}")


if __name__ == "__main__":
    main()
