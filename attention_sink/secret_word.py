"""前置实验（← 原 config.py）：注意力即检索 —— 近因 vs 先行

prompt 里先后声明两个 "secret word"（X= red, Y= blue），
在生成位置（prompt 末尾 "The secret word is" 的下一个 token）考察：
  1. logit(X) - logit(Y)：模型最终"相信"哪个声明；
  2. 生成位置对两处声明的 per-token attention 份额（L0/8/16/24/27）；
  3. 基准：只放一条声明 / 完全无声明（区分检索效应与词先验）。

用法: python attention_sink/secret_word.py [--model 本地目录或HF id]
"""
import argparse

import torch

from common import load_model

# 手动选一对带空格单 token、先验尽量均衡的词
WORD_X, WORD_Y = " red", " blue"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="本地模型目录或 HF repo id")
    args = ap.parse_args()
    tokenizer, model = load_model(model=args.model)

    # ---------- 0. 确认带空格的词是单 token ----------
    def tok(w):
        return tokenizer(w, add_special_tokens=False).input_ids

    print("=== tokenization 检查（带前导空格）===")
    for w in [" cat", " dog", " red", " blue", " sun", " moon", " apple"]:
        print(f"{w!r:>8}: {tok(w)}  (len={len(tok(w))})")
    print()

    assert len(tok(WORD_X)) == 1 and len(tok(WORD_Y)) == 1, "换一对单 token 词"
    X_ID, Y_ID = tok(WORD_X)[0], tok(WORD_Y)[0]

    # ---------- 1. 构造 prompt：X 声明在前 / Y 声明在前 ----------
    def build_prompt(first, second):
        return (f"The secret word is{first}. The sky is green. "
                f"The secret word is{second}. The secret word is")

    prompts = {
        "X→Y": build_prompt(WORD_X, WORD_Y),
        "Y→X": build_prompt(WORD_Y, WORD_X),
    }

    def locate(ids, word_ids):
        """在 ids 里找到 word_ids 出现的所有位置"""
        out = []
        n = len(word_ids)
        for i in range(len(ids) - n + 1):
            if ids[i:i + n] == word_ids:
                out.append(i)
        return out

    print("=== logit 差：生成位置对 X 和 Y 的偏好 ===")
    print("正值 = 偏向 X，负值 = 偏向 Y")
    print(f"{'条件':<6} | {'X位置':>6} {'Y位置':>6} | {'logit(X)':>9} {'logit(Y)':>9} {'差':>7} | {'pref':>6}")
    print("-" * 75)

    records = []
    for name, text in prompts.items():
        ids = tokenizer(text, add_special_tokens=False).input_ids
        x_pos = locate(ids, tok(WORD_X))
        y_pos = locate(ids, tok(WORD_Y))
        with torch.no_grad():
            out = model(
                input_ids=torch.tensor([ids], device=model.device),
                attention_mask=torch.ones(1, len(ids), device=model.device),
            )
        logits = out.logits[0, -1]  # 生成位置
        lx, ly = logits[X_ID].item(), logits[Y_ID].item()
        diff = lx - ly
        pref = "X" if diff > 0 else "Y"
        records.append((name, ids, x_pos, y_pos, diff))
        print(f"{name:<6} | {x_pos[0] if x_pos else -1:>6} {y_pos[0] if y_pos else -1:>6} | "
              f"{lx:>9.3f} {ly:>9.3f} {diff:>+7.3f} | {pref:>6}")

    # ---------- 2. attention：生成位置对两处声明的 per-token 份额 ----------
    def attn_spans(ids, x_pos, y_pos, layers=(0, 8, 16, 24, 27)):
        with torch.no_grad():
            out = model(
                input_ids=torch.tensor([ids], device=model.device),
                attention_mask=torch.ones(1, len(ids), device=model.device),
                output_attentions=True,
            )
        last = -1
        res = {}
        for l in layers:
            if l >= len(out.attentions):
                continue
            a = out.attentions[l][0][:, last, :].mean(0)
            ax = a[x_pos[0]].item() if x_pos else 0.0
            ay = a[y_pos[0]].item() if y_pos else 0.0
            res[l] = (ax, ay)
        return res

    print(f"\n=== 生成位置对 X / Y 的 attention（单 token）===")
    print(f"{'条件':<6} | " + " | ".join(f"L{l:>2} X/Y" for l in (0, 8, 16, 24, 27)))
    print("-" * 72)
    for name, ids, x_pos, y_pos, _ in records:
        r = attn_spans(ids, x_pos, y_pos)
        cols = [f"{r[l][0]:.4f}/{r[l][1]:.4f}" if l in r else "-" for l in (0, 8, 16, 24, 27)]
        print(f"{name:<6} | " + " | ".join(f"{c:>10}" for c in cols))

    # ---------- 3. 基准：只放一条声明，看词本身的先验 ----------
    print("\n=== 基准：只放一条声明（测词先验）===")
    for tag, w in [("X only", WORD_X), ("Y only", WORD_Y)]:
        text = f"The secret word is{w}. The sky is green. The secret word is"
        ids = tokenizer(text, add_special_tokens=False).input_ids
        with torch.no_grad():
            out = model(
                input_ids=torch.tensor([ids], device=model.device),
                attention_mask=torch.ones(1, len(ids), device=model.device),
            )
        logits = out.logits[0, -1]
        lx, ly = logits[X_ID].item(), logits[Y_ID].item()
        print(f"{tag:<7} | logit(X)={lx:.3f}  logit(Y)={ly:.3f}  diff={lx - ly:+.3f}")

    # ---------- 4. 无声明基准：纯词频先验 ----------
    text = "The secret word is"
    ids = tokenizer(text, add_special_tokens=False).input_ids
    with torch.no_grad():
        out = model(
            input_ids=torch.tensor([ids], device=model.device),
            attention_mask=torch.ones(1, len(ids), device=model.device),
        )
    logits = out.logits[0, -1]
    print(f"\n无声明 | logit(X)={logits[X_ID].item():.3f}  logit(Y)={logits[Y_ID].item():.3f}  "
          f"diff={logits[X_ID].item() - logits[Y_ID].item():+.3f}")


if __name__ == "__main__":
    main()
