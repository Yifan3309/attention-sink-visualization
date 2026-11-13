"""路径 / 数据 / 依赖自检。

默认只做静态检查，不加载权重、不联网：
  python scripts/preflight.py
  python scripts/preflight.py --check-tokenizer
  python scripts/preflight.py --model /path/to/Qwen3-0.6B
"""
import argparse
import importlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "attention_sink"))

from common import DEFAULT_MODEL, RESULTS_DIR, load_model_path  # noqa: E402

PROMPT_BASE = "Once upon a time, in a small village by the sea, there lived an old fisherman."
STORY_PROMPT = (
    PROMPT_BASE + " Every morning he rowed his boat out before sunrise, "
    "and every evening he came back with a story to tell. One day, he found "
    "something strange tangled in his net:"
)
LOCAL_MARKERS = "|".join([
    "LLM_IN" + "_ollama",
    "/" + "home/[^/\\s]+",
    "/" + "Users/[^/\\s]+",
    "len" + "ovo",
])
SECRET_PATTERNS = {
    "本地路径/用户名": re.compile(LOCAL_MARKERS, re.I),
    "API key": re.compile(r"sk-[A-Za-z0-9]{16,}|hf_[A-Za-z0-9]{20,}"),
    "密钥字段": re.compile(r"(api[_-]?key|password)\s*[=:]\s*\S+", re.I),
}


def check(condition, message, failures):
    print(("✓ " if condition else "✗ ") + message)
    if not condition:
        failures.append(message)


def scan_text_files():
    suffixes = {".py", ".md", ".txt", ".sh", ".ps1", ".json", ".toml", ".yaml", ".yml"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if "results" in path.parts or path.name in {"pg1342.txt", "preflight.py"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name, pattern in SECRET_PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                yield f"{path.relative_to(ROOT)}:{line}: {name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="本地模型目录或 HF repo id")
    ap.add_argument("--check-tokenizer", action="store_true",
                    help="加载 tokenizer/config，并核对序列长度与模型结构")
    args = ap.parse_args()

    failures = []
    print(f"仓库根目录: {ROOT}")

    print("\n[1/4] 文件与数据")
    data_file = ROOT / "attention_sink" / "pg1342.txt"
    check(data_file.is_file(), f"公版文本存在: {data_file}", failures)
    if data_file.is_file():
        check(data_file.stat().st_size > 100_000,
              f"pg1342.txt 大小: {data_file.stat().st_size:,} bytes", failures)
    for rel in ["requirements.txt", ".env.example", "attention_sink/common.py"]:
        check((ROOT / rel).is_file(), f"存在 {rel}", failures)

    print("\n[2/4] 模型路径")
    try:
        model_path = load_model_path(args.model)
        print(f"✓ 模型解析结果: {model_path}")
        if Path(model_path).is_dir():
            check((Path(model_path) / "config.json").is_file(),
                  "本地模型目录包含 config.json", failures)
        else:
            print(f"  （HF repo id，将在首次运行时下载；默认 {DEFAULT_MODEL}）")
    except Exception as exc:
        check(False, f"模型路径解析失败: {exc}", failures)

    print("\n[3/4] 输出目录")
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        probe = RESULTS_DIR / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        check(True, f"输出目录可写: {RESULTS_DIR}", failures)
    except Exception as exc:
        check(False, f"输出目录不可写: {exc}", failures)

    print("\n[4/4] 依赖 / 脱敏 / tokenizer")
    for module in ["numpy", "torch", "transformers", "matplotlib", "safetensors"]:
        try:
            loaded = importlib.import_module(module)
            print(f"✓ {module} {getattr(loaded, '__version__', '?')}")
        except Exception as exc:
            check(False, f"{module} 导入失败: {exc}", failures)

    leaks = list(scan_text_files())
    check(not leaks, "未发现本地绝对路径 / 密钥残留", failures)
    for leak in leaks:
        print(f"  ! {leak}")

    if args.check_tokenizer:
        try:
            from transformers import AutoConfig, AutoTokenizer

            model_path = load_model_path(args.model)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            config = AutoConfig.from_pretrained(model_path)
            lengths = {
                "20 token（不带尾空格）": len(tokenizer(PROMPT_BASE, add_special_tokens=False).input_ids),
                "21 token（带尾空格）": len(tokenizer(PROMPT_BASE + " ", add_special_tokens=False).input_ids),
                "55 token（完整故事）": len(tokenizer(STORY_PROMPT, add_special_tokens=False).input_ids),
            }
            expected = {"20 token（不带尾空格）": 20, "21 token（带尾空格）": 21,
                        "55 token（完整故事）": 55}
            for name, value in lengths.items():
                check(value == expected[name], f"{name}: 实际 {value}", failures)
            check(config.num_hidden_layers == 28,
                  f"num_hidden_layers={config.num_hidden_layers}（期望 28）", failures)
            check(config.num_attention_heads == 16,
                  f"num_attention_heads={config.num_attention_heads}（期望 16）", failures)
            check(config.hidden_size == 1024,
                  f"hidden_size={config.hidden_size}（期望 1024）", failures)
        except Exception as exc:
            check(False, f"tokenizer/config 检查失败: {exc}", failures)

    print()
    if failures:
        print(f"自检失败：{len(failures)} 项")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("自检通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
