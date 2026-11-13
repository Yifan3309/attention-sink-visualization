"""attention_sink 公共模块：模型路径解析、权重定位、模型加载。

模型路径优先级：
  1. 调用方显式传入的 model（通常是 --model）
  2. 环境变量 MODEL_PATH
  3. 仓库根目录 .env 中的 path_model
  4. 默认 Hugging Face repo id: Qwen/Qwen3-0.6B

本地目录和 HF repo id 都支持。权重文件既支持单文件 model.safetensors，
也支持 model.safetensors.index.json 分片。
"""
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

# Windows 控制台默认 GBK，直接打印 "␣"、"✓" 会 UnicodeEncodeError。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _read_dotenv() -> dict[str, str]:
    """手动解析仓库根目录 .env，不依赖 python-dotenv。"""
    env_file = ROOT_DIR / ".env"
    values: dict[str, str] = {}
    if not env_file.exists():
        return values
    for line in env_file.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _looks_like_local_path(value: str) -> bool:
    return (
        "\\" in value
        or value.startswith((".", "~", "/"))
        or (len(value) > 1 and value[1] == ":")
    )


def load_model_path(model: str | None = None) -> str:
    """返回本地模型目录或 HF repo id。"""
    explicit = model or os.getenv("MODEL_PATH") or _read_dotenv().get("path_model")
    if not explicit:
        return DEFAULT_MODEL

    candidate = str(Path(explicit).expanduser())
    if Path(candidate).is_dir():
        return candidate
    if Path(candidate).exists():
        raise RuntimeError(f"模型路径不是目录: {candidate}")
    if _looks_like_local_path(explicit):
        raise RuntimeError(
            f"模型目录不存在: {candidate}\n"
            "请设置 MODEL_PATH、在仓库根目录 .env 写入 path_model=...，"
            "或直接使用默认的 Qwen/Qwen3-0.6B。"
        )
    if explicit.count("/") != 1:
        raise RuntimeError(f"无法识别的模型标识: {explicit}")
    return explicit


def _shard_map(model_dir: Path) -> dict[str, str]:
    index = model_dir / "model.safetensors.index.json"
    if not index.exists():
        raise RuntimeError(f"目录中没有 model.safetensors 或分片 index: {model_dir}")
    data = json.loads(index.read_text(encoding="utf-8"))
    return data.get("weight_map", {})


def resolve_weight_files(model_path: str, keys: list[str]) -> dict[str, Path]:
    """把参数名映射到实际 safetensors 文件，兼容本地/HF、单文件/分片。"""
    root = Path(model_path)
    if root.is_dir():
        single = root / "model.safetensors"
        if single.exists():
            return {key: single for key in keys}
        weight_map = _shard_map(root)
        missing = [key for key in keys if key not in weight_map]
        if missing:
            raise RuntimeError(f"权重文件中缺少参数: {missing}")
        return {key: root / weight_map[key] for key in keys}

    from huggingface_hub import hf_hub_download

    try:
        single = Path(hf_hub_download(model_path, "model.safetensors"))
        return {key: single for key in keys}
    except Exception:
        index = Path(hf_hub_download(model_path, "model.safetensors.index.json"))
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        missing = [key for key in keys if key not in weight_map]
        if missing:
            raise RuntimeError(f"权重文件中缺少参数: {missing}")
        return {
            key: Path(hf_hub_download(model_path, weight_map[key])) for key in keys
        }


def load_model(eager: bool = True, model: str | None = None):
    """加载 (tokenizer, model)。eager=True 才能拿到 output_attentions。"""
    model_path = load_model_path(model)
    print("模型目录:", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    loaded = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
        attn_implementation="eager" if eager else None,
    ).eval()
    return tokenizer, loaded


def sliced_cache(cache, lo, hi, n_layers):
    """返回删除 KV 位置 [lo, hi) 的新 DynamicCache（驱逐实验用）。
    用 update() 重建而非直接改内部结构，兼容新旧 transformers 的 cache 实现。"""
    from transformers.cache_utils import DynamicCache

    new = DynamicCache()
    for layer in range(n_layers):
        if hasattr(cache, "layers"):           # transformers >= 4.54
            k, v = cache.layers[layer].keys, cache.layers[layer].values
        else:                                  # 旧版
            k, v = cache.key_cache[layer], cache.value_cache[layer]
        k2 = torch.cat([k[:, :, :lo], k[:, :, hi:]], dim=2)
        v2 = torch.cat([v[:, :, :lo], v[:, :, hi:]], dim=2)
        new.update(k2, v2, layer)
    return new
