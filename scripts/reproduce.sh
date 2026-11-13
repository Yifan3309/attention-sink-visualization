#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
RESULTS="attention_sink/results"
mkdir -p "$RESULTS"

run() {
    echo ">>> $PY $*"
    "$PY" "$@"
}

run scripts/preflight.py
run attention_sink/extract_attn.py --layers all --out "$RESULTS/attn.npz"
run attention_sink/find_heads.py
run attention_sink/view.py single "$RESULTS/attn.npz" "$RESULTS/L2_H6_worker.png" --layer 2 --head 6
run attention_sink/view.py single "$RESULTS/attn.npz" "$RESULTS/L8_H4_sink.png" --layer 8 --head 4
run attention_sink/view.py single "$RESULTS/attn.npz" "$RESULTS/L15_H7_full.png" --layer 15 --head 7
run attention_sink/view.py sweep "$RESULTS/attn.npz" "$RESULTS/sweep_last.png" --query -1
run attention_sink/view.py layer-animate "$RESULTS/attn.npz" "$RESULTS/along_layers.gif" --query -1
run attention_sink/extract_attn.py --layers all --hidden --out "$RESULTS/attn_hidden.npz"
run attention_sink/analyze_hidden.py
run attention_sink/long_trace.py --max-new 512
run attention_sink/analyze_long.py
run attention_sink/sublayer_writes.py
run attention_sink/window_ppl.py --max-tokens 2048 --window 512
run attention_sink/sink_evict.py --max-tokens 1024 --evict-at 512
run attention_sink/secret_word.py

echo "全部完成。产物在 $RESULTS"
