$ErrorActionPreference = "Stop"
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Results = "attention_sink/results"
New-Item -ItemType Directory -Force -Path $Results | Out-Null

function Invoke-Step {
    param([string[]]$CmdArgs)
    Write-Host ">>> $Python $($CmdArgs -join ' ')"
    & $Python @CmdArgs
    if ($LASTEXITCODE -ne 0) {
        throw "命令失败 (exit $LASTEXITCODE): $Python $($CmdArgs -join ' ')"
    }
}

Invoke-Step @("scripts/preflight.py")
Invoke-Step @("attention_sink/extract_attn.py", "--layers", "all", "--out", "$Results/attn.npz")
Invoke-Step @("attention_sink/find_heads.py")
Invoke-Step @("attention_sink/view.py", "single", "$Results/attn.npz", "$Results/L2_H6_worker.png", "--layer", "2", "--head", "6")
Invoke-Step @("attention_sink/view.py", "single", "$Results/attn.npz", "$Results/L8_H4_sink.png", "--layer", "8", "--head", "4")
Invoke-Step @("attention_sink/view.py", "single", "$Results/attn.npz", "$Results/L15_H7_full.png", "--layer", "15", "--head", "7")
Invoke-Step @("attention_sink/view.py", "sweep", "$Results/attn.npz", "$Results/sweep_last.png", "--query", "-1")
Invoke-Step @("attention_sink/view.py", "layer-animate", "$Results/attn.npz", "$Results/along_layers.gif", "--query", "-1")
Invoke-Step @("attention_sink/extract_attn.py", "--layers", "all", "--hidden", "--out", "$Results/attn_hidden.npz")
Invoke-Step @("attention_sink/analyze_hidden.py")
Invoke-Step @("attention_sink/long_trace.py", "--max-new", "512")
Invoke-Step @("attention_sink/analyze_long.py")
Invoke-Step @("attention_sink/sublayer_writes.py")
Invoke-Step @("attention_sink/window_ppl.py", "--max-tokens", "2048", "--window", "512")
Invoke-Step @("attention_sink/sink_evict.py", "--max-tokens", "1024", "--evict-at", "512")
Invoke-Step @("attention_sink/secret_word.py")

Write-Host "全部完成。产物在 $Results"
