param(
    [string]$MainPython,
    [string]$TtsPython,
    [string]$BindAddress = '127.0.0.1',
    [int]$Port = 8000
)
$ErrorActionPreference = 'Stop'
$appRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $MainPython -or -not (Test-Path -LiteralPath $MainPython -PathType Leaf)) {
    throw 'Provide -MainPython with the full path to the main environment python.exe.'
}
if (-not $TtsPython -or -not (Test-Path -LiteralPath $TtsPython -PathType Leaf)) {
    throw 'Provide -TtsPython with the full path to the isolated Qwen3-TTS python.exe.'
}
$logDir = Join-Path $appRoot 'outputs\services'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$ttsScript = Join-Path $PSScriptRoot 'serve_local_tts.py'
$ttsModel = Join-Path $appRoot 'models\Qwen3-TTS-12Hz-1.7B-CustomVoice'
try {
    $health = Invoke-RestMethod -Uri 'http://127.0.0.1:9880/health' -TimeoutSec 2
} catch { $health = $null }
if (-not $health.ready) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $arguments = '"{0}" --model-dir "{1}" --device cuda:0' -f $ttsScript, $ttsModel
    $process = Start-Process -FilePath $TtsPython -ArgumentList $arguments -WorkingDirectory $appRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDir "tts_$stamp.out.log") -RedirectStandardError (Join-Path $logDir "tts_$stamp.err.log")
    Write-Output "Started TTS process $($process.Id). Logs: $logDir"
    Write-Output 'Wait for TTS /health to become ready before sending audio; the first model load can take time.'
}
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollama) { throw 'Ollama is not installed or is missing from PATH. Install it and pull qwen3.5:9b first.' }
try { $null = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 2 }
catch {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $process = Start-Process -FilePath $ollama.Source -ArgumentList @('serve') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDir "ollama_$stamp.out.log") -RedirectStandardError (Join-Path $logDir "ollama_$stamp.err.log")
    Write-Output "Started Ollama process $($process.Id)."
}
& $MainPython (Join-Path $PSScriptRoot 'run_demo.py') serve --host $BindAddress --port $Port
exit $LASTEXITCODE
