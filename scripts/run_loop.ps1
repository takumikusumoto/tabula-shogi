# scripts/run_loop.ps1
# TabulaShogi - HalfKP 自律改善ロングランループ起動スクリプト
[CmdletBinding()]
param(
    [string]$RunId = "",
    [int]$Iterations = 5,
    [int]$GamesPerIter = 1000,
    [int]$EvalPairs = 20,
    [int]$Depth = 2,
    [int]$Threads = 6,
    [int]$Epochs = 3,
    [double]$Lr = 0.001,
    [int]$BatchSize = 1024,
    [int]$MinGames = 20
)

$ErrorActionPreference = "Stop"

# リポジトリルートをスクリプト位置から自動解決
$repoRoot = (Resolve-Path "$PSScriptRoot/..").Path
Set-Location $repoRoot

if ($RunId -ne "") {
    $dataDir = Join-Path $repoRoot "data/$RunId"
    $modelsDir = Join-Path $repoRoot "models/$RunId"
} else {
    $dataDir = Join-Path $repoRoot "data"
    $modelsDir = Join-Path $repoRoot "models"
}

if (-not (Test-Path $dataDir)) {
    New-Item -ItemType Directory -Path $dataDir -Force | Out-Null
}
if (-not (Test-Path $modelsDir)) {
    New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null
}

$dataset = Join-Path $dataDir "loop_dataset.tsv"
$deepDataset = Join-Path $dataDir "deep_dataset.tsv"
$summaryLog = Join-Path $dataDir "loop_summary.csv"
$stateFile = Join-Path $dataDir "loop_state.txt"
$runLog = Join-Path $dataDir "loop_run.log"

$bestModel = Join-Path $modelsDir "best_halfkp.bin"
$candidateModel = Join-Path $modelsDir "candidate_halfkp.bin"
$candidateCkpt = Join-Path $modelsDir "candidate_halfkp_ckpt.bin"

$exe = Join-Path $repoRoot "target/release/tabula-shogi.exe"
Write-Host "リリースバイナリの最新状態を確認・ビルドします..." -ForegroundColor Yellow
cargo build --release --locked
if ($LASTEXITCODE -ne 0) {
    Write-Error "ビルドに失敗しました。"
    exit 1
}

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  TabulaShogi HalfKP Autonomous Self-Improvement Loop" -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "設定:" -ForegroundColor Yellow
if ($RunId -ne "") {
    Write-Host "  Run ID                    : $RunId"
}
Write-Host "  世代数 (Iterations)      : $Iterations 世代"
Write-Host "  世代あたり対局数          : $GamesPerIter 局"
Write-Host "  検定対局ペア数 (EvalPairs): $EvalPairs ペア ($([int]($EvalPairs * 2)) 局)"
Write-Host "  SPRT最低対局数 (MinGames) : $MinGames 局"
Write-Host "  探索深さ (Depth)          : $Depth"
Write-Host "  ワーカースレッド数 (Threads): $Threads"
Write-Host "  学習エポック数 (Epochs)   : $Epochs"
Write-Host "  学習率 (LR)               : $Lr"
Write-Host "  バッチサイズ (BatchSize)  : $BatchSize"
Write-Host "  データセット保存先        : $dataset"
Write-Host "  深読み蒸留プール保存先    : $deepDataset"
Write-Host "  最良モデル保存先          : $bestModel"
Write-Host "  候補モデル保存先          : $candidateModel"
Write-Host "  モメンタム保存先          : $candidateCkpt"
Write-Host "  サマリーログ保存先        : $summaryLog"
Write-Host "  世代状態ファイル          : $stateFile"
Write-Host "  実行ログ保存先            : $runLog"
Write-Host "  メモリ上限監視            : 500 MiB (超過時フェイルクローズ強制終了)"
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "自律ループを起動します..." -ForegroundColor Magenta

$argsList = @(
    "loop",
    "-i", "$Iterations",
    "-g", "$GamesPerIter",
    "-p", "$EvalPairs",
    "-t", "$Threads",
    "-d", "$Depth",
    "-e", "$Epochs",
    "--lr", "$Lr",
    "-b", "$BatchSize",
    "--min-games", "$MinGames",
    "--data", "$dataset",
    "--deep-data", "$deepDataset",
    "--best", "$bestModel",
    "--candidate", "$candidateModel",
    "--candidate-ckpt", "$candidateCkpt",
    "--summary", "$summaryLog",
    "--state", "$stateFile"
)

$psi = [System.Diagnostics.ProcessStartInfo]::new()
$psi.FileName = $exe
$psi.Arguments = ($argsList | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join ' '
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.CreateNoWindow = $true

$proc = [System.Diagnostics.Process]::new()
$proc.StartInfo = $psi

$logStream = [System.IO.StreamWriter]::new($runLog, $true, [System.Text.Encoding]::UTF8)

$outHandler = [System.Diagnostics.DataReceivedEventHandler]{
    param($sender, $e)
    if ($null -ne $e.Data) {
        [Console]::WriteLine($e.Data)
        $logStream.WriteLine($e.Data)
        $logStream.Flush()
    }
}
$proc.add_OutputDataReceived($outHandler)
$proc.add_ErrorDataReceived($outHandler)

$proc.Start() | Out-Null
$proc.BeginOutputReadLine()
$proc.BeginErrorReadLine()

$memoryLimitExceeded = $false
$maxMemoryBytes = 500 * 1024 * 1024

try {
    while (-not $proc.WaitForExit(1000)) {
        $proc.Refresh()
        try {
            $ws = $proc.WorkingSet64
            if ($ws -gt $maxMemoryBytes) {
                $memoryLimitExceeded = $true
                $wsMb = [math]::Round($ws / 1MB, 2)
                Write-Error "[FAIL-CLOSED] メモリ上限超過: ${wsMb} MB > 500 MB。プロセスを即時強制終了します。"
                $proc.Kill()
                break
            }
        } catch {
            break
        }
    }
} finally {
    $proc.WaitForExit()
    $logStream.Dispose()
}

if ($memoryLimitExceeded) {
    exit 1
}
if ($proc.ExitCode -ne 0) {
    Write-Error "自律ループが異常終了しました (Exit code: $($proc.ExitCode))。"
    exit $proc.ExitCode
}
Write-Host "自律ループセッションが正常に完了しました。" -ForegroundColor Green

