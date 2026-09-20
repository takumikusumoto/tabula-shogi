# scripts/run_loop.ps1
# TabulaShogi - HalfKP 自律改善ロングランループ起動スクリプト
[CmdletBinding()]
param(
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

$dataDir = Join-Path $repoRoot "data"
$modelsDir = Join-Path $repoRoot "models"

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
if (-not (Test-Path $exe)) {
    Write-Host "リリースバイナリが見つかりません。ビルドを実行します..." -ForegroundColor Yellow
    cargo build --release --locked
    if ($LASTEXITCODE -ne 0) {
        Write-Error "ビルドに失敗しました。"
        exit 1
    }
}

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  TabulaShogi HalfKP Autonomous Self-Improvement Loop" -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "設定:" -ForegroundColor Yellow
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

& $exe $argsList 2>&1 | Tee-Object -FilePath $runLog
if ($LASTEXITCODE -ne 0) {
    Write-Error "自律ループが異常終了しました (Exit code: $LASTEXITCODE)。"
    exit $LASTEXITCODE
}
Write-Host "自律ループセッションが正常に完了しました。" -ForegroundColor Green

