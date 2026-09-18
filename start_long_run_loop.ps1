# start_long_run_loop.ps1
# TabulaShogi - HalfKP 自律改善ロングランループ起動スクリプト

$logFile = "loop_long_run.log"
$bestModel = "models/best_halfkp.bin"
$dataset = "data/loop_dataset.tsv"

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  TabulaShogi HalfKP Autonomous Self-Improvement Loop" -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "設定:" -ForegroundColor Yellow
Write-Host "  世代数 (Iterations) : 40 世代"
Write-Host "  世代あたり対局数     : 1,000 局 (累計 40,000 局)"
Write-Host "  検定対局ペア数       : 20 ペア (40 局 SPRT検定)"
Write-Host "  探索深さ             : 3"
Write-Host "  ワーカースレッド数   : 6 (CPU発熱・ファン負荷抑制)"
Write-Host "  学習エポック数       : 3"
Write-Host "  学習率               : 0.001"
Write-Host "  バッチサイズ         : 1024"
Write-Host "  データセット保存先   : $dataset"
Write-Host "  最良モデル保存先     : $bestModel"
Write-Host "  ログ保存先           : $logFile"
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "自律ループを起動します..." -ForegroundColor Magenta

$cmd = ".\target\release\tabula-shogi.exe loop --iterations 40 --games 1000 --eval-pairs 20 --depth 3 --threads 6 --epochs 3 --lr 0.001 --batch-size 1024 --data $dataset --best $bestModel"

Invoke-Expression $cmd 2>&1 | Tee-Object -FilePath $logFile
