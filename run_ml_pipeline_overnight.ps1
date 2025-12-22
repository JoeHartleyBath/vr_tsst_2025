# VR-TSST ML Pipeline - Sequential SVM + XGBoost with Memory Cleanup
# =====================================================================

$ErrorActionPreference = "Continue"
$startTime = Get-Date

Write-Host "`n=== VR-TSST ML Pipeline - Overnight Execution ===" -ForegroundColor Cyan
Write-Host "Start time: $startTime" -ForegroundColor Gray
Write-Host "System RAM: $([math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB, 2)) GB" -ForegroundColor Gray
Write-Host "CPU cores: $([System.Environment]::ProcessorCount)" -ForegroundColor Gray
Write-Host "Estimated total runtime: ~4-6 hours`n" -ForegroundColor Gray

# =============================================================================
# STAGE 1: SVM LOSO Classification
# =============================================================================
Write-Host "`n[1/2] SVM LOSO Classification" -ForegroundColor Yellow
Write-Host "===========================================" -ForegroundColor Yellow
Write-Host "Expected runtime: ~2-3 hours" -ForegroundColor Gray
Write-Host "Processing: 47 participants x 3 k-values x 2 targets`n" -ForegroundColor Gray

$svmStart = Get-Date

& "C:\Program Files\R\R-4.5.2\bin\Rscript.exe" pipelines/08_r_svm/svm.R
$svmExitCode = $LASTEXITCODE

$svmEnd = Get-Date
$svmDuration = ($svmEnd - $svmStart).ToString("hh\:mm\:ss")

if ($svmExitCode -eq 0) {
    Write-Host "`nSVM completed successfully in $svmDuration" -ForegroundColor Green
    
    # Verify outputs
    if (Test-Path "results/svm/svm_progress_stress_label.csv") {
        $stressRows = (Get-Content "results/svm/svm_progress_stress_label.csv" | Measure-Object -Line).Lines - 1
        Write-Host "  Stress: $stressRows folds completed" -ForegroundColor Green
    }
    if (Test-Path "results/svm/svm_progress_workload_label.csv") {
        $workloadRows = (Get-Content "results/svm/svm_progress_workload_label.csv" | Measure-Object -Line).Lines - 1
        Write-Host "  Workload: $workloadRows folds completed" -ForegroundColor Green
    }
} else {
    Write-Host "`nSVM exited with code $svmExitCode in $svmDuration" -ForegroundColor Red
    Write-Host "Continuing to XGBoost anyway..." -ForegroundColor Yellow
}

# =============================================================================
# STAGE 2: XGBoost LOSO Classification
# =============================================================================
Write-Host "`n[2/2] XGBoost LOSO Classification" -ForegroundColor Yellow
Write-Host "===========================================" -ForegroundColor Yellow
Write-Host "Expected runtime: ~2-3 hours" -ForegroundColor Gray
Write-Host "Processing: 47 participants x Bayesian hyperparameter tuning`n" -ForegroundColor Gray

$xgbStart = Get-Date

& "C:\Program Files\R\R-4.5.2\bin\Rscript.exe" pipelines/09_r_xgboost/xgboost_loso_classification.R
$xgbExitCode = $LASTEXITCODE

$xgbEnd = Get-Date
$xgbDuration = ($xgbEnd - $xgbStart).ToString("hh\:mm\:ss")

if ($xgbExitCode -eq 0) {
    Write-Host "`nXGBoost completed successfully in $xgbDuration" -ForegroundColor Green
    
    # Verify outputs
    if (Test-Path "results/xgboost/summary_LOSO_fold_tuning.csv") {
        Write-Host "  Results saved to results/xgboost/" -ForegroundColor Green
    }
} else {
    Write-Host "`nXGBoost exited with code $xgbExitCode in $xgbDuration" -ForegroundColor Red
}

# =============================================================================
# SUMMARY
# =============================================================================
$endTime = Get-Date
$totalDuration = ($endTime - $startTime).ToString("hh\:mm\:ss")

Write-Host "`n=== Pipeline Complete ===" -ForegroundColor Cyan
Write-Host "Start time:  $startTime" -ForegroundColor Gray
Write-Host "End time:    $endTime" -ForegroundColor Gray
Write-Host "Total duration: $totalDuration`n" -ForegroundColor Gray

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Run model finders to identify best hyperparameters:" -ForegroundColor Gray
Write-Host "     Rscript pipelines/08_r_svm/svm_best_model_finder.R" -ForegroundColor Gray
Write-Host "     Rscript pipelines/09_r_xgboost/xgboost_best_model_finder.R" -ForegroundColor Gray
Write-Host "  2. Compare results with archived pre-fix data`n" -ForegroundColor Gray
