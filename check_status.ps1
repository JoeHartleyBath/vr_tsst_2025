#!/usr/bin/env pwsh
# Quick non-interrupting pipeline status check
# Usage: ./check_status.ps1

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "Pipeline Status Check - $timestamp" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if pipeline process is running
$pythonProcesses = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "*venv*" }
if ($pythonProcesses) {
    Write-Host "✓ Pipeline process is RUNNING (PID: $($pythonProcesses.Id -join ', '))" -ForegroundColor Green
    Write-Host "  Runtime: $([math]::Round(($pythonProcesses[0].CPU), 1)) seconds CPU time" -ForegroundColor Gray
} else {
    Write-Host "✗ No pipeline process detected" -ForegroundColor Red
}
Write-Host ""

# Check latest log file
$logDir = "output/logs"
$latestLog = Get-ChildItem -Path $logDir -Filter "pipeline_*.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if ($latestLog) {
    Write-Host "Latest log file: $($latestLog.Name)" -ForegroundColor Yellow
    Write-Host "Last modified: $($latestLog.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Gray
    Write-Host "Size: $([math]::Round($latestLog.Length / 1KB, 1)) KB" -ForegroundColor Gray
    Write-Host ""
    
    # Extract key status info
    $content = Get-Content -Path $latestLog.FullName -Raw
    
    # Find current stage
    $stageMatches = [regex]::Matches($content, "STAGE (\d): (.+)")
    if ($stageMatches.Count -gt 0) {
        $lastStage = $stageMatches[$stageMatches.Count - 1]
        Write-Host "Current/Last Stage: $($lastStage.Groups[1].Value) - $($lastStage.Groups[2].Value)" -ForegroundColor Cyan
    }
    
    # Count completed stages
    $completedCount = ([regex]::Matches($content, "Stage \d completed successfully")).Count
    Write-Host "Completed stages: $completedCount / 6" -ForegroundColor Green
    
    # Check for errors
    $errorCount = ([regex]::Matches($content, "ERROR|failed with")).Count
    if ($errorCount -gt 0) {
        Write-Host "⚠ Errors detected: $errorCount" -ForegroundColor Red
    }
    
    Write-Host ""
    Write-Host "Last 10 significant lines:" -ForegroundColor Yellow
    Write-Host "------------------------------" -ForegroundColor DarkGray
    $lastLines = Get-Content -Path $latestLog.FullName -Tail 30 | Where-Object { 
        $_ -match "STAGE|Running|completed|ERROR|failed|Processing|Conversion" 
    } | Select-Object -Last 10
    
    foreach ($line in $lastLines) {
        if ($line -match "ERROR|failed") {
            Write-Host $line -ForegroundColor Red
        } elseif ($line -match "completed successfully") {
            Write-Host $line -ForegroundColor Green
        } else {
            Write-Host $line -ForegroundColor Gray
        }
    }
}

Write-Host ""
Write-Host "------------------------------" -ForegroundColor DarkGray
Write-Host "Output Files Status:" -ForegroundColor Yellow

# Check for stage outputs
$outputs = @(
    @{Stage=1; Path="output/sets/*.set"; Desc="SET files (Stage 1)"},
    @{Stage=2; Path="output/cleaned_eeg/*.set"; Desc="Cleaned EEG (Stage 2)"},
    @{Stage=3; Path="output/aggregated/eeg_features.csv"; Desc="EEG features (Stage 3)"},
    @{Stage=4; Path="output/*.csv"; Desc="Physio features (Stage 4)"},
    @{Stage=5; Path="output/final_data.csv"; Desc="Merged data (Stage 5)"},
    @{Stage=6; Path="output/final_data.rds"; Desc="R preprocessed (Stage 6)"}
)

foreach ($output in $outputs) {
    $files = Get-ChildItem -Path $output.Path -ErrorAction SilentlyContinue
    if ($files) {
        Write-Host "  ✓ Stage $($output.Stage): $($output.Desc) - $($files.Count) file(s)" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Stage $($output.Stage): $($output.Desc) - not found" -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host "==============================================================" -ForegroundColor Cyan
