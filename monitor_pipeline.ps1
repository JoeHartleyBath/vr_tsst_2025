<# VR-TSST Pipeline Monitor: processes, logs, outputs #>
$ErrorActionPreference = 'SilentlyContinue'

Write-Host "`n=== VR-TSST Pipeline Monitor ===" -ForegroundColor Cyan
Write-Host ("Timestamp: {0}" -f (Get-Date))

# Active processes
Write-Host "`n--- Active Processes (python/matlab/R) ---" -ForegroundColor Yellow
Get-Process -Name python, python3, matlab, R | Select-Object Name, Id, CPU, StartTime | Format-Table -AutoSize

# Latest orchestrator log
Write-Host "`n--- Latest Orchestrator Log (tail 60) ---" -ForegroundColor Yellow
$logs = Get-ChildItem -Path "output/logs" -Filter "pipeline_*.log" | Sort-Object LastWriteTime -Descending
if ($logs -and $logs.Count -gt 0) {
    $latest = $logs[0].FullName
    Write-Host ("Log: {0}" -f $latest)
    Get-Content -Path $latest -Tail 60
} else {
    Write-Host "No pipeline logs found yet."
}

# Stage outputs
Write-Host "`n--- Stage Outputs ---" -ForegroundColor Yellow
Write-Host "Stage 1 (.set files):"
Get-ChildItem -Path "output/sets" -Filter "*.set" | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize

Write-Host "`nStage 2 (cleaned EEG):"
Get-ChildItem -Path "output/cleaned_eeg" -Filter "*.set" | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize

Write-Host "`nQC Summary:"
Get-ChildItem -Path "output/qc/summary" -Filter "*" | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize

# Guidance
Write-Host "`nTip: Auto-refresh every 10s with:" -ForegroundColor DarkCyan
Write-Host "powershell -NoLogo -NoProfile -Command \"while ($true) { cls; & .\\monitor_pipeline.ps1; Start-Sleep -Seconds 10 }\""
