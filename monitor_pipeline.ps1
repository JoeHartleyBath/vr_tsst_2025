#!/usr/bin/env pwsh
# Pipeline monitoring script - periodically checks latest log file
# Usage: ./monitor_pipeline.ps1

$logDir = "output/logs"
Write-Host "Pipeline Monitor Started - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host "=============================================================" -ForegroundColor Green
Write-Host ""

while ($true) {
    try {
        # Find latest log file
        $latestLog = Get-ChildItem -Path $logDir -Filter "pipeline_*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        
        if ($latestLog) {
            Write-Host "`n[$(Get-Date -Format 'HH:mm:ss')] Latest log: $($latestLog.Name)" -ForegroundColor Cyan
            
            # Get last 20 lines of log
            $lastLines = Get-Content -Path $latestLog.FullName -Tail 20 | Where-Object { $_ -match "STAGE|Running|completed|ERROR|failed" }
            
            if ($lastLines) {
                foreach ($line in $lastLines) {
                    if ($line -match "ERROR|failed") {
                        Write-Host $line -ForegroundColor Red
                    } elseif ($line -match "completed successfully") {
                        Write-Host $line -ForegroundColor Green
                    } elseif ($line -match "STAGE") {
                        Write-Host $line -ForegroundColor Yellow
                    } else {
                        Write-Host $line
                    }
                }
            }
            
            # Check if pipeline completed
            $content = Get-Content -Path $latestLog.FullName -Raw
            if ($content -match "PIPELINE COMPLETE") {
                Write-Host "`n==============================================================" -ForegroundColor Green
                Write-Host "✓ PIPELINE COMPLETED SUCCESSFULLY!" -ForegroundColor Green
                Write-Host "==============================================================" -ForegroundColor Green
                break
            } elseif ($content -match "PIPELINE FAILED") {
                Write-Host "`n==============================================================" -ForegroundColor Red
                Write-Host "✗ PIPELINE FAILED - Check log for details" -ForegroundColor Red
                Write-Host "==============================================================" -ForegroundColor Red
                break
            }
        } else {
            Write-Host "No log files found in $logDir" -ForegroundColor Yellow
        }
        
        Write-Host "`n[Sleeping 60s... Press Ctrl+C to stop monitoring]" -ForegroundColor DarkGray
        Start-Sleep -Seconds 60
        
    } catch {
        Write-Host "Error: $_" -ForegroundColor Red
        break
    }
}

Write-Host "`nMonitor stopped at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
