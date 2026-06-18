param(
  [Parameter(Mandatory = $false)]
  [string]$RunTag,

  [Parameter(Mandatory = $false)]
  [string]$RscriptPath,

  [Parameter(Mandatory = $false)]
  [int]$PermP = 1000,

  [Parameter(Mandatory = $false)]
  [int]$BootR = 1000,

  [Parameter(Mandatory = $false)]
  [int]$K = 5,

  [Parameter(Mandatory = $false)]
  [string[]]$Domains = @("all", "eeg", "peripheral"),

  [Parameter(Mandatory = $false)]
  [int]$Seed = 42,

  [Parameter(Mandatory = $false)]
  [double]$Alpha = 0.05,

  [Parameter(Mandatory = $false)]
  [string]$PythonExe,

  [Parameter(Mandatory = $false)]
  [switch]$SkipPythonSummary
)

$ErrorActionPreference = "Stop"

function Get-TimestampTag {
  return (Get-Date -Format "yyyyMMdd_HHmmss")
}

if ([string]::IsNullOrWhiteSpace($RunTag)) {
  $RunTag = "repro_$(Get-TimestampTag)"
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $repoRoot

try {
  $runner = Join-Path $repoRoot "run_svm_ablation_perm_ci.ps1"
  if (-not (Test-Path $runner)) {
    throw "Missing runner script: $runner"
  }

  $runRoot = Join-Path $repoRoot (Join-Path "results/svm_ablation_runs" $RunTag)

  Write-Host "[Repro] RunTag=$RunTag"
  Write-Host "[Repro] Domains=$($Domains -join ', ')"
  Write-Host "[Repro] K=$K PermP=$PermP BootR=$BootR"

  if (-not [string]::IsNullOrWhiteSpace($RscriptPath)) {
    & $runner -RunTag $RunTag -RscriptPath $RscriptPath -PermP $PermP -BootR $BootR -K $K -Domains $Domains
  } else {
    & $runner -RunTag $RunTag -PermP $PermP -BootR $BootR -K $K -Domains $Domains
  }
  if ($LASTEXITCODE -ne 0) {
    throw "Runner failed (exit=$LASTEXITCODE)"
  }

  Write-Host "[Repro] Finished R outputs: $runRoot"

  if (-not $SkipPythonSummary) {
    if ([string]::IsNullOrWhiteSpace($PythonExe)) {
      $venvPy = Join-Path $repoRoot ".venv/Scripts/python.exe"
      if (Test-Path $venvPy) {
        $PythonExe = $venvPy
      } else {
        $PythonExe = "python"
      }
    }

    $pyScript = Join-Path $repoRoot "tools/svm_perm_metrics_from_predictions.py"
    if (-not (Test-Path $pyScript)) {
      throw "Missing python script: $pyScript"
    }

    $outDir = Join-Path $repoRoot "docs/repro_artifacts"
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null

    $outCsv = Join-Path $outDir "svm_perm_metrics_seed${Seed}_P${PermP}_${RunTag}.csv"

    Write-Host "[Repro] Writing python summary: $outCsv"

    & $PythonExe $pyScript --run-dir $runRoot --P $PermP --seed $Seed --alpha $Alpha --out $outCsv
    if ($LASTEXITCODE -ne 0) {
      throw "Python summary failed (exit=$LASTEXITCODE)"
    }
  }

  Write-Host "[Done] End-to-end reproduction complete."
}
finally {
  Pop-Location
}
