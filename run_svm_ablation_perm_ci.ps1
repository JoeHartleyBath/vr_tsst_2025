param(
  [Parameter(Mandatory = $false)]
  [string]$RunTag,

  [Parameter(Mandatory = $false)]
  [string]$RscriptPath,

  [Parameter(Mandatory = $false)]
  [int]$PermP = 1000,

  [Parameter(Mandatory = $false)]
  [int]$K = 5,

  [Parameter(Mandatory = $false)]
  [int]$BootR = 1000,

  [Parameter(Mandatory = $false)]
  [string[]]$Domains = @("all", "eeg", "peripheral"),

  [Parameter(Mandatory = $false)]
  [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

function Get-TimestampTag {
  return (Get-Date -Format "yyyyMMdd_HHmmss")
}

if ([string]::IsNullOrWhiteSpace($RunTag)) {
  $RunTag = "rerun_$(Get-TimestampTag)"
}

$runRoot = Join-Path "results/svm_ablation_runs" $RunTag

if (Test-Path $runRoot) {
  if (-not $Overwrite) {
    throw "Refusing to overwrite existing run directory: $runRoot`nChoose a new -RunTag or pass -Overwrite (will rename existing dir to a backup)."
  }

  $backup = "$($runRoot)_backup_$(Get-TimestampTag)"
  Write-Host "[Safety] Renaming existing run dir to: $backup"
  Rename-Item -Path $runRoot -NewName (Split-Path $backup -Leaf)
}

New-Item -ItemType Directory -Path $runRoot -Force | Out-Null

$rscript = $null

if (-not [string]::IsNullOrWhiteSpace($RscriptPath)) {
  if (-not (Test-Path $RscriptPath)) {
    throw "RscriptPath does not exist: $RscriptPath"
  }
  $rscript = $RscriptPath
} else {
  $cmd = Get-Command Rscript -ErrorAction SilentlyContinue
  if ($null -ne $cmd) {
    $rscript = $cmd.Source
  } else {
    $fallback = "C:\Program Files\R\R-4.5.2\bin\Rscript.exe"
    if (Test-Path $fallback) {
      $rscript = $fallback
    } else {
      throw "Could not find Rscript. Install R or pass -RscriptPath to Rscript.exe."
    }
  }
}

$rscript = (Resolve-Path $rscript).Path
$obsScript = "pipelines/08_r_svm/svm_ablation_observed.R"
$infScript = "pipelines/08_r_svm/svm_inference_ablation.R"

if (-not (Test-Path $obsScript)) { throw "Missing observed script: $obsScript" }
if (-not (Test-Path $infScript)) { throw "Missing inference script: $infScript" }

Write-Host "[Run] RunTag=$RunTag PermP=$PermP K=$K BootR=$BootR"
Write-Host "[Run] Output root: $runRoot"

foreach ($domain in $Domains) {
  $domainDir = Join-Path $runRoot $domain
  New-Item -ItemType Directory -Path $domainDir -Force | Out-Null

  $logObs = Join-Path $domainDir "svm_observed.log"
  $logInf = Join-Path $domainDir "svm_inference.log"

  # Hard safety gate: these logs should not already exist in a fresh run dir.
  if ((Test-Path $logObs) -or (Test-Path $logInf)) {
    throw "Refusing to overwrite logs in: $domainDir"
  }

  $env:SVM_RUN_DIR = $domainDir
  $env:SVM_RUN_TAG = $RunTag
  $env:SVM_DOMAIN = $domain
  $env:SVM_K = "$K"
  $env:SVM_PERM_P = "$PermP"
  $env:SVM_BOOT_R = "$BootR"
  $env:SVM_OVERWRITE = "0"

  if ($domain -eq "peripheral") {
    # Peripheral has very few features; single-core is more stable than PSOCK fanout.
    $env:SVM_CORES = "1"
  } else {
    Remove-Item Env:\SVM_CORES -ErrorAction SilentlyContinue
  }

  Write-Host "[Run] Domain=$domain (observed)"
  $prevEap = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  & $rscript $obsScript 2>&1 | Tee-Object -FilePath $logObs
  $ErrorActionPreference = $prevEap
  if ($LASTEXITCODE -ne 0) {
    throw "Observed run failed for domain=$domain (exit=$LASTEXITCODE). See: $logObs"
  }

  Write-Host "[Run] Domain=$domain (inference)"
  $prevEap = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  & $rscript $infScript 2>&1 | Tee-Object -FilePath $logInf
  $ErrorActionPreference = $prevEap
  if ($LASTEXITCODE -ne 0) {
    throw "Inference run failed for domain=$domain (exit=$LASTEXITCODE). See: $logInf"
  }
}

Write-Host "[Done] Outputs written to: $runRoot"
