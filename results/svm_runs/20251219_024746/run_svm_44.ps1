param(
  [string]$RunTag = "20251219_024746"
)

$ErrorActionPreference = 'Stop'

$repoRoot = "C:\vr_tsst_2025"
$runRoot  = Join-Path $repoRoot ("results\svm_runs\" + $RunTag)
$svmOut   = Join-Path $runRoot "svm"
$logPath  = Join-Path $runRoot "svm_run.log"

New-Item -ItemType Directory -Force -Path $svmOut | Out-Null

$env:SVM_OUT_DIR   = $svmOut
$env:SVM_EXPECT_N  = "44"
$env:SVM_OVERWRITE = "0"

Set-Location $repoRoot

# Run Rscript and redirect all PowerShell streams to a log file.
$rscript = "C:\Program Files\R\R-4.5.2\bin\Rscript.exe"
$svmScript = "pipelines\08_r_svm\svm.R"

$oldEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $rscript $svmScript *> $logPath
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $oldEap

if ($exitCode -ne 0) {
  throw "SVM run failed with exit code $exitCode. See log: $logPath"
}
