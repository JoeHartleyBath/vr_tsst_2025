# Stage raw data from participant folders into pipeline-aligned structure
# Copies (does not move) files from data/raw/<PN>/ to data/raw/eeg, metadata, subjective

$base = "$(Get-Location)\data\raw"
$eeg_dir = "$base\eeg"
$metadata_dir = "$base\metadata"
$subjective_dir = "$base\subjective"

# Create target directories
@($eeg_dir, $metadata_dir, $subjective_dir) | ForEach-Object {
    if (-not (Test-Path $_)) {
        New-Item -ItemType Directory -Path $_ -Force | Out-Null
        Write-Host "Created directory: $_" -ForegroundColor Green
    }
}

$success_count = 0
$skip_count = 0
$error_count = 0

# Process each participant folder (1-48)
1..48 | ForEach-Object {
    $pnum = $_
    $padded_id = "{0:D2}" -f $pnum
    $pdir = "$base\$pnum"
    
    if (-not (Test-Path $pdir)) {
        Write-Host "Participant folder not found: $pdir" -ForegroundColor Yellow
        $skip_count++
        return
    }
    
    $files = Get-ChildItem $pdir -File
    
    # Find XDF file
    $xdf_src = $files | Where-Object { $_.Extension -eq '.xdf' -and $_.Name -match '_N\.xdf$' } | Select-Object -First 1
    if ($xdf_src) {
        $xdf_dst = "$eeg_dir\P$padded_id.xdf"
        Copy-Item $xdf_src.FullName $xdf_dst -Force
        Write-Host "OK: XDF P$padded_id" -ForegroundColor Green
        $success_count++
    }
    
    # Find physio CSV (matches: PD_43_RAW_DATA_24-11-28-02-20-55_C.csv)
    # Pattern: PD_<number>_RAW_DATA_<timestamp>_C.csv
    $csv_src = $files | Where-Object { $_.Name -match '^PD_\d+_RAW_DATA_' -and $_.Name -match '_C\.csv$' } | Select-Object -First 1
    if ($csv_src) {
        $csv_dst = "$metadata_dir\P$padded_id.csv"
        Copy-Item $csv_src.FullName $csv_dst -Force
        Write-Host "OK: Metadata P$padded_id" -ForegroundColor Green
        $success_count++
    } else {
        $error_count++
    }
    
    # Find compiled PQ file
    $pq_src = $files | Where-Object { $_.Name -match '^PQs_' -and $_.Name -match '_compiled\.csv$' } | Select-Object -First 1
    if ($pq_src) {
        $pq_dst = "$subjective_dir\PQs_$padded_id`_compiled.csv"
        Copy-Item $pq_src.FullName $pq_dst -Force
        Write-Host "OK: PQ P$padded_id" -ForegroundColor Green
        $success_count++
    }
}

Write-Host "`n=== Summary ===" -ForegroundColor White
Write-Host "Total file copies: $success_count" -ForegroundColor Green
Write-Host "Metadata CSV errors: $error_count" -ForegroundColor Red
Write-Host "Done! Check data/raw/eeg, data/raw/metadata, data/raw/subjective" -ForegroundColor Green

