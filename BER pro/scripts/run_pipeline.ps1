<#
.SYNOPSIS
    PowerShell script to run the Business Entity Resolution Pipeline.
.DESCRIPTION
    Runs data ingestion, candidate generation, pairwise feature extraction,
    model inference, thresholding, and validation.
#>

[CmdletBinding()]
param (
    [string]$TrainDir = "../resources/dataset/train",
    [string]$TestDir = "../resources/dataset/test",
    [string]$OutputDir = "output",
    [string]$ModelDir = "artifacts",
    [string]$Mode = "all",
    [Nullable[int]]$SampleTrain = $null,
    [Nullable[int]]$SampleTest = $null,
    [int]$ChunkSize = 50000
)

$ErrorActionPreference = "Stop"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "             STARTING BUSINESS ENTITY RESOLUTION PIPELINE                      " -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$PythonExe = (Get-Command py, python | Select-Object -First 1).Source

$cmdArgs = @(
    "src/pipeline.py",
    "--train-dir", $TrainDir,
    "--test-dir", $TestDir,
    "--output-dir", $OutputDir,
    "--model-dir", $ModelDir,
    "--mode", $Mode,
    "--chunk-size", $ChunkSize
)

if ($SampleTrain) {
    $cmdArgs += @("--sample-train", $SampleTrain)
}
if ($SampleTest) {
    $cmdArgs += @("--sample-test", $SampleTest)
}

Write-Host "Executing: $PythonExe $($cmdArgs -join ' ')" -ForegroundColor Yellow
& $PythonExe @cmdArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nPipeline completed successfully with exit code 0!" -ForegroundColor Green
} else {
    Write-Host "`nPipeline failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
