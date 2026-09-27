<#
.SYNOPSIS
    Starts the Business Entity Resolution interactive web review application.
.DESCRIPTION
    Runs the embedded Python HTTP server at http://localhost:8080/ and opens the default browser.
#>

[CmdletBinding()]
param (
    [int]$Port = 8080
)

$ErrorActionPreference = "Stop"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "         LAUNCHING BER PRO INTERACTIVE WEB REVIEW DASHBOARD                     " -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "`nServer URL: http://localhost:$Port/`n" -ForegroundColor Green

$PythonExe = (Get-Command py, python | Select-Object -First 1).Source

Start-Process "http://localhost:$Port/"
& $PythonExe web/server.py $Port
