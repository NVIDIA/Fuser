# Run all nvFuser tutorial examples
# Usage: .\run_all_examples.ps1 [-SkipTMA]

param(
    [switch]$SkipTMA,
    [switch]$Help
)

if ($Help) {
    Write-Host "Usage: .\run_all_examples.ps1 [-SkipTMA]"
    Write-Host "  -SkipTMA    Skip TMA examples (requires Hopper GPU)"
    exit 0
}

$ErrorActionPreference = "Continue"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

function Run-Example {
    param(
        [string]$File,
        [string]$Description
    )
    
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Blue
    Write-Host "Running: $File" -ForegroundColor Blue
    Write-Host "$Description" -ForegroundColor Blue
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Blue
    
    $result = python $File
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ $File completed successfully`n" -ForegroundColor Green
        return $true
    } else {
        Write-Host "✗ $File failed`n" -ForegroundColor Red
        return $false
    }
}

$FailedTests = @()
$PassedTests = @()

# Basic Examples
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "BASIC EXAMPLES" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Yellow

if (Run-Example "01-memcpy.py" "Basic memory copy") { $PassedTests += "01-memcpy.py" } else { $FailedTests += "01-memcpy.py" }
if (Run-Example "02-memcpy_scheduled.py" "Scheduled memory copy") { $PassedTests += "02-memcpy_scheduled.py" } else { $FailedTests += "02-memcpy_scheduled.py" }
if (Run-Example "03-reduction.py" "Reduction operations") { $PassedTests += "03-reduction.py" } else { $FailedTests += "03-reduction.py" }
if (Run-Example "04-reduction_rfactor.py" "Rfactor reductions") { $PassedTests += "04-reduction_rfactor.py" } else { $FailedTests += "04-reduction_rfactor.py" }
if (Run-Example "05-reshape.py" "Reshape operations") { $PassedTests += "05-reshape.py" } else { $FailedTests += "05-reshape.py" }
if (Run-Example "06-id_model_reshape_analysis.py" "IdModel analysis") { $PassedTests += "06-id_model_reshape_analysis.py" } else { $FailedTests += "06-id_model_reshape_analysis.py" }

# TMA Examples (Hopper GPU required)
if (-not $SkipTMA) {
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host "TMA EXAMPLES (Hopper GPU Required)" -ForegroundColor Yellow
    Write-Host "========================================`n" -ForegroundColor Yellow
    
    if (Run-Example "07-basic_tma_example1.py" "Single 1D TMA") { $PassedTests += "07-basic_tma_example1.py" } else { $FailedTests += "07-basic_tma_example1.py" }
    if (Run-Example "08-basic_tma_example2.py" "Multiple 1D TMA (for loop)") { $PassedTests += "08-basic_tma_example2.py" } else { $FailedTests += "08-basic_tma_example2.py" }
    if (Run-Example "09-basic_tma_example3.py" "Thread-parallelized 1D TMA") { $PassedTests += "09-basic_tma_example3.py" } else { $FailedTests += "09-basic_tma_example3.py" }
    if (Run-Example "10-basic_tma_example4.py" "TMA store") { $PassedTests += "10-basic_tma_example4.py" } else { $FailedTests += "10-basic_tma_example4.py" }
    if (Run-Example "11-basic_tma_example5.py" "2D TMA tiling") { $PassedTests += "11-basic_tma_example5.py" } else { $FailedTests += "11-basic_tma_example5.py" }
    if (Run-Example "12-basic_tma_example6.py" "2D TMA store") { $PassedTests += "12-basic_tma_example6.py" } else { $FailedTests += "12-basic_tma_example6.py" }
    if (Run-Example "13-vectorize_store_pointwise_tma.py" "Vectorized pointwise") { $PassedTests += "13-vectorize_store_pointwise_tma.py" } else { $FailedTests += "13-vectorize_store_pointwise_tma.py" }
    if (Run-Example "14-pointwise_broadcast_tma.py" "Broadcasting with TMA") { $PassedTests += "14-pointwise_broadcast_tma.py" } else { $FailedTests += "14-pointwise_broadcast_tma.py" }
    if (Run-Example "15-tma_bank_conflict_free_transpose.py" "Bank-conflict-free transpose") { $PassedTests += "15-tma_bank_conflict_free_transpose.py" } else { $FailedTests += "15-tma_bank_conflict_free_transpose.py" }
} else {
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host "TMA EXAMPLES (SKIPPED)" -ForegroundColor Yellow
    Write-Host "========================================`n" -ForegroundColor Yellow
    Write-Host "TMA examples skipped (use without -SkipTMA to run)`n" -ForegroundColor Yellow
}

# Auto Scheduler Examples
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "AUTO SCHEDULER EXAMPLES" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Yellow

if (Run-Example "16-pointwise_auto_scheduler.py" "Pointwise auto scheduler") { $PassedTests += "16-pointwise_auto_scheduler.py" } else { $FailedTests += "16-pointwise_auto_scheduler.py" }
if (Run-Example "17-reduction_auto_scheduler.py" "Reduction auto scheduler") { $PassedTests += "17-reduction_auto_scheduler.py" } else { $FailedTests += "17-reduction_auto_scheduler.py" }
if (Run-Example "18-inner_persistent_auto_scheduler.py" "Inner persistent auto scheduler") { $PassedTests += "18-inner_persistent_auto_scheduler.py" } else { $FailedTests += "18-inner_persistent_auto_scheduler.py" }

# Summary
Write-Host "`n========================================" -ForegroundColor Yellow
Write-Host "SUMMARY" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "Passed: $($PassedTests.Count)" -ForegroundColor Green
Write-Host "Failed: $($FailedTests.Count)" -ForegroundColor Red

if ($FailedTests.Count -gt 0) {
    Write-Host "`nFailed tests:" -ForegroundColor Red
    foreach ($test in $FailedTests) {
        Write-Host "  ✗ $test" -ForegroundColor Red
    }
    exit 1
} else {
    Write-Host "`nAll tests passed! 🎉" -ForegroundColor Green
    exit 0
}

