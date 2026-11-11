$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
$ctnetLogDir = Join-Path $PSScriptRoot 'logs_CTNet\validation_2'
$conformerLogDir = Join-Path $PSScriptRoot 'logs_Conformer\validation_2'
$statusPath = Join-Path $repoRoot '.git\validation_2_runner.status.txt'

New-Item -ItemType Directory -Force -Path $ctnetLogDir, $conformerLogDir | Out-Null
Set-Location -LiteralPath $repoRoot

# Make CUDA unavailable to the child processes. The model scripts already use
# CPU tensors, but this guard prevents accidental GPU use if they change later.
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

function Write-RunnerStatus([string]$message) {
    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss K'
    [System.IO.File]::WriteAllText(
        $statusPath,
        "$timestamp - $message`r`n",
        (New-Object System.Text.UTF8Encoding($false))
    )
}

function Convert-LogToUtf8([string]$path) {
    if (Test-Path -LiteralPath $path) {
        $lines = Get-Content -LiteralPath $path
        [System.IO.File]::WriteAllLines(
            $path,
            $lines,
            (New-Object System.Text.UTF8Encoding($false))
        )
    }
}

Add-Type @'
using System;
using System.Runtime.InteropServices;

public static class ExecutionState {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@

$esContinuous = [uint32]2147483648
$esSystemRequired = [uint32]0x00000001
[void][ExecutionState]::SetThreadExecutionState($esContinuous -bor $esSystemRequired)

try {
    $cudaAvailable = & $python -c "import torch; print(torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0 -or $cudaAvailable -ne 'False') {
        throw "CPU-only preflight failed (torch.cuda.is_available()=$cudaAvailable)"
    }

    $ctnetLog = Join-Path $ctnetLogDir 'run1.txt'
    Write-RunnerStatus 'running CTNet validation_2'
    $savedErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $python -u 'reproduced_model_trainings\CTNet.py' --validation-version validation_2 2>&1 |
        Tee-Object -FilePath $ctnetLog
    $ctnetExitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorActionPreference
    Convert-LogToUtf8 $ctnetLog
    if ($ctnetExitCode -ne 0) {
        throw "CTNet exited with code $ctnetExitCode"
    }

    $conformerLog = Join-Path $conformerLogDir 'run1.txt'
    Write-RunnerStatus 'CTNet complete; running Conformer validation_2'
    $ErrorActionPreference = 'Continue'
    & $python -u 'reproduced_model_trainings\Conformer.py' --validation-version validation_2 2>&1 |
        Tee-Object -FilePath $conformerLog
    $conformerExitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorActionPreference
    Convert-LogToUtf8 $conformerLog
    if ($conformerExitCode -ne 0) {
        throw "Conformer exited with code $conformerExitCode"
    }

    Write-RunnerStatus 'complete'
}
catch {
    Write-RunnerStatus ('failed: ' + $_.Exception.Message)
    throw
}
finally {
    [void][ExecutionState]::SetThreadExecutionState($esContinuous)
}
