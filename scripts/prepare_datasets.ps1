param(
    [switch]$AcceptLicenses,
    [switch]$SkipDownload,
    [switch]$ForceDownload,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if ($Python) {
    $pythonParts = $Python -split ' '
    $pythonCommand = $pythonParts[0]
    $pythonPrefix = @($pythonParts | Select-Object -Skip 1)
} elseif (Test-Path -LiteralPath $venvPython -PathType Leaf) {
    $pythonCommand = $venvPython
    $pythonPrefix = @()
} else {
    $pythonCommand = "py"
    $pythonPrefix = @("-3.11")
}

function Invoke-PythonScript {
    param([Parameter(Mandatory = $true)][string]$Script)
    Write-Host "Running $Script"
    & $pythonCommand @pythonPrefix (Join-Path $projectRoot $Script) --root $projectRoot
    if ($LASTEXITCODE -ne 0) { throw "$Script failed with exit code $LASTEXITCODE" }
}

Push-Location $projectRoot
try {
    if (-not $SkipDownload) {
        $downloadArgs = @(
            "-ExecutionPolicy", "Bypass", "-File", (Join-Path $PSScriptRoot "download_datasets.ps1"),
            "-Python", (($pythonCommand, $pythonPrefix) -join " ")
        )
        if ($AcceptLicenses) { $downloadArgs += "-AcceptLicenses" }
        if ($ForceDownload) { $downloadArgs += "-Force" }
        & powershell @downloadArgs
        if ($LASTEXITCODE -ne 0) { throw "Dataset download failed with exit code $LASTEXITCODE" }
    }

    Invoke-PythonScript "scripts\audit_datasets.py"
    Invoke-PythonScript "scripts\build_feasibility_manifest.py"
    Invoke-PythonScript "scripts\materialize_feasibility_set.py"
    Invoke-PythonScript "scripts\build_feasibility_task_views.py"
    Invoke-PythonScript "scripts\build_cubit_registry.py"
    Invoke-PythonScript "scripts\audit_cubit_duplicates.py"
    Invoke-PythonScript "scripts\build_cubit_exact_dedup_manifest.py"
    Invoke-PythonScript "scripts\build_codebrim_registry.py"
    Invoke-PythonScript "scripts\build_v1_manifest.py"
    Invoke-PythonScript "scripts\build_v1_training_inputs.py"

    Write-Host "Dataset preparation completed."
    Write-Host "Raw sources: $projectRoot\datasets"
    Write-Host "Frozen manifests: $projectRoot\data\manifests"
    Write-Host "Generated training inputs: $projectRoot\data\v1\training_inputs"
} finally {
    Pop-Location
}

