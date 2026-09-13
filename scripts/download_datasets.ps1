param(
    [string[]]$Dataset = @("all"),
    [switch]$AcceptLicenses,
    [switch]$Force,
    [string]$Python = "py -3.11"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$parts = $Python -split ' '
$command = $parts[0]
$baseArgs = @($parts | Select-Object -Skip 1)

& $command @baseArgs -c "import gdown, huggingface_hub" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing download-only helpers gdown and huggingface_hub..."
    & $command @baseArgs -m pip install "gdown==5.2.0" "huggingface-hub==1.30.0"
    if ($LASTEXITCODE -ne 0) { throw "Could not install dataset download helpers" }
}

$arguments = @((Join-Path $projectRoot "scripts\download_datasets.py"), "--root", $projectRoot)
foreach ($name in $Dataset) { $arguments += @("--dataset", $name) }
if ($AcceptLicenses) { $arguments += "--accept-licenses" }
if ($Force) { $arguments += "--force" }

& $command @baseArgs @arguments
if ($LASTEXITCODE -ne 0) { throw "Dataset acquisition failed" }
