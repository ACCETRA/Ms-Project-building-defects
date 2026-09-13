param(
    [string]$Python = "py -3.11"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venv = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venv "Scripts\python.exe"
$wheelhouse = Join-Path $projectRoot "vendor\wheels"
$torchWheel = Join-Path $wheelhouse "torch-2.6.0+cu124-cp311-cp311-win_amd64.whl"
$visionWheel = Join-Path $wheelhouse "torchvision-0.21.0+cu124-cp311-cp311-win_amd64.whl"
$requirements = Join-Path $projectRoot "requirements-runtime.txt"

foreach ($required in @($torchWheel, $visionWheel, $requirements)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required offline asset is missing: $required"
    }
}

$expected = @{
    $torchWheel = @{ Size = 2532350702; Sha256 = "6A1FB2714E9323F11EDB6E8ABF7AAD5F79E45AD25C081CDE87681A18D99C29EB" }
    $visionWheel = @{ Size = 6140993; Sha256 = "000A013584AD2304AB30496318145F284AC364622ADDB5EE3A5ABD2769BA146F" }
}
foreach ($wheel in $expected.Keys) {
    $item = Get-Item -LiteralPath $wheel
    if ($item.Length -ne $expected[$wheel].Size) { throw "Wheel size mismatch: $wheel" }
    if ((Get-FileHash -LiteralPath $wheel -Algorithm SHA256).Hash -ne $expected[$wheel].Sha256) {
        throw "Wheel SHA-256 mismatch: $wheel"
    }
}

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    $parts = $Python -split ' '
    $command = $parts[0]
    $arguments = @($parts | Select-Object -Skip 1) + @('-m', 'venv', $venv)
    & $command @arguments
    if ($LASTEXITCODE -ne 0) { throw "Python 3.11 could not create $venv" }
}

& $venvPython -m pip install --no-index --find-links $wheelhouse --no-deps $torchWheel $visionWheel
if ($LASTEXITCODE -ne 0) { throw "Offline PyTorch installation failed" }

& $venvPython -m pip install --no-index --find-links $wheelhouse -r $requirements
if ($LASTEXITCODE -ne 0) { throw "Offline runtime dependency installation failed" }

Push-Location $projectRoot
try {
    & $venvPython -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw "Project tests failed" }
    & $venvPython scripts\verify_model_assets.py --root $projectRoot
    if ($LASTEXITCODE -ne 0) { throw "Model asset verification failed" }
} finally {
    Pop-Location
}

Write-Host "Offline environment is ready."
Write-Host "Start the demo with: .\.venv\Scripts\python.exe harness\server.py"
