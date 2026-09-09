$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$environmentPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$torchWheel = Join-Path $projectRoot "vendor\wheels\torch-2.6.0+cu124-cp311-cp311-win_amd64.whl"
$visionWheel = Join-Path $projectRoot "vendor\wheels\torchvision-0.21.0+cu124-cp311-cp311-win_amd64.whl"

$expected = @{
    $torchWheel = @{
        Size = 2532350702
        Sha256 = "6A1FB2714E9323F11EDB6E8ABF7AAD5F79E45AD25C081CDE87681A18D99C29EB"
    }
    $visionWheel = @{
        Size = 6140993
        Sha256 = "000A013584AD2304AB30496318145F284AC364622ADDB5EE3A5ABD2769BA146F"
    }
}

foreach ($wheel in $expected.Keys) {
    if (-not (Test-Path -LiteralPath $wheel -PathType Leaf)) {
        throw "Required wheel is missing: $wheel"
    }
    $file = Get-Item -LiteralPath $wheel
    if ($file.Length -ne $expected[$wheel].Size) {
        throw "Incomplete wheel: $wheel ($($file.Length) of $($expected[$wheel].Size) bytes)"
    }
    $actualHash = (Get-FileHash -LiteralPath $wheel -Algorithm SHA256).Hash
    if ($actualHash -ne $expected[$wheel].Sha256) {
        throw "SHA256 mismatch for $wheel"
    }
}

if (-not (Test-Path -LiteralPath $environmentPython -PathType Leaf)) {
    & uv venv --python 3.11 (Join-Path $projectRoot ".venv")
    if ($LASTEXITCODE -ne 0) { throw "uv could not create the Python 3.11 environment" }
}

& uv pip install --python $environmentPython $torchWheel $visionWheel
if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed" }

& $environmentPython (Join-Path $projectRoot "scripts\verify_cuda.py") --root $projectRoot
if ($LASTEXITCODE -ne 0) { throw "CUDA verification failed" }
