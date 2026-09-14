param(
    [string]$Output = "artifacts\Project-Alpha-FYP-offline.zip"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$outputPath = if ([IO.Path]::IsPathRooted($Output)) { $Output } else { Join-Path $projectRoot $Output }
$outputDirectory = Split-Path -Parent $outputPath
$manifest = Join-Path $projectRoot "artifacts\offline_bundle_manifest.sha256"
$listFile = Join-Path $env:TEMP "project-alpha-offline-files.txt"
$sevenZip = Join-Path $projectRoot "vendor\7zip-portable\x64\7za.exe"

$includeRoots = @(
    ".gitattributes", ".gitignore", "README.md", "START_HERE.md", "PROJECT_READINESS.md",
    "requirements-audit.txt", "requirements-runtime.txt", "bdi", "config", "confirmation", "docs",
    "examples", "harness", "schemas", "scripts", "source", "tests", "data\manifests", "artifacts", "weights", "vendor",
    "datasets\README.md", "datasets\license.md", "datasets\CODEBRIM\README.md", "datasets\CODEBRIM\license.md",
    "datasets\DACL10K\README.md", "datasets\building-target\README.md", "datasets\building-target\S2DS\README.md",
    "datasets\building-target\S2DS\LICENSE", "datasets\CiF-tiled\README.md", "datasets\UAV-candidates\README.md",
    "runs\detect\runs\comparison\yolo\yolo11n_detect_v1_queue",
    "runs\segment\runs\comparison\yolo\yolo11n_seg_v1_queue",
    "runs\comparison\resnet50_v1_queue", "runs\comparison\yolo_sam\v1-sample-10",
    "runs\evaluation", "runs\florence\v1-validation-sample-10", "runs\sam\decoder_v1_fp32_smoke", "runs\v1_queue"
)

$files = foreach ($relative in $includeRoots) {
    $path = Join-Path $projectRoot $relative
    if (Test-Path -LiteralPath $path -PathType Leaf) { Get-Item -LiteralPath $path }
    elseif (Test-Path -LiteralPath $path -PathType Container) { Get-ChildItem -LiteralPath $path -Recurse -File -Force }
    else { throw "Bundle input is missing: $relative" }
}
$files = $files | Where-Object {
    $_.FullName -ne $outputPath -and
    $_.FullName -ne $manifest -and
    $_.FullName -notmatch '[\\/]__pycache__[\\/]' -and
    $_.FullName -notmatch '[\\/]\.cache[\\/]' -and
    $_.Name -notmatch '\.tmp$'
} | Sort-Object FullName -Unique

$manifestLines = foreach ($file in $files) {
    $relative = if ($file.FullName.StartsWith($projectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        $file.FullName.Substring($projectRoot.Length).TrimStart('\', '/')
    } else {
        $file.FullName
    }
    $relativeSlash = $relative.Replace('\', '/')
    $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $relativeSlash"
}
$manifestLines | Set-Content -LiteralPath $manifest -Encoding utf8
$files += Get-Item -LiteralPath $manifest
$relativeFiles = $files | ForEach-Object {
    if ($_.FullName.StartsWith($projectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        $_.FullName.Substring($projectRoot.Length).TrimStart('\', '/')
    } else {
        $_.FullName
    }
}
$relativeFiles | Set-Content -LiteralPath $listFile -Encoding utf8

if (-not (Test-Path -LiteralPath $sevenZip -PathType Leaf)) { throw "7-Zip is missing: $sevenZip" }
if (Test-Path -LiteralPath $outputPath) { Remove-Item -LiteralPath $outputPath -Force }
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
Push-Location $projectRoot
try {
    & $sevenZip a -tzip -mx=1 -mmt=on $outputPath "@$listFile"
    if ($LASTEXITCODE -ne 0) { throw "7-Zip failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
    if (Test-Path -LiteralPath $listFile) { Remove-Item -LiteralPath $listFile -Force }
}

$archiveHash = (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash
$archive = Get-Item -LiteralPath $outputPath
Write-Host "Created $outputPath"
Write-Host ("Size: {0:N2} GiB" -f ($archive.Length / 1GB))
Write-Host "SHA-256: $archiveHash"
