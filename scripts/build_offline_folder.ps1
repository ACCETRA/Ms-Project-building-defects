param(
    [string]$Output = "C:\Users\Z B O O K\OneDrive\Documents\LAST OF USE\Project-Alpha-FYP-Complete"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$outputPath = [IO.Path]::GetFullPath($Output)
$outputParent = Split-Path -Parent $outputPath
$manifestRelative = "artifacts\offline_bundle_manifest.sha256"

if ($outputPath -eq $projectRoot -or $outputPath.StartsWith($projectRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Output must be outside the project workspace: $outputPath"
}
if (Test-Path -LiteralPath $outputPath) {
    throw "Output folder already exists: $outputPath"
}

$includeRoots = @(
    ".gitattributes", ".gitignore", "README.md", "START_HERE.md", "PROJECT_READINESS.md",
    "requirements-audit.txt", "requirements-runtime.txt", "bdi", "config", "confirmation", "docs",
    "examples", "harness", "schemas", "scripts", "tests", "data\manifests", "weights", "vendor",
    "datasets\README.md", "datasets\license.md", "datasets\CODEBRIM\README.md", "datasets\CODEBRIM\license.md",
    "datasets\DACL10K\README.md", "datasets\building-target\README.md", "datasets\building-target\S2DS\README.md",
    "datasets\building-target\S2DS\LICENSE", "datasets\CiF-tiled\README.md", "datasets\UAV-candidates\README.md",
    "artifacts", "runs\detect\runs\comparison\yolo\yolo11n_detect_v1_queue",
    "runs\segment\runs\comparison\yolo\yolo11n_seg_v1_queue",
    "runs\comparison\resnet50_v1_queue", "runs\comparison\yolo_sam\v1-sample-10",
    "runs\evaluation", "runs\florence\v1-validation-sample-10", "runs\sam\decoder_v1_fp32_smoke", "runs\v1_queue"
)

$excludedDocs = @(
    "docs\DECISION_LOG.md",
    "docs\ROADMAP_V2.md",
    "docs\COMPLETION_PLAN.md",
    "docs\CURATED_DATASET_V0_1_PROPOSAL.md",
    "docs\FIRST_WEEK_PLAN.md",
    "docs\EXPERIMENT_MATRIX.md",
    "docs\SAM_DECODER_TRAINING.md"
)

$files = foreach ($relative in $includeRoots) {
    $path = Join-Path $projectRoot $relative
    if (Test-Path -LiteralPath $path -PathType Leaf) {
        Get-Item -LiteralPath $path
    } elseif (Test-Path -LiteralPath $path -PathType Container) {
        Get-ChildItem -LiteralPath $path -Recurse -File -Force
    } else {
        throw "Handoff input is missing: $relative"
    }
}

$files = $files | Where-Object {
    $relative = $_.FullName.Substring($projectRoot.Length).TrimStart('\', '/')
    $relative -ne $manifestRelative -and
    $relative -notin $excludedDocs -and
    $relative -notmatch '^artifacts[\\/].*\.zip$' -and
    $relative -notmatch '[\\/]__pycache__[\\/]' -and
    $relative -notmatch '[\\/]\.cache[\\/]' -and
    $_.Name -notmatch '\.tmp$'
} | Sort-Object FullName -Unique

New-Item -ItemType Directory -Path $outputParent -Force | Out-Null
New-Item -ItemType Directory -Path $outputPath | Out-Null

foreach ($file in $files) {
    $relative = $file.FullName.Substring($projectRoot.Length).TrimStart('\', '/')
    $destination = Join-Path $outputPath $relative
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Copy-Item -LiteralPath $file.FullName -Destination $destination
}

$manifestPath = Join-Path $outputPath $manifestRelative
New-Item -ItemType Directory -Path (Split-Path -Parent $manifestPath) -Force | Out-Null
$manifestLines = Get-ChildItem -LiteralPath $outputPath -Recurse -File -Force |
    Where-Object { $_.FullName -ne $manifestPath } |
    Sort-Object FullName |
    ForEach-Object {
        $relative = $_.FullName.Substring($outputPath.Length).TrimStart('\', '/').Replace('\', '/')
        $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        "$hash  $relative"
    }
$manifestLines | Set-Content -LiteralPath $manifestPath -Encoding utf8

$bytes = (Get-ChildItem -LiteralPath $outputPath -Recurse -File -Force | Measure-Object Length -Sum).Sum
Write-Host "Created $outputPath"
Write-Host ("Files: {0:N0}" -f ((Get-ChildItem -LiteralPath $outputPath -Recurse -File -Force).Count))
Write-Host ("Size: {0:N2} GiB" -f ($bytes / 1GB))
