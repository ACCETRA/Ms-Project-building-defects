$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$manifest = Join-Path $projectRoot "artifacts\offline_bundle_manifest.sha256"
if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
    throw "Missing integrity manifest: $manifest"
}

$checked = 0
foreach ($line in Get-Content -LiteralPath $manifest) {
    if (-not $line.Trim()) { continue }
    if ($line -notmatch '^([0-9a-fA-F]{64})  (.+)$') { throw "Invalid manifest line: $line" }
    $expected = $Matches[1].ToUpperInvariant()
    $relative = $Matches[2] -replace '/', '\'
    $path = Join-Path $projectRoot $relative
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing bundle file: $relative" }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    if ($actual -ne $expected) { throw "SHA-256 mismatch: $relative" }
    $checked += 1
}
Write-Host "Verified $checked files from the offline bundle manifest."
