# Fetch the llama.cpp Windows build the installer bundles.
#
# The Vulkan build is one artifact that covers both worlds: GPU inference
# wherever a Vulkan driver exists (NVIDIA, AMD, Intel), and runtime-dispatched
# CPU paths everywhere else. No per-vendor SKUs, no CUDA runtime to ship.
#
# The release tag is resolved at build time rather than pinned: llama.cpp cuts
# releases near-daily and a pinned tag would rot in a week. What IS pinned is
# the contract - the archive must contain llama-server.exe - and the script
# fails loudly when the asset naming changes, which is a build failure a
# person sees, not a broken installer a user sees. The resolved tag is written
# to build/llama-version.txt so every installer records what it shipped.

param(
    [string]$Tag = "latest",
    [string]$Pattern = "*bin-win-vulkan-x64.zip",
    [string]$OutDir = "build/llama"
)

$ErrorActionPreference = "Stop"

$api = if ($Tag -eq "latest") {
    "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
} else {
    "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/$Tag"
}

$headers = @{ "User-Agent" = "OpenKnowledge-build" }
if ($env:GITHUB_TOKEN) { $headers["Authorization"] = "Bearer $($env:GITHUB_TOKEN)" }

$release = Invoke-RestMethod -Uri $api -Headers $headers
$asset = $release.assets | Where-Object { $_.name -like $Pattern } | Select-Object -First 1
if (-not $asset) {
    $names = ($release.assets | ForEach-Object { $_.name }) -join "`n  "
    throw "No asset matches '$Pattern' in llama.cpp $($release.tag_name). Assets:`n  $names"
}

Write-Host "llama.cpp $($release.tag_name): downloading $($asset.name) ($([math]::Round($asset.size / 1MB)) MB)"

$zip = Join-Path ([System.IO.Path]::GetTempPath()) $asset.name
Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zip -Headers $headers

$extracted = Join-Path ([System.IO.Path]::GetTempPath()) "llama-extract"
if (Test-Path $extracted) { Remove-Item -Recurse -Force $extracted }
Expand-Archive -Path $zip -DestinationPath $extracted

# Layouts have moved between releases (root vs build/bin); find the server
# and take its whole directory - the DLLs beside it are its dependencies.
$server = Get-ChildItem -Path $extracted -Recurse -Filter "llama-server.exe" | Select-Object -First 1
if (-not $server) {
    throw "llama-server.exe is not in $($asset.name) - the packaging contract broke."
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Copy-Item -Path (Join-Path $server.DirectoryName "*") -Destination $OutDir -Recurse -Force

if (-not (Test-Path (Join-Path $OutDir "llama-server.exe"))) {
    throw "copy failed: llama-server.exe missing from $OutDir"
}

Set-Content -Path "build/llama-version.txt" -Value $release.tag_name
$count = (Get-ChildItem $OutDir -File).Count
Write-Host "bundled llama.cpp $($release.tag_name) -> $OutDir ($count files)"
