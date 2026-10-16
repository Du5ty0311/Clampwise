# Publish a new version of Clampwise. Every copy with "Check for a new version when Clampwise starts" on (Settings,
# Updates) then offers it, and installs it when the user chooses Update.
#
#   powershell -ExecutionPolicy Bypass -File packaging\release.ps1 -Version 1.0.1 [-SkipTests]
#
# First describe the version in CHANGELOG.md under a heading "## 1.0.1" (its text becomes the release notes the
# users read before updating), and commit your work on main. This script then sets the version, builds and checks the
# program (packaging\build.ps1: the tests, the built program's self-test), commits and tags the version, pushes, and
# makes the GitHub release with the zip (GitHub lists its SHA-256, which the update checks).

param([Parameter(Mandatory = $true)][string]$Version, [switch]$SkipTests)
$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $PSScriptRoot
Set-Location $Project

if ($Version -notmatch '^\d+\.\d+\.\d+$') { throw "Give the version as three numbers, e.g. 1.0.1." }
if ((git branch --show-current) -ne "main") { throw "Release from the main branch." }
if (git status --porcelain) { throw "Commit (or put aside) your changes first: the release is made from main as committed." }
$init = Join-Path $Project "src\clampwise\__init__.py"
$current = [regex]::Match([IO.File]::ReadAllText($init), '__version__ = "([^"]+)"').Groups[1].Value
if (-not ([version]$Version -gt [version]$current)) { throw "$Version is not above the current version $current." }
if (git tag --list "v$Version") { throw "The tag v$Version exists already." }

$changelog = [IO.File]::ReadAllText((Join-Path $Project "CHANGELOG.md"))
$section = [regex]::Match($changelog, "(?ms)^## $([regex]::Escape($Version))\b[^\n]*\n(.*?)(?=^## |\z)")
if (-not $section.Success -or -not $section.Groups[1].Value.Trim()) {
    throw "CHANGELOG.md has no section '## $Version' saying what changed: write it first (it becomes the release notes)."
}
$notes = $section.Groups[1].Value.Trim()

Write-Host "==> Version $current -> $Version" -ForegroundColor Cyan
$utf8 = New-Object Text.UTF8Encoding($false)
$pyproject = Join-Path $Project "pyproject.toml"
[IO.File]::WriteAllText($pyproject, ([IO.File]::ReadAllText($pyproject) -replace '(?m)^version = "[^"]+"', "version = `"$Version`""), $utf8)
[IO.File]::WriteAllText($init, ([IO.File]::ReadAllText($init) -replace '__version__ = "[^"]+"', "__version__ = `"$Version`""), $utf8)

Write-Host "==> Build" -ForegroundColor Cyan
$buildArgs = @("-ExecutionPolicy", "Bypass", "-File", (Join-Path $PSScriptRoot "build.ps1"))
if ($SkipTests) { $buildArgs += "-SkipTests" }
& powershell @buildArgs
if ($LASTEXITCODE -ne 0) {
    git checkout -- $pyproject $init
    throw "The build failed: nothing was committed or published (the version put back)."
}

Write-Host "==> Commit, tag and push" -ForegroundColor Cyan
git add $pyproject $init
git commit -q -m "Clampwise $Version"
git tag -a "v$Version" -m "Clampwise $Version"
git push -q origin main "v$Version"
if ($LASTEXITCODE -ne 0) { throw "Pushing failed." }

Write-Host "==> GitHub release" -ForegroundColor Cyan
$zip = Join-Path $Project "dist\Clampwise-windows.zip"
$body = @"
$notes

## Install or update

Clampwise $current or later: **Settings, Updates, Update to $Version** (or wait for the message when it starts).
New installs: download **Clampwise-windows.zip** below, unzip it anywhere and run ``Clampwise\Clampwise.exe``.

GNU General Public License, version 3. The source for this release is this tag.
"@
$notesFile = Join-Path $env:TEMP "clampwise-release-notes.md"
[IO.File]::WriteAllText($notesFile, $body, $utf8)
gh release create "v$Version" $zip --title "Clampwise $Version" --notes-file $notesFile --latest
if ($LASTEXITCODE -ne 0) { throw "The release was not made (the tag v$Version is pushed: run gh release create by hand)." }
Write-Host "==> Published Clampwise $Version" -ForegroundColor Green
