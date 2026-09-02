# SPDX-License-Identifier: MIT
<#
.SYNOPSIS
    hearth - one-command install.

.DESCRIPTION
    Creates a virtual environment, installs the dependencies, writes .env, and
    registers any model runners you point it at.

    **hearth holds no torch and no model.** It is a few hundred kilobytes of
    python whose job is to start model runners, keep exactly one of them loaded,
    and pass requests through. The models live in their own repositories with
    their own virtual environments; this only needs to know where they are.

.PARAMETER Root
    Where the virtual environment and the output directory go.
    Defaults to this repository.

.PARAMETER Runner
    A runner to register, as `name=path`. Repeat it for several. The path is the
    runner's repository, which must already be installed.

.PARAMETER Python
    The python used to create the virtual environment.

.EXAMPLE
    .\install.ps1

.EXAMPLE
    .\install.ps1 -Runner hunyuan3d=C:\dev\hunyuan3d-strix-halo `
                  -Runner trellis=C:\dev\trellis-strix-halo
#>
[CmdletBinding()]
param(
    [string]$Root = "",
    [string[]]$Runner = @(),
    [string]$Python = "py -3.12"
)

# Native tools report progress on stderr. Under output redirection, Windows
# PowerShell 5.1 turns those lines into error records, and a "Stop" preference
# would kill the script on the first one. So the preference stays "Continue" and
# every native step is checked through its exit code instead.
$ErrorActionPreference = "Continue"
function Assert-Ok([string]$step) {
    if ($LASTEXITCODE) { throw "$step failed with exit code $LASTEXITCODE" }
}

# $PSScriptRoot can be empty while parameter defaults are evaluated under
# Windows PowerShell 5.1, so the paths are resolved here instead.
$repo = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $Root) { $Root = $repo }

$venv = Join-Path $Root ".venv"
$py = Join-Path $venv "Scripts\python.exe"
$output = Join-Path $Root "output"

Write-Host "==> Root: $Root"
New-Item -ItemType Directory -Force -Path $Root | Out-Null

# 1. Virtual environment ------------------------------------------------------
if (-not (Test-Path $py)) {
    Write-Host "==> Creating virtual environment"
    & cmd /c "$Python -m venv `"$venv`""
    Assert-Ok "virtual environment creation"
}
& $py -m pip install --upgrade pip
Assert-Ok "pip upgrade"

# 2. Dependencies -------------------------------------------------------------
# **There is no torch here and there never will be.** See requirements.txt.
Write-Host "==> Installing dependencies"
& $py -m pip install --no-cache-dir -r (Join-Path $repo "requirements.txt")
Assert-Ok "dependency installation"

# 3. .env ---------------------------------------------------------------------
$envPath = Join-Path $repo ".env"
if (-not (Test-Path $envPath)) {
    Write-Host "==> Writing .env"
    (Get-Content (Join-Path $repo ".env.example") -Raw).Replace("__ROOT__", $Root) |
        Set-Content -Path $envPath -Encoding utf8
} else {
    Write-Host "==> Keeping the existing .env"
}
New-Item -ItemType Directory -Force -Path $output | Out-Null

# 4. Runners ------------------------------------------------------------------
# Each one is registered by tools/add_runner.ps1, which also checks that it
# answers before writing the entry.
foreach ($entry in $Runner) {
    $parts = $entry.Split('=', 2)
    if ($parts.Count -ne 2) {
        throw "-Runner takes name=path, got '$entry'"
    }
    Write-Host "==> Registering runner '$($parts[0])'"
    & (Join-Path $repo "tools\add_runner.ps1") -Name $parts[0] -Path $parts[1]
    Assert-Ok "registering $($parts[0])"
}

# 5. Smoke test: hearth starts and answers ------------------------------------
Write-Host "==> Checking that hearth starts"
Push-Location $repo
$reply = '{"id":1,"method":"ping"}', '{"id":2,"method":"shutdown"}' | & $py -m hearth
Pop-Location
if (-not ($reply -match '"role": "hearth"')) {
    throw "hearth did not answer ping: $reply"
}
Write-Host "    it answers"

Write-Host ""
Write-Host "Done."
Write-Host "  python   $py"
Write-Host "  settings $envPath"
Write-Host "  output   $output"
Write-Host ""
# Report what .env actually declares, not just what this run added: the file may
# already have been carrying runners from an earlier install.
$declared = ""
foreach ($line in Get-Content -LiteralPath $envPath -Encoding UTF8) {
    if ($line -match '^\s*HEARTH_RUNNERS\s*=(.*)$') { $declared = $Matches[1].Trim() }
}
if (-not $declared) {
    Write-Host "No runners registered yet. Install a model, then:"
    Write-Host "  .\tools\add_runner.ps1 -Name <name> -Path <its repository>"
} else {
    Write-Host "Runners: $declared"
    Write-Host "Try it:"
    Write-Host "  $py tools\rpc_call.py status"
}
