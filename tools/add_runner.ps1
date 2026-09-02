# SPDX-License-Identifier: MIT
<#
.SYNOPSIS
    Register a model runner with hearth, or remove one.

.DESCRIPTION
    Writes the four .env entries a runner needs (its name in HEARTH_RUNNERS,
    plus PYTHON, MODULE and CWD) and checks that it actually answers.

    **Nothing is guessed silently.** The python and the module are worked out
    from the runner's repository, and if either cannot be determined the script
    says which one and stops, rather than writing an entry that will fail later.

.PARAMETER Name
    What to call the runner. This is the name callers use, and the name that
    goes in HEARTH_RUNNERS.

.PARAMETER Path
    The runner's repository: the directory holding its `runners/<package>/`.

.PARAMETER Python
    The python that runs it. Worked out from the repository when omitted.

.PARAMETER Module
    The module to run, such as `runners.hunyuan3d`. Worked out when omitted.

.PARAMETER Remove
    Remove the runner's entries instead of adding them.

.PARAMETER SkipCheck
    Do not start the runner to verify it answers. **Only for a machine where the
    weights are not installed yet**; the entry is then unverified.

.EXAMPLE
    .\tools\add_runner.ps1 -Name hunyuan3d -Path C:\dev\hunyuan3d-strix-halo

.EXAMPLE
    .\tools\add_runner.ps1 -Name hunyuan3d -Remove
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [string]$Path = "",
    [string]$Python = "",
    [string]$Module = "",
    [switch]$Remove,
    [switch]$SkipCheck
)

$ErrorActionPreference = "Continue"

# $PSScriptRoot can be empty while parameter defaults are evaluated under
# Windows PowerShell 5.1, so the paths are resolved here instead.
$here = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$repo = Split-Path -Parent $here
$envPath = Join-Path $repo ".env"

if (-not (Test-Path $envPath)) {
    throw "no .env at $envPath. Run .\install.ps1 first."
}

$key = ($Name -replace '-', '_').ToUpper()
if ($Name -notmatch '^[A-Za-z0-9_-]+$') {
    throw "runner name '$Name' may only contain letters, digits, '-' and '_'."
}

# --- Read .env into an ordered list of lines ---------------------------------
# **Rewritten line by line, not regenerated.** Everything the operator wrote,
# comments included, has to survive being edited by a script.
$lines = [System.Collections.Generic.List[string]]::new()
Get-Content -LiteralPath $envPath -Encoding UTF8 | ForEach-Object { [void]$lines.Add($_) }

function Get-EnvValue([string]$name) {
    foreach ($line in $lines) {
        if ($line -match "^\s*$([regex]::Escape($name))\s*=(.*)$") { return $Matches[1].Trim() }
    }
    return $null
}

function Set-EnvValue([string]$name, [string]$value) {
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match "^\s*$([regex]::Escape($name))\s*=") {
            $lines[$i] = "$name=$value"
            return
        }
    }
    [void]$lines.Add("$name=$value")
}

function Remove-EnvValue([string]$name) {
    for ($i = $lines.Count - 1; $i -ge 0; $i--) {
        if ($lines[$i] -match "^\s*$([regex]::Escape($name))\s*=") { $lines.RemoveAt($i) }
    }
}

function Save-Env {
    # **This file is read by people.** Adding and removing entries repeatedly
    # otherwise leaves it pitted with double blank lines and blocks run together,
    # so runs of blank lines are collapsed and blank lines are trimmed off both
    # ends before writing.
    $out = [System.Collections.Generic.List[string]]::new()
    foreach ($line in $lines) {
        if ($line.Trim() -eq "" -and $out.Count -gt 0 -and $out[$out.Count - 1].Trim() -eq "") {
            continue
        }
        [void]$out.Add($line)
    }
    while ($out.Count -gt 0 -and $out[0].Trim() -eq "") { $out.RemoveAt(0) }
    while ($out.Count -gt 0 -and $out[$out.Count - 1].Trim() -eq "") { $out.RemoveAt($out.Count - 1) }
    Set-Content -LiteralPath $envPath -Value $out -Encoding UTF8
}

$names = @()
$current = Get-EnvValue "HEARTH_RUNNERS"
if ($current) { $names = @($current.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ }) }

# --- Remove ------------------------------------------------------------------
if ($Remove) {
    $names = @($names | Where-Object { $_ -ne $Name })
    Set-EnvValue "HEARTH_RUNNERS" ($names -join ',')
    foreach ($suffix in @("PYTHON", "MODULE", "CWD")) {
        Remove-EnvValue "HEARTH_RUNNER_${key}_${suffix}"
    }
    Save-Env
    Write-Host "removed runner '$Name'. HEARTH_RUNNERS is now: $($names -join ', ')"
    exit 0
}

# --- Work out the repository, python and module ------------------------------
if (-not $Path) { throw "-Path is required when adding a runner." }
$Path = (Resolve-Path -LiteralPath $Path -ErrorAction SilentlyContinue).Path
if (-not $Path -or -not (Test-Path $Path)) { throw "no such directory: $Path" }

if (-not $Module) {
    # Exactly one package under runners/ means there is no ambiguity to resolve.
    $runnersDir = Join-Path $Path "runners"
    if (-not (Test-Path $runnersDir)) {
        throw "no runners/ directory in $Path. Pass -Module explicitly if the layout differs."
    }
    $packages = @(Get-ChildItem -LiteralPath $runnersDir -Directory |
        Where-Object { Test-Path (Join-Path $_.FullName "__main__.py") })
    if ($packages.Count -ne 1) {
        $found = ($packages | ForEach-Object { $_.Name }) -join ', '
        throw "expected exactly one runner package under $runnersDir, found: $found. Pass -Module."
    }
    $Module = "runners.$($packages[0].Name)"
}

if (-not $Python) {
    # The virtual environment usually sits in the repository, and sometimes
    # beside it in a data directory the model's own installer created.
    $candidates = @(
        (Join-Path $Path ".venv\Scripts\python.exe"),
        (Join-Path (Split-Path -Parent $Path) "$(Split-Path -Leaf $Path)-data\.venv\Scripts\python.exe")
    )
    $Python = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $Python) {
        throw ("could not find a python for '$Name'. Looked in:`n  " +
            ($candidates -join "`n  ") + "`nPass -Python explicitly.")
    }
}
if (-not (Test-Path $Python)) { throw "no python at $Python" }

# --- Check that it answers ---------------------------------------------------
# **An entry that has never answered is worse than no entry**: the failure then
# surfaces in the middle of a generation instead of here.
if (-not $SkipCheck) {
    Write-Host "==> Checking that '$Name' answers"
    Push-Location $Path
    $reply = '{"id":1,"method":"capabilities"}', '{"id":2,"method":"shutdown"}' | & $Python -m $Module
    Pop-Location
    if (-not ($reply -match '"capabilities"')) {
        throw "'$Module' did not answer capabilities. Its own install may be incomplete.`n$reply"
    }
    Write-Host "    it answers"
}

# --- Write --------------------------------------------------------------------
if ($names -notcontains $Name) { $names += $Name }
Set-EnvValue "HEARTH_RUNNERS" ($names -join ',')
# A blank line before a block that is about to be appended, so a new runner does
# not end up glued to the previous one.
if ($null -eq (Get-EnvValue "HEARTH_RUNNER_${key}_PYTHON")) { [void]$lines.Add("") }
Set-EnvValue "HEARTH_RUNNER_${key}_PYTHON" $Python
Set-EnvValue "HEARTH_RUNNER_${key}_MODULE" $Module
Set-EnvValue "HEARTH_RUNNER_${key}_CWD" $Path
Save-Env

Write-Host "registered '$Name'"
Write-Host "    python $Python"
Write-Host "    module $Module"
Write-Host "    cwd    $Path"
Write-Host "HEARTH_RUNNERS is now: $($names -join ', ')"
