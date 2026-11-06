# Uses a project venv first, then the machine-local interpreter recorded at installation.
$ErrorActionPreference = 'Stop'
$pythonExe = Join-Path $PSScriptRoot '.venv-models\Scripts\python.exe'
$localPath = Join-Path $PSScriptRoot '.member2-python.txt'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    if (Test-Path -LiteralPath $localPath) {
        $pythonExe = (Get-Content -LiteralPath $localPath -Raw).Trim()
    } else {
        throw 'Create .venv-models and install requirements-models.txt first.'
    }
}
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Configured Python environment no longer exists.' }
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
Push-Location -LiteralPath $PSScriptRoot
try {
    & $pythonExe (Join-Path $PSScriptRoot '02_models.py') @args
    if ($LASTEXITCODE -ne 0) { throw "Model command failed with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
