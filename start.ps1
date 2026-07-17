param(
  [switch]$NoInstall
)

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Backend = Join-Path $Root "backend"
$Frontend = Join-Path $Root "frontend"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Avalon" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if (-not $NoInstall) {
  Write-Host "[1/3] Installing Python dependencies..." -ForegroundColor Yellow
  python -m pip install -r (Join-Path $Backend "requirements.txt") -q 2>&1 | Out-Null
  if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
    Write-Host "pip install failed. Try: python -m pip install -r `"$Backend\requirements.txt`"" -ForegroundColor Red
    exit 1
  }
  Write-Host "      Python deps OK" -ForegroundColor Green

  Write-Host "[2/3] Installing Node dependencies..." -ForegroundColor Yellow
  Push-Location $Frontend
  npm install --silent 2>&1 | Out-Null
  Pop-Location
  if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
    Write-Host "npm install failed. Try: cd `"$Frontend`" && npm install" -ForegroundColor Red
    exit 1
  }
  Write-Host "      Node deps OK" -ForegroundColor Green
} else {
  Write-Host "[1/3] Skipping Python install (--NoInstall)" -ForegroundColor Gray
  Write-Host "[2/3] Skipping Node install (--NoInstall)" -ForegroundColor Gray
}

Write-Host "[3/3] Starting services..." -ForegroundColor Yellow
Write-Host ""

# Kill leftover processes on exit
$BackendJob = $null
try {
  $BackendJob = Start-Job -Name "bench-backend" -ScriptBlock {
    python $using:Backend\main.py
  }

  Start-Sleep -Seconds 3

  # Check if backend started successfully
  try {
    $null = Invoke-RestMethod -Uri "http://localhost:8771/api/gpu/list" -ErrorAction Stop -TimeoutSec 5
    Write-Host "  Backend  -> http://localhost:8771 (FastAPI)" -ForegroundColor Green
  }
  catch {
    Write-Host "  Backend  -> failed to start" -ForegroundColor Red
    Write-Host "  Check: python `"$Backend\main.py`"" -ForegroundColor Gray
  }

  Write-Host "  Frontend -> http://localhost:5173 (Vite + React)" -ForegroundColor Green
  Write-Host ""
  Write-Host "========================================" -ForegroundColor Cyan
  Write-Host "  Press Ctrl+C to stop all services" -ForegroundColor White
  Write-Host "========================================" -ForegroundColor Cyan
  Write-Host ""

  Push-Location $Frontend
  npx vite --host
  Pop-Location
}
finally {
  if ($BackendJob) {
    Write-Host "`nShutting down backend..." -ForegroundColor Yellow
    Stop-Job $BackendJob -ErrorAction SilentlyContinue
    Remove-Job $BackendJob -ErrorAction SilentlyContinue
  }
}
