@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ========================================
echo   Avalon
echo ========================================
echo.

where python >nul 2>&1
if %errorlevel% neq 0 (
  echo Python not found. Please install Python 3.
  exit /b 1
)

where npm >nul 2>&1
if %errorlevel% neq 0 (
  echo npm not found. Please install Node.js.
  exit /b 1
)

echo [1/3] Installing Python dependencies...
python -m pip install -r backend\requirements.txt -q
if %errorlevel% neq 0 (
  echo pip install failed. Try: python -m pip install -r backend\requirements.txt
  exit /b 1
)
echo       Python deps OK

echo [2/3] Installing Node dependencies...
cd frontend
call npm install --silent
cd ..
echo       Node deps OK

echo [3/3] Starting services...
echo.

:: Start backend in background
start "bench-backend" /b python backend\main.py
timeout /t 3 /nobreak >nul

echo   Backend  -^> http://localhost:8771 (FastAPI)
echo   Frontend -^> http://localhost:5173 (Vite + React)
echo.
echo ========================================
echo   Press Ctrl+C to stop all services
echo ========================================
echo.

:: Start frontend in foreground
cd frontend
call npx vite --host
cd ..

:: Cleanup background processes
echo.
echo Shutting down backend...
for /f "tokens=2" %%i in ('tasklist /fi "WindowTitle eq bench-backend" /nh 2^>nul') do taskkill /pid %%i /f >nul 2>&1
