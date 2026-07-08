@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
set "NODE_DIR=C:\Program Files\nodejs"
set "NPM_CMD=C:\Program Files\nodejs\npm.cmd"
set "PATH=%NODE_DIR%;%PATH%"
set "MPLCONFIGDIR=%PROJECT_DIR%.matplotlib_cache"
set "LOKY_MAX_CPU_COUNT=8"

if not exist "%PYTHON_EXE%" (
  echo Python 3.11 was not found at:
  echo %PYTHON_EXE%
  echo.
  echo Install Python 3.11 or update PYTHON_EXE inside start_risknot.bat.
  pause
  exit /b 1
)

if not exist "%NPM_CMD%" (
  echo npm was not found at:
  echo %NPM_CMD%
  echo.
  echo Install Node.js or update NPM_CMD inside start_risknot.bat.
  pause
  exit /b 1
)

if not exist "%PROJECT_DIR%frontend\node_modules" (
  echo Frontend dependencies are not installed yet.
  echo Run this once:
  echo cd "%PROJECT_DIR%frontend"
  echo "%NPM_CMD%" install
  pause
  exit /b 1
)

echo Closing old RiskNot servers on ports 8000 and 5173 if they exist...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5173" ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>nul

start "RiskNot API" cmd /k "cd /d "%PROJECT_DIR%" && "%PYTHON_EXE%" -m uvicorn api.main:app"
start "RiskNot Frontend" cmd /k "cd /d "%PROJECT_DIR%frontend" && "%NPM_CMD%" run dev"

echo RiskNot is starting...
echo.
echo Backend health: http://localhost:8000/health
echo Frontend UI:    http://localhost:5173
echo.
echo Keep both terminal windows open while using the app.
pause
