@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
set "NODE_DIR=C:\Program Files\nodejs"
set "NPM_CMD=C:\Program Files\nodejs\npm.cmd"
set "PATH=%NODE_DIR%;%PATH%"

if not exist "%PYTHON_EXE%" (
  echo Python 3.11 was not found at:
  echo %PYTHON_EXE%
  echo.
  echo Install Python 3.11 or update PYTHON_EXE inside setup_risknot.bat.
  pause
  exit /b 1
)

if not exist "%NPM_CMD%" (
  echo npm was not found at:
  echo %NPM_CMD%
  echo.
  echo Install Node.js or update NPM_CMD inside setup_risknot.bat.
  pause
  exit /b 1
)

echo Installing Python dependencies...
"%PYTHON_EXE%" -m pip install -r "%PROJECT_DIR%requirements.txt"
if errorlevel 1 (
  echo Python dependency installation failed.
  pause
  exit /b 1
)

echo Installing frontend dependencies...
cd /d "%PROJECT_DIR%frontend"
"%NPM_CMD%" install
if errorlevel 1 (
  echo Frontend dependency installation failed.
  pause
  exit /b 1
)

echo.
echo RiskNot setup complete.
pause
