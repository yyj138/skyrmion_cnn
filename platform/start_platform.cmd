@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "%~dp0"

set "PYTHON_EXE=..\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo Creating Python 3.11 environment...
  py -3.11 -m venv ..\.venv
  if errorlevel 1 goto failed
)

"%PYTHON_EXE%" -c "import fastapi, uvicorn, multipart, PIL, numpy, pandas, tensorflow" >nul 2>&1
if errorlevel 1 (
  echo Installing platform dependencies...
  "%PYTHON_EXE%" -m pip install -r requirements.txt
  if errorlevel 1 goto failed
)

"%PYTHON_EXE%" launch_demo.py %*
if errorlevel 1 goto failed
exit /b 0

:failed
echo.
echo Platform startup failed. Read the message above and press any key to close.
pause >nul
exit /b 1
