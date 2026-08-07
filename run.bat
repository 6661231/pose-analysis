@echo off
setlocal
title Pose Analysis System

echo ========================================
echo   POSE ANALYSIS SYSTEM v4.0
echo   YOLOv8-Pose Engine
echo ========================================
echo.

set "POSE_PYTHON="
if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "POSE_PYTHON=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not defined POSE_PYTHON if exist "%ProgramFiles%\Python313\python.exe" set "POSE_PYTHON=%ProgramFiles%\Python313\python.exe"
if not defined POSE_PYTHON if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "POSE_PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined POSE_PYTHON if exist "%ProgramFiles%\Python312\python.exe" set "POSE_PYTHON=%ProgramFiles%\Python312\python.exe"
if not defined POSE_PYTHON set "POSE_PYTHON=python"

"%POSE_PYTHON%" --version >nul 2>&1
if errorlevel 1 goto no_python

echo [*] Checking dependencies...
"%POSE_PYTHON%" -c "import fastapi, uvicorn, ultralytics, cv2, torch, numpy, multipart" >nul 2>&1
if not errorlevel 1 goto start_server

echo [*] Installing missing dependencies...
"%POSE_PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto dependencies_failed

:start_server
echo [*] Starting the service with:
"%POSE_PYTHON%" --version
echo Browser URL: http://localhost:8080
echo Press Ctrl+C to stop.
echo.
"%POSE_PYTHON%" api.py
if errorlevel 1 goto server_failed
goto done

:no_python
echo [ERROR] Python was not found.
goto failed

:dependencies_failed
echo [ERROR] Dependency installation failed. Check the network and try again.
goto failed

:server_failed
echo [ERROR] The service stopped unexpectedly. Keep the error shown above.
goto failed

:failed
pause
exit /b 1

:done
endlocal
