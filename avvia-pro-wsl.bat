@echo off
setlocal

set "WSL_DISTRO=Ubuntu-24.04"
set "WSL_PROJECT_DIR=~/audio-transcription-PRO"

where wsl.exe >nul 2>nul
if errorlevel 1 (
    echo WSL is not installed or is not available in PATH.
    pause
    exit /b 1
)

wsl.exe --distribution "%WSL_DISTRO%" --cd "%WSL_PROJECT_DIR%" -- ./.venv/bin/python app.py
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" (
    echo App failed to start. Check the WSL distro, project path, and .venv setup.
    pause
)
exit /b %APP_EXIT%
