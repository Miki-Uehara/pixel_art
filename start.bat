@echo off
cd /d "%~dp0"

set PYTHON=C:\Users\belle\AppData\Local\Programs\Python\Python310\python.exe

if not exist "%PYTHON%" (
    echo ERROR: Python not found at %PYTHON%
    pause
    exit /b 1
)

taskkill /F /IM python.exe /T >nul 2>&1
timeout /t 1 /nobreak >nul

"%PYTHON%" app.py
