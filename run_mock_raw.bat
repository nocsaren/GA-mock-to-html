@echo off
setlocal enableextensions

REM Run the RAW (pull_from_bq-like) mock data generator.
REM Usage:
REM   run_mock_raw.bat [out_dir] [config_json]
REM Example:
REM   run_mock_raw.bat .\mock\output_raw .\mock\emoji_oracle_mock\config\example.json

REM Jump to repo root (parent of this mock folder)
cd /d "%~dp0.."

set "PY=.venv\Scripts\python.exe"
if exist "%PY%" (
  set "PY=%PY%"
) else (
  set "PY=python"
)

set "OUT=./mock/output_raw"
set "CFG=./mock/emoji_oracle_mock/config/example.json"

if not "%~1"=="" set "OUT=%~1"
if not "%~2"=="" set "CFG=%~2"

echo Using Python: %PY%
echo Output dir:  %OUT%
echo Config:      %CFG%
echo.

%PY% -m mock.emoji_oracle_mock --out "%OUT%" --kind raw --config "%CFG%"
if errorlevel 1 (
  echo.
  echo ERROR: mock generation failed.
  exit /b 1
)

echo.
echo Done.
echo Raw output: %OUT%\raw\pulled_from_bq.jsonl
endlocal
