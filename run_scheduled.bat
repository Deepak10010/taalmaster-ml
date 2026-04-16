@echo off
REM =====================================================================
REM  Scheduled pipeline wrapper — invoked by Windows Task Scheduler daily.
REM  Runs python pipeline.py and captures stdout+stderr to a timestamped
REM  log file under logs\. The scheduler itself has no UI output, so the
REM  log file is how you debug failures.
REM =====================================================================

cd /d "%~dp0"

if not exist logs mkdir logs

for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set dt=%%a
set stamp=%dt:~0,4%%dt:~4,2%%dt:~6,2%_%dt:~8,2%%dt:~10,2%%dt:~12,2%

set logfile=logs\pipeline_%stamp%.log

echo ===== Pipeline run started at %date% %time% ===== > "%logfile%"
python pipeline.py >> "%logfile%" 2>&1
set rc=%errorlevel%
echo. >> "%logfile%"
echo ===== Pipeline finished at %date% %time% (exit %rc%) ===== >> "%logfile%"

exit /b %rc%
