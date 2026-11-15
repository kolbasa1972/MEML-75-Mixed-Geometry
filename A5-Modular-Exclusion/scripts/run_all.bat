@echo off
chcp 65001 > nul
echo ===================================================
echo Starting generation of all logs...
echo ===================================================

for %%f in (*.py) do (
    echo Running %%f ...
    python "%%f" > "%%~nf_log.txt" 2>&1
    if errorlevel 1 (
        echo [ERROR] Failed to run %%f. Check %%~nf_log.txt for details.
    ) else (
        echo [OK] Finished %%f.
    )
)

echo ===================================================
echo All scripts processed. Press any key to exit.
pause