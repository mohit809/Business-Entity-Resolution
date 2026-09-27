@echo off
setlocal enabledelayedexpansion

echo ================================================================================
echo             STARTING BUSINESS ENTITY RESOLUTION PIPELINE
echo ================================================================================

py src/pipeline.py --train-dir ../resources/dataset/train --test-dir ../resources/dataset/test --output-dir output --model-dir artifacts --mode all %*

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Pipeline completed successfully with exit code 0!
) else (
    echo.
    echo Pipeline execution failed with exit code %ERRORLEVEL%!
    exit /b %ERRORLEVEL%
)
