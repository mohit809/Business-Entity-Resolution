@echo off
setlocal

echo ================================================================================
echo             LAUNCHING BER PRO INTERACTIVE WEB REVIEW DASHBOARD
echo ================================================================================
echo.
echo Starting web server at http://localhost:8080/ ...
echo Press Ctrl+C in this terminal window to stop the server anytime.
echo.

start "" http://localhost:8080/
py web/server.py 8080
