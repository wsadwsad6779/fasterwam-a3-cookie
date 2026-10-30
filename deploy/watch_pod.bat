@echo off
rem Double-click to start the retry loop in its own window.
rem It needs deploy\pod.env (copy deploy\pod.env.example). Start the paid
rem instance whenever you like; this loop picks it up on the next attempt.
setlocal
cd /d "%~dp0.."

if not exist "deploy\pod.env" (
  echo [watch_pod] missing deploy\pod.env - copy deploy\pod.env.example first
  pause
  exit /b 2
)

rem Prefer Git Bash: C:\Windows\System32\bash.exe is WSL and sees different paths.
set "GITBASH="
if exist "D:\APP\Git\bin\bash.exe" set "GITBASH=D:\APP\Git\bin\bash.exe"
if not defined GITBASH if exist "%ProgramFiles%\Git\bin\bash.exe" set "GITBASH=%ProgramFiles%\Git\bin\bash.exe"
if not defined GITBASH if exist "%LocalAppData%\Programs\Git\bin\bash.exe" set "GITBASH=%LocalAppData%\Programs\Git\bin\bash.exe"
if not defined GITBASH (
  echo [watch_pod] Git Bash not found - edit GITBASH in this file
  pause
  exit /b 3
)

"%GITBASH%" deploy/retry_launch.sh
echo.
echo [watch_pod] loop ended; see outputs\launch_retry.log
pause
