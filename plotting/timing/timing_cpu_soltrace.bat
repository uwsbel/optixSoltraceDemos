@echo off
cd /d "C:\stdev\soltrace\coretrace\strace"

REM Enable delayed variable expansion
setlocal enabledelayedexpansion

REM Run 10 times to average
for /l %%i in (1,1,10) do (
    REM Loop through input sets
    for %%j in (1) do (
        for %%k in (15500 155000 1550000) do (
            set /a ARG2=%%k * 100

            set ARG1=%%k
            set ARG3=123
            set ARG4=0
            set ARG5=0
            set ARG6=%%j
            strace.exe ..\..\app\deploy\samples\small-system.stinput !ARG1! !ARG2! !ARG3! !ARG4! !ARG5! !ARG6! >> timing_results.log
        )
    )
)
 
REM Keep the window open to view output
pause