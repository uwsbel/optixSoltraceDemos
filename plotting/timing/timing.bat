@echo off
cd /d "C:\Users\fang\Documents\NREL_SOLAR\optix\build_debug\bin\Release"


for /l %%i in (1,1,10) do (
    REM Loop through input sets
    for %%k in (100000, 500000, 1000000, 10000000) do (
        demo_cylinder_receiver.exe 1 %%k >> timing_results.log
    )
)



REM Keep the window open to view output
pause