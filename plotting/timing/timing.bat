@echo off
cd /d "C:\Users\fang\Documents\NREL_SOLAR\optix\build\bin\Release"


for /l %%i in (1,1,10) do (
    REM Loop through input sets
    for %%k in (112843, 1127489, 11244875) do (
        demo_large_scene.exe 0 0 %%k >> timing_results_f_new.log
    )
)

REM Keep the window open to view output
pause