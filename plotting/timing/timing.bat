@echo off
cd /d "C:\optixSoltraceDemos_build\bin\Release"

REM Run 10 times to average
for /l %%i in (1,1,10) do (
    REM Loop through input sets
    for %%j in (30 200 1700) do (
        for %%k in (10000 100000 1000000 10000000 100000000) do (
            demo_large_scene.exe %%j %%k >> timing_results.log
        )
    )
)

REM Keep the window open to view output
pause