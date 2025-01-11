# -------------------------------------------------------------------------------
# List of demos
# -------------------------------------------------------------------------------

set(DEMOS
    demo_three_heliostats
    demo_large_scene
)

message(STATUS "Adding demo programs for OptiX SolTrace ...")

# Loop through demos
foreach(PROGRAM ${DEMOS})
    message(STATUS "Adding ${PROGRAM}")

    # Add executable using OPTIX_add_sample_executable
    OPTIX_add_sample_executable(${PROGRAM} target_name
        ${SAMPLES_CUDA_DIR}/Soltrace.h
        ${SAMPLES_CUDA_DIR}/GeometryDataST.h
        ${SAMPLES_CUDA_DIR}/MaterialDataST.h
        ${SAMPLES_CUDA_DIR}/sun.cu
        ${SAMPLES_CUDA_DIR}/intersection.cu
        ${SAMPLES_CUDA_DIR}/materials.cu
        ${PROGRAM}.cpp
    )

    # Add compile definitions (specific to OptiX)
    target_compile_definitions(${PROGRAM} PRIVATE CUDA_NVRTC_ENABLED)

    # Copy DLLs to the bin folder
    copy_dll_to_bin(${PROGRAM})
endforeach()
