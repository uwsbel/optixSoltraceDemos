# Point to the prebuilt sutil library in the SDK
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0/include"
  "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0/SDK/"
)

# Prebuilt sutil library path
link_directories("C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0/build/lib/Release")

# Add only the CUDA files or other project-specific files (NO sutil sources)
set(sources
    ${SAMPLES_CUDA_DIR}/helpers.h
    ${SAMPLES_CUDA_DIR}/sun.cu
    ${SAMPLES_CUDA_DIR}/materials.cu
    ${SAMPLES_CUDA_DIR}/parallelogram.cu
    ${SAMPLES_CUDA_DIR}/Soltrace.h
    ${SAMPLES_CUDA_DIR}/GeometryDataST.h
    ${SAMPLES_CUDA_DIR}/MaterialDataST.h
)

# Link against the prebuilt sutil library
set(sutil_target "sutil_local")
add_library(${sutil_target} STATIC ${sources})

# Link the prebuilt sutil library
target_link_libraries(${sutil_target} PRIVATE
    ${GLFW_LIB_NAME}
    # glad
    imgui
    sutil_7_sdk
    ${CUDA_LIBRARIES}
)

# Windows-specific definitions
if(WIN32)
  target_compile_definitions(${sutil_target} PUBLIC GLAD_GLAPI_EXPORT)
  target_link_libraries(${sutil_target} PRIVATE winmm.lib)
endif()
