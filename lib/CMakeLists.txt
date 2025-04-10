# Add the geometry implementation
add_library(geometry STATIC
    geometry_manager.cpp
    data_manager.cpp
    pipeline_manager.cpp
    soltrace_system.cpp
    aperture.cpp
    element.cpp
)

# Make the library available for other targets
target_include_directories(geometry PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
