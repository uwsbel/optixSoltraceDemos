# Add the geometry implementation
add_library(geometry STATIC
    geometry.cpp
)

# Make the library available for other targets
target_include_directories(geometry PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
