# Add the geometry implementation
add_library(optix_soltrace STATIC
    soltrace_optix_interface.cpp
)

# add interface folder to the include path
target_include_directories(optix_soltrace
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}       # expose interface folder to public
    PRIVATE ${CMAKE_SOURCE_DIR}/cuda
            ${CMAKE_SOURCE_DIR}/lib        
)

# Add the implementation file to the sources
target_sources(optix_soltrace PRIVATE
    soltrace_optix_interface.cpp
)
