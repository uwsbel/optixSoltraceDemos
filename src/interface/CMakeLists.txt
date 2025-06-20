add_library(OptixCSP_interface_soltrace STATIC
    soltrace_optix_interface.cpp
)

# add interface folder to the include path
target_include_directories(OptixCSP_interface_soltrace
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}       # expose interface folder to public
    PRIVATE ${PROJECT_SOURCE_DIR}/src/shaders
            ${PROJECT_SOURCE_DIR}/src/core    
)

target_link_libraries(OptixCSP_interface_soltrace
    PRIVATE OptixCSP_core)   