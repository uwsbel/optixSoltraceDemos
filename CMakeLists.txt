cmake_minimum_required(VERSION 3.18)

# This sets up the name of our project.  For our purposes the main thing this controls is
# the name of the VS solution file.
project(optixSoltrace LANGUAGES CXX CUDA)
# This enforces a particular version of CMake that we require to process the script files
# properly.  We rely on VERSION_GREATER_EQUAL which requires CMake v. 3.7.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 75;86;89)


cmake_policy(VERSION 3.5)

if( POLICY CMP0074 )
  # find_package uses <PackageName>_ROOT variables.
  cmake_policy(SET CMP0074 NEW)
endif()

# Add paths to our CMake code to the module path, so they can be found automatically by CMake.
set(CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/cmake"
  ${CMAKE_MODULE_PATH}
)

# Set the default build to Release.  Note this doesn't do anything for the VS
# default build target which defaults to Debug when you first start it.
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

# Tells CMake to build all the libraries as shared libraries by default.  This can be
# overrided by individual libraries later.
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

##########
# Process our custom setup scripts here.

# Enable C++11. Needs to be done before the include of ConfigCompilerFlags.cmake below.
set(GCC_LIBSTDCPP11 ON)

# Include all CMake Macros.
include(Macros)
# Determine information about the compiler
include (CompilerInfo)
# Check for specific machine/compiler options.
include (ConfigCompilerFlags)

# Turn off the warning that NVCC issues when generating PTX from our CUDA samples.  This
# is a custom extension to the FindCUDA code distributed by CMake.
OPTION(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING "Suppress the \"Advisory: Cannot tell what pointer points to, assuming global memory space\" warning nvcc makes." ON)

# For Xcode 5, gcc is actually clang, so we have to tell CUDA to treat the compiler as
# clang, so that it doesn't mistake it for something else.
if(USING_CLANG_C)
  set(CUDA_HOST_COMPILER "clang" CACHE FILEPATH "Host side compiler used by NVCC")
endif()

option(CUDA_GENERATE_DEPENDENCIES_DURING_CONFIGURE "Generate dependencies during configure time instead of only during build time." OFF)

# Passing the --use-local-env option to NVCC can dramatically speed up CUDA compilation
if(WIN32)
  option(CUDA_USE_LOCAL_ENV "Pass the '--use-local-env' option to NVCC; only applies to initial configure" ON)
endif()

find_package(CUDA 12.0 REQUIRED)

option( SAMPLES_INPUT_GENERATE_PTX "Generate PTX OptiX shaders" ON )

# Determine if we are going to use the static CRT on windows.
if(WIN32)
  option(RELEASE_USE_STATIC_CRT "Build using the static CRT library" ON)
endif()

# Helper for fixing compiler flags
function(replace_flag var old_flag new_flag)
  string(REPLACE "${old_flag}" "${new_flag}" "${var}" "${${var}}")
  set(${var} "${${var}}" CACHE STRING "Default compiler flags" FORCE)
endfunction()

function(replace_flags old_flag new_flag)
  foreach(build "" _DEBUG _MINSIZEREL _RELEASE _RELWITHDEBINFO)
    replace_flag(CMAKE_C_FLAGS${build}   "${old_flag}" "${new_flag}")
    replace_flag(CMAKE_CXX_FLAGS${build} "${old_flag}" "${new_flag}")
  endforeach()
endfunction()

if(WIN32)
  if(RELEASE_USE_STATIC_CRT)
    replace_flags("/MD" "/MT")
  else()
    replace_flags("/MT" "/MD")
  endif()
endif(WIN32)

# Present the CUDA_64_BIT_DEVICE_CODE on the default set of options.
mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)

set(CUDA_MIN_SM_TARGET sm_50 CACHE STRING "Minimum CUDA SM architecture to use for compilation.")

function(optix_add_cuda_flag_config config flag)
  string(TOUPPER "${config}" config)
  list(FIND CUDA_NVCC_FLAGS${config} ${flag} index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS${config} ${flag})
    set(CUDA_NVCC_FLAGS${config} ${CUDA_NVCC_FLAGS${config}} CACHE STRING ${CUDA_NVCC_FLAGS_DESCRIPTION} FORCE)
  endif()
endfunction()

function(optix_add_cuda_flag flag)
  optix_add_cuda_flag_config( "" ${flag} )
endfunction()

# Add some useful default arguments to the NVCC and NVRTC flags.  This is an example of
# how we use PASSED_FIRST_CONFIGURE.  Once you have configured, this variable is TRUE
# and following block of code will not be executed leaving you free to edit the values
# as much as you wish from the GUI or from ccmake.
if( NOT PASSED_FIRST_CONFIGURE )
  set(CUDA_NVCC_FLAGS_DESCRIPTION "Semi-colon delimit multiple arguments.")
  string(REPLACE "sm_" "compute_" CUDA_MIN_SM_COMPUTE_TARGET ${CUDA_MIN_SM_TARGET})

  list(FIND CUDA_NVCC_FLAGS "-arch" index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS -arch ${CUDA_MIN_SM_TARGET})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
  endif()

  optix_add_cuda_flag("--use_fast_math")
  optix_add_cuda_flag("-lineinfo")

  # CMAKE_CONFIGURATION_TYPES is only defined for multi-config build systems like
  # MSVC and Ninja, but we need to generate flags for each configuration
  # regardless.
  if( DEFINED CMAKE_CONFIGURATION_TYPES )
    set( OPTIX_CONFIGURATION_TYPES ${CMAKE_CONFIGURATION_TYPES} )
  else()
    set( OPTIX_CONFIGURATION_TYPES "Debug" "Release" "RelWithDebInfo" "MinSizeRel" )
  endif()

  foreach( config ${OPTIX_CONFIGURATION_TYPES} )
    if( ${config} STREQUAL "Debug" )
      optix_add_cuda_flag_config( _${config} "-G" )
      optix_add_cuda_flag_config( _${config} "-O0" )
    endif()
  endforeach()
  optix_add_cuda_flag("-Wno-deprecated-gpu-targets")

  if(CUDA_USE_LOCAL_ENV)
    optix_add_cuda_flag("--use-local-env")
  endif()

  if( NOT DEFINED CMAKE_CONFIGURATION_TYPES )
    if( NOT CMAKE_BUILD_TYPE STREQUAL CMAKE_BUILD_TYPE_PREVIOUS )
      message( STATUS "Resetting CUDA_NVRTC_FLAGS" )
      unset( CUDA_NVRTC_FLAGS CACHE )
    endif()
    set( CMAKE_BUILD_TYPE_PREVIOUS ${CMAKE_BUILD_TYPE} CACHE INTERNAL "Previous configured value (CMAKE_BUILD_TYPE)" FORCE )
  endif()
endif()

# This passes a preprocessor definition to cl.exe when processing CUDA code.
if(USING_WINDOWS_CL)
  list(APPEND CUDA_NVCC_FLAGS --compiler-options /D_USE_MATH_DEFINES)
endif()

# Put all the runtime stuff in the same directory.  By default, CMake puts each targets'
# output into their own directory.  We want all the targets to be put in the same
# directory, and we can do this by setting these variables.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Set the environment variables for PTX directory
set(ENV{OPTIX_SAMPLES_SDK_PTX_DIR} "${CMAKE_BINARY_DIR}/lib/ptx")
set(ENV{OPTIX_EXP_SAMPLES_SDK_PTX_DIR} "${CMAKE_BINARY_DIR}/lib/ptx")

# Create a flag for mac which will allow apps to add the local cuda toolkit
# install path to the app's rpath.
if( APPLE )
  set( CUDA_TOOLKIT_RPATH_FLAG "-Wl,-rpath,${CUDA_TOOLKIT_ROOT_DIR}/lib" )
endif()

# Search for the OptiX libraries and include files.
find_package(OptiX REQUIRED)

link_directories("$<IF:$<CONFIG:Debug>,${OptiX_INSTALL_DIR}/build/lib/Debug,${OptiX_INSTALL_DIR}/build/lib/Release>")

# Add the path to the OptiX headers to our include paths.
include_directories(
  "${OptiX_INCLUDE}"
  "${OptiX_INSTALL_DIR}/SDK"
)

# Select whether to use NVRTC or NVCC to generate PTX
if( NOT SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT AND SAMPLES_INPUT_GENERATE_OPTIXIR )
  message( SEND_ERROR "Must enable SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT to enable SAMPLES_INPUT_GENERATE_OPTIXIR" )
endif()


##################################################################
# SUtil compilation
set(SAMPLES_PTX_DIR "${CMAKE_BINARY_DIR}/lib/ptx/")
set(SAMPLES_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(CUDA_GENERATED_OUTPUT_DIR ${SAMPLES_PTX_DIR})

if( WIN32 )
  string(REPLACE "/" "\\\\" SAMPLES_PTX_DIR ${SAMPLES_PTX_DIR})
else( WIN32 )
  if( USING_GNU_C AND NOT APPLE )
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DM_PI=3.14159265358979323846" )
  endif()
endif( WIN32 )

set(SAMPLES_CUDA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cuda")


configure_file(sampleConfig.h.in sampleConfig.h @ONLY)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}
                     ${CMAKE_CURRENT_BINARY_DIR}
                     ${CUDA_INCLUDE_DIRS}
                     )

set(SAMPLES_CUDA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cuda)


# These calls will group PTX and CUDA files into their own directories in the Visual
# Studio projects.
macro(OPTIX_add_source_groups)
  source_group("PTX Files"  REGULAR_EXPRESSION ".+\\.ptx$")
  source_group("CUDA Files" REGULAR_EXPRESSION ".+\\.cu$")
endmacro()

#########################################################
# OPTIX_add_sample_executable
#
# Convience function for adding samples to the code.  You can copy the contents of this
# funtion into your individual project if you wish to customize the behavior.  Note that
# in CMake, functions have their own scope, whereas macros use the scope of the caller.
function(OPTIX_add_sample_executable target_name_base target_name_var)

  set( target_name ${target_name_base} )
  set( ${target_name_var} ${target_name} PARENT_SCOPE )

  OPTIX_add_source_groups()

  # Separate the sources from the CMake and CUDA options fed to the macro.  This code
  # comes from the CUDA_COMPILE_PTX macro found in FindCUDA.cmake.  We are copying the
  # code here, so that we can use our own name for the target.  target_name is used in the
  # creation of the output file names, and we want this to be unique for each target in
  # the SDK.
  CUDA_GET_SOURCES_AND_OPTIONS(source_files cmake_options options ${ARGN})

  # Isolate OBJ target files. NVCC should only process these files and leave PTX targets for NVRTC
  set(cu_obj_source_files)
  set(cu_optix_source_files)
  foreach(file ${source_files})
    get_source_file_property(_cuda_source_format ${file} CUDA_SOURCE_PROPERTY_FORMAT)
    if(${_cuda_source_format} MATCHES "OBJ")
      list(APPEND cu_obj_source_files ${file})
    else()
      list(APPEND cu_optix_source_files ${file})
    endif()
  endforeach()

  # Create the rules to build the OBJ from the CUDA files.
  CUDA_WRAP_SRCS( ${target_name} OBJ generated_files ${cu_obj_source_files} ${cmake_options} OPTIONS ${options} )

  CUDA_WRAP_SRCS( "" PTX generated_files3 ${cu_optix_source_files} ${cmake_options} OPTIONS ${options} )

  list(APPEND generated_files ${generated_files3})

  # Here is where we create the rule to make the executable.  We define a target name and
  # list all the source files used to create the target.  In addition we also pass along
  # the cmake_options parsed out of the arguments.
  add_executable(${target_name}
    ${source_files}
    ${generated_files}
    ${cmake_options}
    )
  message(STATUS "Target ${target_name} source files: ${source_files}")
    
  target_link_libraries( ${target_name}
    ${CUDA_LIBRARIES}
    geometry
    )

  set_target_properties( ${target_name} PROPERTIES
    COMPILE_DEFINITIONS
    "OPTIX_SAMPLE_NAME_DEFINE=${target_name};OPTIX_SAMPLE_DIR_DEFINE=${target_name}" )

  if( UNIX AND NOT APPLE )
    # Force using RPATH instead of RUNPATH on Debian
    target_link_libraries( ${target_name} "-Wl,--disable-new-dtags" )
  endif()

  if(USING_GNU_CXX)
    target_link_libraries( ${target_name} m ) # Explicitly link against math library (C samples don't do that by default)
  endif()
endfunction()

#------------------------------------------------------------
# Copy data directory to BUILD tree 
# Install data directory 
# Install sample template_project
#------------------------------------------------------------

get_property(MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(MULTI_CONFIG)
    file(COPY ${CMAKE_SOURCE_DIR}/data/ DESTINATION ${CMAKE_BINARY_DIR}/bin/data/)
    message(STATUS "Using a multi-config generator (${CMAKE_GENERATOR})")
    message(STATUS "Data directory copied to: ${CMAKE_BINARY_DIR}/bin/data/")
    message(STATUS "Binaries will be created in ${CMAKE_BINARY_DIR}/bin/<CONFIGURATION>/")
else()
    file(COPY ${CMAKE_SOURCE_DIR}/data/ DESTINATION ${CMAKE_BINARY_DIR}/data/)
    message(STATUS "Using a single configuration generator (${CMAKE_GENERATOR})")
    message(STATUS "Data directory copied to: ${CMAKE_BINARY_DIR}/data/")
    message(STATUS "Binaries will be created in ${CMAKE_BINARY_DIR}/bin/")
endif()

#########################################################
#  List of samples found in subdirectories.
#
# If you wish to start your own sample, you can copy one of the sample's directories.
# Just make sure you rename all the occurances of the sample's name in the C code as well
# and the CMakeLists.txt file.
add_subdirectory(lib)
add_subdirectory(demos)
add_subdirectory(interface)
#################################################################

# Now that everything is done, indicate that we have finished configuring at least once.
# We use this variable to set certain defaults only on the first pass, so that we don't
# continually set them over and over again.
set(PASSED_FIRST_CONFIGURE ON CACHE INTERNAL "Already Configured once?")
