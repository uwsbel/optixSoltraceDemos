# TODO
# auto copy sutil, glfw, glad dll libraries to the bin folder
# linux 

# SPDX-FileCopyrightText: Copyright (c) 2019 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Welcome to the OptiX SDK build.  We have chosen CMake, because it can generate multiple
# build systems for multiple architectures from a single script.  There are many resources
# for CMake on-line at http://www.cmake.org and their wiki page,
# http://www.cmake.org/Wiki/CMake, in addition to the documentation that comes with the
# distribution.  There is also a book available if you wish to delve more deeply into
# various topics.

# If you wish to create your own project and use the SDK as a template there are a number
# of things you should do.
#
# 1. You should copy the contents of the SDK to a place of your choice.
#
# 2. You can remove any sample's directory you don't wish to build.  Be careful about
#    the following directories.
#
#    a. CMake - contains helper scripts that make this all work.  You should keep this.
#
#    b. sutil and putil
#              - Almost all of the samples make use of this shared code one way or another, so
#               you should probably keep them until you have your own frameowrk for your
#               code.
#
#    d. data  - This directory contains the cow.obj file used as an example for
#               many of the samples.  You can move cow.obj anywhere as long as
#               you fix all the file paths in the samples you wish to use it in.
#
# 3. You should update the list of sub directories that CMake needs to process below (look
#    for the comment "List of samples found in subdirectories.")
#

# The basic flow of execution of this file is to do the following.
#
# 1. Setup the project and other global settings.  This involves processing some helper
#    scripts.
#
# 2. Look for external dependencies, CUDA, and OptiX.
#
# 3. Process all the subdirectories' CMakeLists.txt files.  These files create all the
#    executable and library targets that are used to build the SDK.
#
# 4. As a convenience on Windows, copy the OptiX dlls into the build directories, so OptiX
#    doesn't have to be in the path to run the samples.
#
# 5. Set a CMake variable that indicates we have configured for the first time.  This
#    allows us to override and set varibles' defaults while allowing them to be modified
#    later.

# If you have any questions, don't feel shy about posting to the OptiX forums:
# https://devtalk.nvidia.com/default/board/90/

cmake_minimum_required(VERSION 3.18)

# This sets up the name of our project.  For our purposes the main thing this controls is
# the name of the VS solution file.
project(optixSoltrace LANGUAGES CXX CUDA)
# This enforces a particular version of CMake that we require to process the script files
# properly.  We rely on VERSION_GREATER_EQUAL which requires CMake v. 3.7.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 75;86;89)


cmake_policy(VERSION 3.5)

if( POLICY CMP0072 )
  # FindOpenGL prefers GLVND by default when available
  cmake_policy(SET CMP0072 NEW)
endif()

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

# CUDA 8 is broken for generating dependencies during configure
option(CUDA_GENERATE_DEPENDENCIES_DURING_CONFIGURE "Generate dependencies during configure time instead of only during build time." OFF)

# Passing the --use-local-env option to NVCC can dramatically speed up CUDA compilation
if(WIN32)
  option(CUDA_USE_LOCAL_ENV "Pass the '--use-local-env' option to NVCC; only applies to initial configure" ON)
endif()

find_package(CUDA 12.0 REQUIRED)
# If NVRTC was enabled/disabled, reset OPTIXIR/PTX to default
set( CUDA_NVRTC_ENABLED OFF CACHE BOOL "Use NVRTC to compile OPTIXIR/PTX at run-time instead of NVCC at build-time" )
if( NOT ( NOT ( NOT CUDA_NVRTC_ENABLED ) ) EQUAL ( NOT ( NOT CUDA_NVRTC_ENABLED_INTERNAL ) ) )
  message( STATUS "Resetting OPTIXIR/PTX support" )
  unset( SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT CACHE )
endif()
set( CUDA_NVRTC_ENABLED_INTERNAL ${CUDA_NVRTC_ENABLED} CACHE INTERNAL "Previous configured value (NVRTC)" FORCE )

# Disable OptiXIR support
set(SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT OFF CACHE BOOL "Enable support for generating OptiX-IR targetted input files" FORCE)


# This code looks funny, but CMake doesn't have an equality operator for boolean types
# (only integer and string). By doing NOT NOT VAL, you can force the original value into 0
# or 1 and allow the EQUAL operator to function correctly.
if( NOT ( NOT ( NOT SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT ) ) EQUAL ( NOT ( NOT SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT_INTERNAL ) ) )
  message( STATUS "Unsetting values associated with OptiX code generation" )
  # This allows us to reset dependent options if you change it.
  unset( SAMPLES_INPUT_GENERATE_OPTIXIR CACHE )
  unset( SAMPLES_INPUT_GENERATE_PTX CACHE )
  unset( GENERATE_DEBUG_DEVICE_CODE CACHE )
endif()
set(SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT_INTERNAL ${SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT} CACHE INTERNAL "Previous configured value (OPTIXIR)" FORCE)

if( SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT )
  option( SAMPLES_INPUT_GENERATE_OPTIXIR "Generate Optix-IR OptiX shaders" ON )
  option( SAMPLES_INPUT_GENERATE_PTX "Generate PTX OptiX shaders" OFF )
else()
  option( SAMPLES_INPUT_GENERATE_OPTIXIR "Generate Optix-IR OptiX shaders" OFF )
  option( SAMPLES_INPUT_GENERATE_PTX "Generate PTX OptiX shaders" ON )
endif()

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

  if( CUDA_VERSION VERSION_LESS "3.0" )
    optix_add_cuda_flag("--keep")
  endif()

  # Some CUDA 11.x toolkits erroneously complain about sm_50 being deprecated
  if(CUDA_VERSION VERSION_GREATER "11.0")
    optix_add_cuda_flag("-Wno-deprecated-gpu-targets")
  endif()

  if(CUDA_USE_LOCAL_ENV)
    optix_add_cuda_flag("--use-local-env")
  endif()

  if(CMAKE_CXX_STANDARD EQUAL 11)
    set(SAMPLES_NVRTC_CXX "-std=c++11")
  else()
    set(SAMPLES_NVRTC_CXX "")
  endif()

  if( NOT DEFINED CMAKE_CONFIGURATION_TYPES )
    if( NOT CMAKE_BUILD_TYPE STREQUAL CMAKE_BUILD_TYPE_PREVIOUS )
      message( STATUS "Resetting CUDA_NVRTC_FLAGS" )
      unset( CUDA_NVRTC_FLAGS CACHE )
    endif()
    set( CMAKE_BUILD_TYPE_PREVIOUS ${CMAKE_BUILD_TYPE} CACHE INTERNAL "Previous configured value (CMAKE_BUILD_TYPE)" FORCE )

    set( configs "Debug" "Release" "RelWithDebInfo" "MinSizeRel" )
    foreach( config ${configs} )
      if( ${config} STREQUAL "Debug" )
        set( SAMPLES_NVRTC_DEBUG "-G" )
      else()
        set( SAMPLES_NVRTC_DEBUG "-lineinfo" )
      endif()

      string( TOUPPER ${config} config_upper )
      set( CUDA_NVRTC_FLAGS_${config_upper} ${SAMPLES_NVRTC_CXX} -arch ${CUDA_MIN_SM_COMPUTE_TARGET} ${SAMPLES_NVRTC_DEBUG} -use_fast_math -default-device -rdc true -D__x86_64 CACHE STRING "List of NVRTC options just for the samples" FORCE )

      if( ${config} STREQUAL ${CMAKE_BUILD_TYPE} )
        set( CUDA_NVRTC_FLAGS ${CUDA_NVRTC_FLAGS_${config_upper}} CACHE STRING "List of NVRTC options just for the samples" )
      endif()
    endforeach()
  else()
    set( CUDA_NVRTC_FLAGS ${SAMPLES_NVRTC_CXX} -arch ${CUDA_MIN_SM_COMPUTE_TARGET} -lineinfo -use_fast_math -default-device -rdc true -D__x86_64 CACHE STRING "List of NVRTC options just for the samples" FORCE )
    set( CUDA_NVRTC_FLAGS_DEBUG ${SAMPLES_NVRTC_CXX} -arch ${CUDA_MIN_SM_COMPUTE_TARGET} -G -use_fast_math -default-device -rdc true -D__x86_64 CACHE STRING "List of NVRTC options just for the samples" FORCE )
  endif()
endif()

mark_as_advanced(CUDA_NVRTC_FLAGS)

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
  "${OptiX_INSTALL_DIR}/SDK/support"
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

# NVRTC include paths relative to the sample path
set(SAMPLES_RELATIVE_INCLUDE_DIRS "\\
  \"cuda\", \\
  \"sutil\", \\
  \".\", ")

# NVRTC absolute include paths to the headers used to build the samples
set(SAMPLES_ABSOLUTE_INCLUDE_DIRS "\\
  \"${OptiX_INCLUDE}\", \\
  \"${CUDA_INCLUDE_DIRS}\", ")

# Build a null-terminated option list for NVRTC
set( config_suffixes "_RELEASE" "_DEBUG" )
foreach( config_suffix ${config_suffixes} )
  # CMake doesn't allow empty strings in lists, so use a dummy suffix
  if( ${config_suffix} STREQUAL "_RELEASE" )
    set( config_suffix "" )
  endif()
  set(CUDA_NVRTC_OPTIONS${config_suffix})
  foreach(flag ${CUDA_NVRTC_FLAGS${config_suffix}})
    set(CUDA_NVRTC_OPTIONS${config_suffix} "${CUDA_NVRTC_OPTIONS${config_suffix}} \\\n  \"${flag}\",")
  endforeach()
  set(CUDA_NVRTC_OPTIONS${config_suffix} "${CUDA_NVRTC_OPTIONS${config_suffix}}")
endforeach()

configure_file(sampleConfig.h.in sampleConfig.h @ONLY)

# Path to sutil.h that all the samples need
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}
                     "${CMAKE_BINARY_DIR}/include"
                     ${CMAKE_CURRENT_BINARY_DIR}
                     ${CUDA_INCLUDE_DIRS}
                     )

set(SAMPLES_CUDA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cuda)

# Helper macro to generate PTX from the CUDA files in sutil.
macro(OPTIX_sutil_compile_to_optix_input generated_files)
  if( NOT CUDA_NVRTC_ENABLED )
    if( SAMPLES_INPUT_GENERATE_OPTIXIR )
      message("sutil OPTIXIR")
      CUDA_WRAP_SRCS( sutil OPTIXIR generated_files2 ${ARGN} )
      list(APPEND ${generated_files} ${generated_files2})
    endif()
    if( SAMPLES_INPUT_GENERATE_PTX )
      message("sutil PTX")
      CUDA_WRAP_SRCS( sutil PTX generated_files3 ${ARGN} )
      list(APPEND ${generated_files} ${generated_files3})
    endif()
  endif()
  message("${generated_files} = ${${generated_files}}")
endmacro()

# These calls will group PTX and CUDA files into their own directories in the Visual
# Studio projects.
macro(OPTIX_add_source_groups)
  if( NOT CUDA_NVRTC_ENABLED )
    if( SAMPLES_INPUT_GENERATE_PTX )
      source_group("PTX Files"  REGULAR_EXPRESSION ".+\\.ptx$")
    endif()
    if( SAMPLES_INPUT_GENERATE_OPTIXIR )
      source_group("OptixIR Files"  REGULAR_EXPRESSION ".+\\.optixir$")
    endif()
  endif()
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

  # Create the rules to build the PTX and/or OPTIX files.
  # Luning TODO: ptx.cu only need to get generated once. move this out of the loop. 
  if( SAMPLES_INPUT_GENERATE_OPTIXIR )
    CUDA_WRAP_SRCS( ${target_name} OPTIXIR generated_files2 ${cu_optix_source_files} ${cmake_options} OPTIONS ${options} )
    list(APPEND generated_files ${generated_files2})
  endif()
  if( SAMPLES_INPUT_GENERATE_PTX AND NOT CUDA_NVRTC_ENABLED)
    # CUDA_WRAP_SRCS( ${target_name} PTX generated_files3 ${cu_optix_source_files} ${cmake_options} OPTIONS ${options} )
    CUDA_WRAP_SRCS( "" PTX generated_files3 ${cu_optix_source_files} ${cmake_options} OPTIONS ${options} )

    list(APPEND generated_files ${generated_files3})
  endif()

  # Here is where we create the rule to make the executable.  We define a target name and
  # list all the source files used to create the target.  In addition we also pass along
  # the cmake_options parsed out of the arguments.
  add_executable(${target_name}
    ${source_files}
    ${generated_files}
    ${cmake_options}
    )
  message(STATUS "Target ${target_name} source files: ${source_files}")

    
    # find_package( OpenGL REQUIRED )

    if( UNIX )
      # On Unix-like systems, shared libraries can use the soname system.
      set(GLFW_LIB_NAME glfw)
    else()
      set(GLFW_LIB_NAME glfw3dll)
    endif()
    
  # Most of the samples link against the sutil library and the optix library.  Here is the
  # rule that specifies this linkage.
  target_link_libraries( ${target_name}
    ${CUDA_LIBRARIES}
    sutil_7_sdk
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

# COPY DLL FILES
set(DLLS_TO_COPY
    "${OptiX_INSTALL_DIR}/build/bin/$<CONFIG>/glad.dll"
    "${OptiX_INSTALL_DIR}/build/bin/$<CONFIG>/glfw3.dll"
    "${OptiX_INSTALL_DIR}/build/bin/$<CONFIG>/sutil_7_sdk.dll"
)

# Function to copy DLLs to the appropriate directory
function(copy_dll_to_bin target_name)
    # Use generator expressions to ensure the configuration is properly handled
    foreach(dll ${DLLS_TO_COPY})
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${dll}"
            "$<TARGET_FILE_DIR:${target_name}>"
            COMMENT "Copying ${dll} to $<TARGET_FILE_DIR:${target_name}>"
        )
    endforeach()
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

#################################################################

# Now that everything is done, indicate that we have finished configuring at least once.
# We use this variable to set certain defaults only on the first pass, so that we don't
# continually set them over and over again.
set(PASSED_FIRST_CONFIGURE ON CACHE INTERNAL "Already Configured once?")
