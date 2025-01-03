#pragma once
#include <optix.h>
#include <sutil/vec_math.h>
#include <vector>
#include <cuda/GeometryDataST.h>
#include <cuda/Soltrace.h>
#include <string>


struct SoltraceState
{
    OptixDeviceContext          context                         = 0;
    OptixTraversableHandle      gas_handle                      = {};
    CUdeviceptr                 d_gas_output_buffer             = {};

    OptixModule                 geometry_module                 = 0;
    OptixModule                 shading_module                  = 0;
    OptixModule                 sun_module                      = 0;

    OptixProgramGroup           raygen_prog_group               = 0;
    OptixProgramGroup           radiance_miss_prog_group        = 0;
    OptixProgramGroup           radiance_mirror_prog_group      = 0;
    OptixProgramGroup           radiance_receiver_prog_group    = 0;

    OptixPipeline               pipeline                        = 0;
    OptixPipelineCompileOptions pipeline_compile_options        = {};

    CUstream                    stream                          = 0;
    
    soltrace::LaunchParams      params;
    soltrace::LaunchParams*     d_params                        = nullptr;

    OptixShaderBindingTable     sbt                             = {};

    // TODO: list of geometries - add geometries first and then iterate through list to create SBT
};
