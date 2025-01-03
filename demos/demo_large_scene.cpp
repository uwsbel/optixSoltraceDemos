// TODO: 
// refactor, so things like loading shaders and creating modules are in separate cpp files
// obj count is hardcoded, make it dynamic
// aabb_input_flags should be the size of the object count


#include <cuda_runtime.h>
#include <sampleConfig.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <sutil/Exception.h>
#include <sutil/Record.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <chrono>

#include <iomanip>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>


#include <cuda/Soltrace.h>

typedef sutil::Record<soltrace::HitGroupData> HitGroupRecord;

const int      max_trace = 5;

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

const GeometryData::Parallelogram receiver(
    make_float3(9.0f, 0.0f, 0.0f),    // v1
    make_float3(0.0f, 0.0f, 7.0f),    // v2
    make_float3(-4.5f, 0.0f, 76.5f)     // anchor
);

// Compute an axis-aligned bounding box (AABB) for a parallelogram.
//   v1, v2: Vectors defining the parallelogram's sides.
//   anchor: The anchor point of the parallelogram.
// TODO: check why it's v1/dot(v1,v1) if v1 and v2 define the actual edge. 
inline OptixAabb parallelogram_bound( float3 v1, float3 v2, float3 anchor )
{
    const float3 tv1  = v1 / dot( v1, v1 );
    const float3 tv2  = v2 / dot( v2, v2 );
    // Compute the four corners of the parallelogram in 3D space.
    const float3 p00  = anchor;                 // Lower-left corner
    const float3 p01  = anchor + tv1;           // Lower-right corner
    const float3 p10  = anchor + tv2;           // Upper-left corner
    const float3 p11  = anchor + tv1 + tv2;     // Upper-right corner

    float3 m_min = fminf( fminf( p00, p01 ), fminf( p10, p11 ));
    float3 m_max = fmaxf( fmaxf( p00, p01 ), fmaxf( p10, p11 ));
    return {
        m_min.x, m_min.y, m_min.z,
        m_max.x, m_max.y, m_max.z
    };
}

// Build a GAS (Geometry Acceleration Structure) for the scene.
static void buildGas(
    const SoltraceState &state,
    const OptixAccelBuildOptions &accel_options,
    const OptixBuildInput &build_input,
    OptixTraversableHandle &gas_handle,
    CUdeviceptr &d_gas_output_buffer
    ) {

    OptixAccelBufferSizes gas_buffer_sizes;     // Holds required sizes for temp and output buffers.
    CUdeviceptr d_temp_buffer_gas;              // Temporary buffer for building the GAS.

    // Query the memory usage required for building the GAS.
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
        state.context,
        &accel_options,
        &build_input,
        1,
        &gas_buffer_sizes));

    // Allocate memory for the temporary buffer on the device.
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &d_temp_buffer_gas ),
        gas_buffer_sizes.tempSizeInBytes));

    // Non-compacted output and size of compacted GAS
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ),
                compactedSizeOffset + 8
                ) );

    // Emit property to store the compacted GAS size.
    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    // Build the GAS.
    OPTIX_CHECK( optixAccelBuild(
        state.context,                                  // OptiX context
        0,                                              // CUDA stream (default is 0)
        &accel_options,                                 // Acceleration build options
        &build_input,                                   // Build inputs
        1,                                              // Number of build inputs
        d_temp_buffer_gas,                              // Temporary buffer
        gas_buffer_sizes.tempSizeInBytes,               // Size of temporary buffer
        d_buffer_temp_output_gas_and_compacted_size,    // Output buffer
        gas_buffer_sizes.outputSizeInBytes,             // Size of output buffer
        &gas_handle,                                    // Output handle
        &emitProperty,                                  // Emitted properties
        1) );                                           // Number of emitted properties
        
    CUDA_CHECK( cudaFree( (void*)d_temp_buffer_gas ) );

    size_t compacted_gas_size;
    CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );

    // If the compacted GAS size is smaller, allocate a smaller buffer and compact the GAS
    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_gas_output_buffer ), compacted_gas_size ) );

        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact( state.context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle ) );

        CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
    }
    else
    {
        d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}

// Print a float3 structure
void printFloat3(const char* label, const float3& vec) {
    std::cout << label << ": (" << vec.x << ", " << vec.y << ", " << vec.z << ")\n";
}

// Load ptx given shader strings 
std::string loadPtxFromFile(const std::string& kernel_name) {
    // Construct the full PTX file path based on the kernel name
    const std::string ptx_file = std::string(SAMPLES_PTX_DIR) + "demo_large_scene_generated_" + kernel_name + ".cu.ptx";

    std::cout << "PTX file name: " << ptx_file << "\n";
    // Read the PTX file into a string
    std::ifstream f(ptx_file);
    if (!f.good()) {
        throw std::runtime_error("PTX file not found: " + ptx_file);
    }

    std::stringstream source_buffer;
    source_buffer << f.rdbuf();
    return source_buffer.str(); // Return the PTX content
}

// PRE-PROCESSING SCENE GEOMETRY AND SUN DEFINITION
// Compute the sun's coordinate frame
void computeSunFrame(SoltraceState& state, float3& sun_u, float3& sun_v) {
    float3 sun_vector_hat = normalize(state.params.sun_vector);
    float3 axis = (abs(state.params.sun_vector.x) < 0.9f) ? make_float3(1.0f, 0.0f, 0.0f) : make_float3(0.0f, 1.0f, 0.0f);
    // TODO: cross need to follow left
    sun_u = normalize(cross(axis, sun_vector_hat));
    sun_v = normalize(cross(sun_vector_hat, sun_u));
}

// Find the distance to the closest object along the sun vector
float computeSunPlaneDistance(SoltraceState& state, std::vector<soltrace::BoundingBoxVertex>& bounding_box_vertices) {
    float3 sun_vector_hat = normalize(state.params.sun_vector);
    // Max distance from Origin along sun vector
    float max_distance = 0.0f;
    for (auto& vertex : bounding_box_vertices) {
        float distance = dot(vertex.point, sun_vector_hat);
        vertex.distance = distance;
        if (distance > max_distance) {
            max_distance = distance;
        }
    }
    return max_distance;
}

// Project a point onto the plane at distance d along the sun vector
float3 projectOntoPlaneAtDistance(SoltraceState& state, const float3& point, float d) {
    float3 sun_vector_hat = normalize(state.params.sun_vector);
    float3 plane_center = d * sun_vector_hat;
    return point - dot(point - plane_center, sun_vector_hat) * sun_vector_hat;
}

// Compute the bounding box of all projected objects onto sun plane
void computeBoundingSunBox(SoltraceState& state, std::vector<soltrace::BoundingBoxVertex>& bounding_box_vertices) {
    float d = computeSunPlaneDistance(state, bounding_box_vertices);

    float3 sun_u, sun_v;
    computeSunFrame(state, sun_u, sun_v);

    // Project points onto the sun plane
    std::vector<soltrace::ProjectedPoint> projected_points;
    for (const auto& vertex : bounding_box_vertices) {
        // Compute the buffer for this point
        float buffer = vertex.distance * state.params.max_sun_angle;
        // Project the point onto the sun plane
        float3 projected_point = projectOntoPlaneAtDistance(state, vertex.point, d);
        float u = dot(projected_point, sun_u);
        float v = dot(projected_point, sun_v);
        soltrace::ProjectedPoint projected_point_uv = { buffer, make_float2(u, v) };
        projected_points.emplace_back(projected_point_uv);
    }

    // Find bounding box in the sun's frame
    float u_min = FLT_MAX;  // Initialize to the largest possible float
    float u_max = -FLT_MAX; // Initialize to the smallest possible float
    float v_min = FLT_MAX;
    float v_max = -FLT_MAX;
    for (const auto& point : projected_points) {
        u_min = fminf(u_min, point.point.x - point.buffer);
        u_max = fmaxf(u_max, point.point.x + point.buffer);
        v_min = fminf(v_min, point.point.y - point.buffer);
        v_max = fmaxf(v_max, point.point.y + point.buffer);
    }
    //std::cout << "u min: " << u_min << "\n";
    //std::cout << "u max: " << u_max << "\n";
    //std::cout << "v min: " << v_min << "\n";
    //std::cout << "v max: " << v_max << "\n";

    // Define sun bounding box vertices in the sun's frame
    std::vector<float2> sun_bounds_sun_frame = {
        make_float2(u_min, v_min), make_float2(u_max, v_min),   // bottom-left, bottom-right
        make_float2(u_max, v_max), make_float2(u_min, v_max)    // top-right, top-left
    };

    // Transform sun bounding box vertices to global frame
    std::vector<float3> sun_bounds_global_frame;
    for (const auto& vertex : sun_bounds_sun_frame) {
        float3 global_vertex = vertex.x * sun_u + vertex.y * sun_v + d * normalize(state.params.sun_vector);
        sun_bounds_global_frame.push_back(global_vertex);
    }

    state.params.sun_v0 = sun_bounds_global_frame[0];   // bottom-left
    state.params.sun_v1 = sun_bounds_global_frame[1];   // bottom-right
    state.params.sun_v2 = sun_bounds_global_frame[2];   // top-right
    state.params.sun_v3 = sun_bounds_global_frame[3];   // top-left
}

// Function to compute all 8 vertices of an AABB
void getAABBVertices(const OptixAabb& aabb, std::vector<soltrace::BoundingBoxVertex>& vertices) {
    // Min and max corners
    float3 minCorner = make_float3(aabb.minX, aabb.minY, aabb.minZ);
    float3 maxCorner = make_float3(aabb.maxX, aabb.maxY, aabb.maxZ);

	std::cout << "Min corner: (" << minCorner.x << ", " << minCorner.y << ", " << minCorner.z << ")\n";
	std::cout << "Max corner: (" << maxCorner.x << ", " << maxCorner.y << ", " << maxCorner.z << ")\n";

    // Compute the 8 vertices and distances
    std::vector<float3> points = {
        make_float3(minCorner.x, minCorner.y, minCorner.z), // 0
        make_float3(maxCorner.x, minCorner.y, minCorner.z), // 1
        make_float3(minCorner.x, maxCorner.y, minCorner.z), // 2
        make_float3(maxCorner.x, maxCorner.y, minCorner.z), // 3
        make_float3(minCorner.x, minCorner.y, maxCorner.z), // 4
        make_float3(maxCorner.x, minCorner.y, maxCorner.z), // 5
        make_float3(minCorner.x, maxCorner.y, maxCorner.z), // 6
        make_float3(maxCorner.x, maxCorner.y, maxCorner.z)  // 7
    };

    for (const auto& point : points) {
        float distance = 0.0f;
        vertices.push_back({ distance, point });
    }
}

// Function to collect AABB vertices for all objects
void collectAllAABBVertices(const OptixAabb aabbs[], int count, std::vector<soltrace::BoundingBoxVertex>& allVertices) {
    for (int i = 0; i < count; ++i) {
        getAABBVertices(aabbs[i], allVertices);
    }
}


// Build custom primitives (parallelograms for now, TODO generalize)
// add variable of the list of heliostats
void createGeometry(SoltraceState& state, std::vector<GeometryData::Parallelogram>& helistat_list)
{
    // TODO: receiver is treated as one for now, change that
    int obj_count = helistat_list.size() + 1;
    //std::vector<OptixAabb> aabb_list;
	// optixaabb vector where the size is the number of objects + 1 for the receiver
    std::vector<OptixAabb> aabb_list;
	aabb_list.resize(obj_count);

    
	//OptixAabb aabb[OBJ_COUNT];
    for (int i = 0; i < helistat_list.size(); i++) {
        
        aabb_list[i] = parallelogram_bound(helistat_list[i].v1, helistat_list[i].v2, helistat_list[i].anchor);
    }

    aabb_list[obj_count - 1] = parallelogram_bound(receiver.v1, receiver.v2, receiver.anchor);

    // Container to store all vertices
    std::vector<soltrace::BoundingBoxVertex> bounding_box_vertices;

    // Collect all vertices from AABBs
    collectAllAABBVertices(aabb_list.data(), obj_count, bounding_box_vertices);
	std::cout << "quick check: " << aabb_list.at(2).minX << "\n";
	std::cout << "val: " << helistat_list[2].v1.x << "\n";

    // Pass the vertices to computeBoundingSunBox
    computeBoundingSunBox(state, bounding_box_vertices);

    // Allocate memory on the device for the AABB array.
    CUdeviceptr d_aabb;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_aabb
        ), obj_count * sizeof( OptixAabb ) ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_aabb ),
                aabb_list.data(),
                obj_count * sizeof( OptixAabb ),
                cudaMemcpyHostToDevice
                ) );

	// initialize aabb_input_flags vector
    std::vector<uint32_t> aabb_input_flags(obj_count);
    for (int i = 0; i < obj_count; i++) {
        aabb_input_flags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    }

    // Define shader binding table (SBT) indices for each geometry. TODO generalize
    std::vector<uint32_t> sbt_index(obj_count);
    for (int i = 0; i < obj_count; i++) {
		sbt_index[i] = i;
	}

    // host to device
    CUdeviceptr    d_sbt_index;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_sbt_index ), obj_count * sizeof(uint32_t)));
    CUDA_CHECK( cudaMemcpy(
        reinterpret_cast<void*>( d_sbt_index ),
        sbt_index.data(),
        obj_count * sizeof(uint32_t),
        cudaMemcpyHostToDevice ) );

    // Configure the input for the GAS build process.
    OptixBuildInput aabb_input = {};
    aabb_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabb_input.customPrimitiveArray.aabbBuffers   = &d_aabb;
    aabb_input.customPrimitiveArray.flags         = aabb_input_flags.data();
    aabb_input.customPrimitiveArray.numSbtRecords = obj_count;
    aabb_input.customPrimitiveArray.numPrimitives = obj_count;
    aabb_input.customPrimitiveArray.sbtIndexOffsetBuffer         = d_sbt_index;
    aabb_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes    = sizeof( uint32_t );
    aabb_input.customPrimitiveArray.primitiveIndexOffset         = 0;

    // Set up acceleration structure (AS) build options.
    OptixAccelBuildOptions accel_options = {
        OPTIX_BUILD_FLAG_ALLOW_COMPACTION,  // buildFlags. Enable compaction to reduce memory usage.
        OPTIX_BUILD_OPERATION_BUILD         // operation. Build a new acceleration structure (not an update).
    };

    // Build the GAS using the defined AABBs and options.
    buildGas(
        state,             // Application state with OptiX context.
        accel_options,     // Build options.
        aabb_input,        // AABB input description.
        state.gas_handle,  // Output: traversable handle for the GAS.
        state.d_gas_output_buffer // Output: device buffer for the GAS.
    );

    CUDA_CHECK( cudaFree( (void*)d_aabb) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>(d_sbt_index) ) );
}

// Create OptiX modules for different components of the application.
// Modules correspond to different functionality, such as geometry handling, materials, and the sun.
void createModules( SoltraceState &state )
{
    // Options to control optimization and debugging settings.
    OptixModuleCompileOptions module_compile_options = {};
#if !defined( NDEBUG )
    module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

    // Create geometry module.
    {

        size_t inputSize = 0;   // Variable to store the size of the CUDA input source.
        //const char* input = sutil::getInputData(
        //    "optixSoltraceDemo",                      // No additional sample name.
        //    SAMPLES_PTX_DIR,              // Use your project's build directory.
        //    "parallelogram.cu",           // Name of the CUDA file.
        //    inputSize                     // Output: Size of the input CUDA source code.
        //);

		std::string ptx = loadPtxFromFile("parallelogram");

        OPTIX_CHECK_LOG(optixModuleCreate(
            state.context,                       // OptiX context for the application.
            &module_compile_options,             // Module compilation options.
            &state.pipeline_compile_options,     // Pipeline-level compile options.
            ptx.c_str(),                               // CUDA source code as input.
            ptx.size(),                           // Size of the CUDA source code.
            LOG, &LOG_SIZE,                      // Logs for diagnostic output.
            &state.geometry_module               // Output: Handle for the compiled module.
        ));
    }

    // Create shading/materials module.
    {
		std::string ptx = loadPtxFromFile("materials");
        OPTIX_CHECK_LOG(optixModuleCreate(
            state.context,
            &module_compile_options,
            &state.pipeline_compile_options,
            ptx.c_str(),
            ptx.size(),
            LOG, &LOG_SIZE,
            &state.shading_module
        ));
    }

    // Create the sun module.
    {
        std::string ptx = loadPtxFromFile("sun");

        OPTIX_CHECK_LOG(optixModuleCreate(
            state.context,
            &module_compile_options,
            &state.pipeline_compile_options,
            ptx.c_str(),
            ptx.size(),
            LOG, &LOG_SIZE,
            &state.sun_module
        ));
    }
}

// Create program group for the sun's ray generation program.
static void createSunProgram( SoltraceState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           sun_prog_group;                 // Handle for the sun program group.
    OptixProgramGroupOptions    sun_prog_group_options = {};    // Options for the program group (none needed in this case).
    OptixProgramGroupDesc       sun_prog_group_desc = {};       // Descriptor to define the program group.
    
    // Specify the kind of program group (Ray Generation).
    sun_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    // Link the ray generation program to the sun module and specify the function name.
    sun_prog_group_desc.raygen.module            = state.sun_module;
    sun_prog_group_desc.raygen.entryFunctionName = "__raygen__sun_source";

    // Create the program group
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,                 // OptiX context.
        &sun_prog_group_desc,          // Descriptor defining the program group.
        1,                             // Number of program groups to create (1 in this case).
        &sun_prog_group_options,       // Options for the program group.
        LOG, &LOG_SIZE,                // Logs to capture diagnostic information.
        &sun_prog_group                // Output: Handle for the created program group.
    ));

    program_groups.push_back(sun_prog_group);
    state.raygen_prog_group = sun_prog_group;
}

// Create program group for handling rays interacting with mirrors.
static void createMirrorProgram( SoltraceState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           radiance_mirror_prog_group;                 // Handle for the mirror program group.
    OptixProgramGroupOptions    radiance_mirror_prog_group_options = {};    // Options for the program group (none needed).
    OptixProgramGroupDesc       radiance_mirror_prog_group_desc = {};       // Descriptor for the program group.

    // Specify the kind of program group (Hit Group for handling intersections and shading).
    radiance_mirror_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    // Link the intersection shader (geometry handling) to the geometry module.
    radiance_mirror_prog_group_desc.hitgroup.moduleIS               = state.geometry_module;
    radiance_mirror_prog_group_desc.hitgroup.entryFunctionNameIS    = "__intersection__parallelogram";
    // Link the closest-hit shader (shading logic) to the shading module.
    radiance_mirror_prog_group_desc.hitgroup.moduleCH               = state.shading_module;
    radiance_mirror_prog_group_desc.hitgroup.entryFunctionNameCH    = "__closesthit__mirror";
    // No any-hit shader is used in this configuration (set to nullptr).
    radiance_mirror_prog_group_desc.hitgroup.moduleAH               = nullptr;
    radiance_mirror_prog_group_desc.hitgroup.entryFunctionNameAH    = nullptr;

    // Create the program group
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &radiance_mirror_prog_group_desc,
        1,
        &radiance_mirror_prog_group_options,
        LOG, &LOG_SIZE,
        &radiance_mirror_prog_group ) );

    program_groups.push_back(radiance_mirror_prog_group);
    state.radiance_mirror_prog_group = radiance_mirror_prog_group;
}

// Create program group for handling rays interacting with the receiver.
static void createReceiverProgram( SoltraceState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           radiance_receiver_prog_group;               // Handle for the receiver program group.
    OptixProgramGroupOptions    radiance_receiver_prog_group_options = {};  // Options for the program group (none needed).
    OptixProgramGroupDesc       radiance_receiver_prog_group_desc = {};     // Descriptor for the program group.

    // Specify the kind of program group (Hit Group for handling intersections and shading).
    radiance_receiver_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    // Link the intersection shader (geometry handling) to the geometry module.
    radiance_receiver_prog_group_desc.hitgroup.moduleIS               = state.geometry_module;
    radiance_receiver_prog_group_desc.hitgroup.entryFunctionNameIS    = "__intersection__parallelogram";
    // Link the closest-hit shader (shading logic) to the shading module.
    radiance_receiver_prog_group_desc.hitgroup.moduleCH               = state.shading_module;
    radiance_receiver_prog_group_desc.hitgroup.entryFunctionNameCH    = "__closesthit__receiver";
    // No any-hit shader is used in this configuration (set to nullptr).
    radiance_receiver_prog_group_desc.hitgroup.moduleAH               = nullptr;
    radiance_receiver_prog_group_desc.hitgroup.entryFunctionNameAH    = nullptr;

    // Create the program group
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &radiance_receiver_prog_group_desc,
        1,
        &radiance_receiver_prog_group_options,
        LOG, &LOG_SIZE,
        &radiance_receiver_prog_group ) );

    program_groups.push_back(radiance_receiver_prog_group);
    state.radiance_receiver_prog_group = radiance_receiver_prog_group;
}

// Create program group for handling rays that miss all geometry.
static void createMissProgram( SoltraceState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           radiance_miss_prog_group;       // Handle for the miss program group.
    OptixProgramGroupOptions    miss_prog_group_options = {};   // Options for the program group (none needed).
    OptixProgramGroupDesc       miss_prog_group_desc = {};      // Descriptor for the program group.
    
    // Specify the kind of program group (Miss Program for handling missed rays).
    miss_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    // Link the miss shader (background or environment shading) to the shading module.
    miss_prog_group_desc.miss.module             = state.shading_module;
    miss_prog_group_desc.miss.entryFunctionName  = "__miss__ms";

    // Create the program grou
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &miss_prog_group_desc,
        1,
        &miss_prog_group_options,
        LOG, &LOG_SIZE,
        &radiance_miss_prog_group ) );

    program_groups.push_back(radiance_miss_prog_group);
    state.radiance_miss_prog_group = radiance_miss_prog_group;
}

// Create and configure the OptiX pipeline.
// The pipeline is a core component in OptiX that ties together all program groups, modules, 
// and other configurations needed for ray tracing execution.
void createPipeline( SoltraceState &state )
{
    std::vector<OptixProgramGroup> program_groups;

    // Configure the pipeline compile options.
    state.pipeline_compile_options = {
        false,                                                  // usesMotionBlur: Disable motion blur.
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,          // traversableGraphFlags: Allow only a single GAS.
        2,    /* RadiancePRD uses 5 payloads */                 // numPayloadValues
        5,    /* Parallelogram intersection uses 5 attrs */     // numAttributeValues 
        OPTIX_EXCEPTION_FLAG_NONE,                              // exceptionFlags
        "params"                                                // pipelineLaunchParamsVariableName
    };

    // Prepare modules and program groups
    createModules( state );
    createSunProgram( state, program_groups );
    createMirrorProgram( state, program_groups );
    createReceiverProgram( state, program_groups );
    createMissProgram( state, program_groups );

    // Link program groups to pipeline
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth            = max_trace; // Maximum recursion depth for ray tracing.
    
    // Create the OptiX pipeline by linking the program groups.
    OPTIX_CHECK_LOG(optixPipelineCreate(
        state.context,                        // OptiX context.
        &state.pipeline_compile_options,      // Compile options for the pipeline.
        &pipeline_link_options,               // Link options for the pipeline.
        program_groups.data(),                // Array of program groups.
        static_cast<unsigned int>(program_groups.size()), // Number of program groups.
        LOG, &LOG_SIZE,                       // Logs for diagnostics.
        &state.pipeline                       // Output: Handle for the created pipeline.
    ));

    // Compute and configure the stack sizes for the pipeline.
    OptixStackSizes stack_sizes = {};
    for( auto& prog_group : program_groups )
    {
        OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes, state.pipeline ) );
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;

    // Compute stack sizes based on the maximum trace depth and other settings.
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,                      // Input stack sizes.
        max_trace,                         // Maximum trace depth.
        0,                                 // maxCCDepth: Maximum depth of continuation callables (none in this case).
        0,                                 // maxDCDepth: Maximum depth of direct callables (none in this case).
        &direct_callable_stack_size_from_traversal, // Output: Stack size for callable traversal.
        &direct_callable_stack_size_from_state,     // Output: Stack size for callable state.
        &continuation_stack_size                    // Output: Stack size for continuation stack.
    ));
    // Set the computed stack sizes for the pipeline.
    OPTIX_CHECK(optixPipelineSetStackSize(
        state.pipeline,                               // Pipeline to configure.
        direct_callable_stack_size_from_traversal,    // Stack size for direct callable traversal.
        direct_callable_stack_size_from_state,        // Stack size for direct callable state.
        continuation_stack_size,                      // Stack size for continuation stack.
        1                                            // maxTraversableDepth: Maximum depth of traversable hierarchy.
    ));
}

// Ccreate and configure the Shader Binding Table (SBT).
// The SBT is a crucial data structure in OptiX that links geometry and ray types
// with their corresponding programs (ray generation, miss, and hit group).
void createSBT( SoltraceState &state, std::vector<GeometryData::Parallelogram>& helistat_list)
{
    // TODO: we are assumping there's only one receiver!!
	int obj_count = helistat_list.size() + 1;
   
    // Ray generation program record
    {
        CUdeviceptr d_raygen_record;                   // Device pointer to hold the raygen SBT record.
        size_t      sizeof_raygen_record = sizeof( sutil::EmptyRecord );
        
        // Allocate memory for the raygen SBT record on the device.
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_raygen_record ),
            sizeof_raygen_record ) );

        sutil::EmptyRecord rg_sbt;  // Host representation of the raygen SBT record.

        // Pack the program header into the raygen SBT record.
        optixSbtRecordPackHeader( state.raygen_prog_group, &rg_sbt );

        // Copy the raygen SBT record from host to device.
        CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>( d_raygen_record ),
            &rg_sbt,
            sizeof_raygen_record,
            cudaMemcpyHostToDevice
        ) );

        // Assign the device pointer to the raygenRecord field in the SBT.
        state.sbt.raygenRecord = d_raygen_record;
    }

    // Miss program record
    {
        CUdeviceptr d_miss_record;
        size_t sizeof_miss_record = sizeof( sutil::EmptyRecord );
        
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_miss_record ),
            sizeof_miss_record*soltrace::RAY_TYPE_COUNT ) );

        sutil::EmptyRecord ms_sbt[soltrace::RAY_TYPE_COUNT];
        // Pack the program header into the first miss SBT record.
        optixSbtRecordPackHeader( state.radiance_miss_prog_group, &ms_sbt[0] );

        CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>( d_miss_record ),
            ms_sbt,
            sizeof_miss_record*soltrace::RAY_TYPE_COUNT,
            cudaMemcpyHostToDevice
        ) );

        // Configure the SBT miss program fields.
        state.sbt.missRecordBase          = d_miss_record;                   // Base address of the miss records.
        state.sbt.missRecordCount         = soltrace::RAY_TYPE_COUNT;        // Number of miss records.
        state.sbt.missRecordStrideInBytes = static_cast<uint32_t>( sizeof_miss_record );    // Stride between miss records.
    }

    // Hitgroup program record
    {
        // Total number of hitgroup records : one per ray type per object.
        const size_t count_records = soltrace::RAY_TYPE_COUNT * obj_count;
		std::vector<HitGroupRecord> hitgroup_records_list(count_records);

        // Note: Fill SBT record array the same order that acceleration structure is built.
        int sbt_idx = 0; // Index to track current record.

        // TODO: Material params - arbitrary right now

		for (int i = 0; i < helistat_list.size(); i++) {
			// Configure Heliostat SBT record.
            OPTIX_CHECK(optixSbtRecordPackHeader(
				state.radiance_mirror_prog_group,
				&hitgroup_records_list[sbt_idx]));
			hitgroup_records_list[sbt_idx].data.geometry_data.setParallelogram(helistat_list[i]);
            hitgroup_records_list[sbt_idx].data.material_data.mirror = {
                0.875425f, // Reflectivity.
				0.0f,  // Transmissivity.
				0.0f,  // Slope error.
				0.0f   // Specularity error.
			};
			sbt_idx++;
		}
         
        // Configure Receiver SBT record, this is for the last
	    OPTIX_CHECK( optixSbtRecordPackHeader(
            state.radiance_receiver_prog_group,
            &hitgroup_records_list[sbt_idx] ) );
        hitgroup_records_list[ sbt_idx ].data.geometry_data.setParallelogram( receiver );
        hitgroup_records_list[ sbt_idx ].data.material_data.receiver = {
            0.95f, // Reflectivity.
            0.0f,  // Transmissivity.
            0.0f,  // Slope error.
            0.0f   // Specularity error.
        };
        sbt_idx++;

        // Allocate memory for hitgroup records on the device.
        CUdeviceptr d_hitgroup_records;
        size_t      sizeof_hitgroup_record = sizeof( HitGroupRecord );
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_hitgroup_records ),
            sizeof_hitgroup_record*count_records
        ) );

        // Copy hitgroup records from host to device.
        CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>( d_hitgroup_records ),
            hitgroup_records_list.data(),
            sizeof_hitgroup_record*count_records,
            cudaMemcpyHostToDevice
        ) );

        // Configure the SBT hitgroup fields.
        state.sbt.hitgroupRecordBase            = d_hitgroup_records;             // Base address of hitgroup records.
        state.sbt.hitgroupRecordCount           = count_records;                  // Total number of hitgroup records.
        state.sbt.hitgroupRecordStrideInBytes   = static_cast<uint32_t>( sizeof_hitgroup_record );  // Stride size.
    }
}

// Callback function for logging messages from the OptiX context.
static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    // Format and print the log message to the standard error stream.
    // The message includes:
    // - The log level (e.g., error, warning, info).
    // - A tag for categorization (e.g., "API", "Internal").
    // - The actual message content.
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
        << message << "\n";
}

// Create and initialize an OptiX context.
void createContext( SoltraceState& state )
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( 0 ) );

    OptixDeviceContext context;
    CUcontext          cuCtx = 0;  // Use the current CUDA context (zero indicates current).

    // Set OptiX device context options.
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;    // Optional: Set a logging callback function for debugging.
    options.logCallbackLevel          = 4;                  // Verbosity level for logging (e.g., errors, warnings, etc.).
    
    // Create and store OptiX device context
    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );
    state.context = context;
}

// Initialize the launch parameters used for ray tracing.
void initLaunchParams( SoltraceState& state )
{
    // Set maximum ray depth.
    state.params.max_depth = max_trace;

    // Allocate memory for the hit point buffer.
    // The size depends on the sun (ray generation resolution) parameters and the maximum ray depth.
    // Start the timer
    auto start_buff = std::chrono::high_resolution_clock::now();
    const size_t hit_point_buffer_size = state.params.width * state.params.height * sizeof(float4) * state.params.max_depth;
    
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&state.params.hit_point_buffer),
        hit_point_buffer_size
    ));
    CUDA_CHECK(cudaMemset(state.params.hit_point_buffer, 0, hit_point_buffer_size));

    // Reflected direction buffer (commented out for memory saving).
    /*
    const size_t reflected_dir_buffer_size = state.params.width * state.params.height * sizeof(float4) * state.params.max_depth;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&state.params.reflected_dir_buffer),
        reflected_dir_buffer_size
    ));
    CUDA_CHECK(cudaMemset(state.params.reflected_dir_buffer, 0, reflected_dir_buffer_size));
    */ 

    // Create a CUDA stream for asynchronous operations.
    CUDA_CHECK( cudaStreamCreate( &state.stream ) );

    // Allocate memory for device-side launch parameters.
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_params ), sizeof( soltrace::LaunchParams ) ) );

    // Link the GAS handle.
    state.params.handle = state.gas_handle;
}

// Clean up resources and deallocate memory
void cleanupState( SoltraceState& state )
{
    OPTIX_CHECK( optixPipelineDestroy     ( state.pipeline                ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.raygen_prog_group       ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_mirror_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_receiver_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_miss_prog_group         ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.shading_module          ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.geometry_module         ) );
    OPTIX_CHECK( optixDeviceContextDestroy( state.context                 ) );


    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.raygenRecord       ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.missRecordBase     ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.hitgroupRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_gas_output_buffer    ) ) );
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.hit_point_buffer)));
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_params               ) ) );
}

// Check if components of a float4 are all zero. 
// If they are, it returns TRUE; otherwise, it returns FALSE.
// TODO: Move to util file
bool allZeros(float4 element) {
    return (element.y == 0 && element.z == 0 && element.w == 0);
}

// Write a vector of float4 data to a CSV file, filtering out rows based on allZeros.
void writeVectorToCSV(const std::string& filename, const std::vector<float4>& data) {
    std::ofstream outFile(filename);

    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open the file " << filename << " for writing." << std::endl;
        return;
    }

    // Write header
    outFile << "number,stage,loc_x,loc_y,loc_z\n";

    // Write data
    int ray_idx = 1;
    int idx = 0;
    for (const auto& element : data) {
        ++idx;
        if (idx <= max_trace && !allZeros(element)) {
            outFile << ray_idx << "," << element.x << "," << element.y << "," << element.z << "," << element.w << "\n";
        }
        else {
            idx = 0;
            ++ray_idx;
        }
    }

    outFile.close();
    std::cout << "Data successfully written to " << filename << std::endl;
}

int main(int argc, char* argv[])

{

    // command line input, read csv file to extract heliostat data
    std::string heliostat_data_file = "C:/Users/fang/Documents/NREL_SOLAR/large_scene/debug-system_rotated.csv";

	// skip first line, first go through to determine the number of lines (number of heliostats)
	std::ifstream file(heliostat_data_file);
	std::string line;
	int num_heliostats = 0;

    // print filename
	std::cout << "input filename: " << heliostat_data_file << std::endl;

    // skip the first line 
	std::getline(file, line);

    // list of heliostats
	std::vector<GeometryData::Parallelogram> heliostats_list;

    while (std::getline(file, line)) {
		num_heliostats++;

		std::istringstream ss(line);
		std::string token;
		
        // comma delimited values (float)
		std::vector<float> values;
		while (std::getline(ss, token, ',')) {
			values.push_back(stof(token));
        }

		// now create heliostat object
		GeometryData::Parallelogram heliostat(
			make_float3(values[0], values[1], values[2]),    // v1
			make_float3(values[3], values[4], values[5]),    // v2
			make_float3(values[6], values[7], values[8])  // anchor
		);

		heliostats_list.push_back(heliostat);
		printf("added heliostat %d at location %.4f, %.4f, %.4f\n", num_heliostats, values[6], values[7], values[8]);
    }

	std::cout << "number of heliostats: " << num_heliostats << std::endl;


    SoltraceState state;
	std::cout << "Starting Soltrace OptiX simulation..." << std::endl;
	std::cout << "samples ptx dir: " << SAMPLES_PTX_DIR << std::endl;

    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();

    try
    {
        // Initialize simulation parameters
        //state.params.sun_center = make_float3(0.0f, 0.0f, state.params.sun_radius);state.params.sun_center = make_float3(0.0f, 0.0f, state.params.sun_radius);    // z-component computed in raygen based on dims of sun
        state.params.sun_vector = make_float3(-235.034f, -5723.13f, 8196.98f);
        state.params.max_sun_angle = 0.00465;     // 4.65 mrad
        state.params.num_sun_points = 10000;

        state.params.width  = state.params.num_sun_points;
        state.params.height = 1;

        // Initialize OptiX components
        createContext(state);
        createGeometry(state, heliostats_list);
        createPipeline(state);
        createSBT(state, heliostats_list);
        initLaunchParams(state);

        // Copy launch parameters to device memory
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params), &state.params,
            sizeof(soltrace::LaunchParams), cudaMemcpyHostToDevice, state.stream));

        // Start the timer
        auto start_buff = std::chrono::high_resolution_clock::now();

        // Launch the OptiX pipeline
        OPTIX_CHECK( optixLaunch(
        state.pipeline,     // OptiX pipeline
        state.stream,       // CUDA stream used for this launch
        reinterpret_cast<CUdeviceptr>( state.d_params ),    // Device pointer to the launch parameters structure.
        sizeof( soltrace::LaunchParams ),                   // Size of launch parameters structure
        &state.sbt,          // Shader Binding Table(SBT): contains the pointers to program groups and associated data.
        state.params.width,  // Number of threads to launch along the X dimension.
        state.params.height, // Number of threads to launch along the Y dimension.
        1                    // Number of threads to launch along the Z dimension. Often set to 1 for 2D images or other non-volumetric workloads.
        ) );

        CUDA_SYNC_CHECK();

        // Stop the timer
        auto end_buff = std::chrono::high_resolution_clock::now();
        auto duration_ms_buff = std::chrono::duration_cast<std::chrono::milliseconds>(end_buff - start_buff);
        std::cout << "Execution time ray launch: " << duration_ms_buff.count() << " milliseconds" << std::endl;

        // Stop the timer
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Execution time full sim: " << duration_ms.count() << " milliseconds" << std::endl;

        // Copy hit point results from device memory
        std::vector<float4> hp_output_buffer(state.params.width * state.params.height * state.params.max_depth);
        CUDA_CHECK(cudaMemcpy(hp_output_buffer.data(), state.params.hit_point_buffer, state.params.width * state.params.height * state.params.max_depth * sizeof(float4), cudaMemcpyDeviceToHost));

        // Copy reflected direction results from device memory
        /*
        std::vector<float4> rd_output_buffer(state.params.width * state.params.height * state.params.max_depth);
        CUDA_CHECK(cudaMemcpy(rd_output_buffer.data(), state.params.reflected_dir_buffer, state.params.width * state.params.height * state.params.max_depth * sizeof(float4), cudaMemcpyDeviceToHost));
        */

        writeVectorToCSV("debug_scene-hit_counts-10000_rays_with_buffer.csv", hp_output_buffer);

        cleanupState(state);
    }
    catch(const std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}