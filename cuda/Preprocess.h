#pragma once

#include <optix.h>
#include <sutil/vec_math.h>
#include <vector_types.h>


const uint32_t OBJ_COUNT = 1692;

struct SoltraceState
{
    OptixDeviceContext          context = 0;
    OptixTraversableHandle      gas_handle = {};
    CUdeviceptr                 d_gas_output_buffer = {};

    OptixModule                 geometry_module = 0;
    OptixModule                 shading_module = 0;
    OptixModule                 sun_module = 0;

    OptixProgramGroup           raygen_prog_group = 0;
    OptixProgramGroup           radiance_miss_prog_group = 0;
    OptixProgramGroup           radiance_mirror_prog_group = 0;
    OptixProgramGroup           radiance_receiver_prog_group = 0;

    OptixPipeline               pipeline = 0;
    OptixPipelineCompileOptions pipeline_compile_options = {};

    CUstream                    stream = 0;

    soltrace::LaunchParams      params;
    soltrace::LaunchParams*     d_params = nullptr;

    OptixShaderBindingTable     sbt = {};

    std::vector<GeometryData::Parallelogram> heliostats;
    GeometryData::Parallelogram receiver;
};

// Compute an axis-aligned bounding box (AABB) for a parallelogram.
//   v1, v2: Vectors defining the parallelogram's sides.
//   anchor: The anchor point of the parallelogram.
inline OptixAabb parallelogram_bound(float3 v1, float3 v2, float3 anchor)
{
    const float3 tv1 = v1 / dot(v1, v1);
    const float3 tv2 = v2 / dot(v2, v2);
    // Compute the four corners of the parallelogram in 3D space.
    const float3 p00 = anchor;                 // Lower-left corner
    const float3 p01 = anchor + tv1;           // Lower-right corner
    const float3 p10 = anchor + tv2;           // Upper-left corner
    const float3 p11 = anchor + tv1 + tv2;     // Upper-right corner

    float3 m_min = fminf(fminf(p00, p01), fminf(p10, p11));
    float3 m_max = fmaxf(fmaxf(p00, p01), fmaxf(p10, p11));
    return {
        m_min.x, m_min.y, m_min.z,
        m_max.x, m_max.y, m_max.z
    };
}

// PRE-PROCESSING SCENE GEOMETRY AND SUN DEFINITION
// Compute the sun's coordinate frame
void computeSunFrame(SoltraceState& state, float3& sun_u, float3& sun_v) {
    float3 sun_vector_hat = normalize(state.params.sun_vector);
    float3 axis = (abs(state.params.sun_vector.x) < 0.9f) ? make_float3(1.0f, 0.0f, 0.0f) : make_float3(0.0f, 1.0f, 0.0f);
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
        std::cout << "Sun plane distance: " << distance << "\n";
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
    for (auto& vertex : bounding_box_vertices) {
        // Compute the buffer for this point
        if (vertex.distance < 0.0f) {
            vertex.distance = d + abs(vertex.distance);
        }
        float buffer = vertex.distance * tan(state.params.max_sun_angle);
        // Project the point onto the sun plane
        float3 projected_point = projectOntoPlaneAtDistance(state, vertex.point, d);
        float u = dot(projected_point, sun_u);
        float v = dot(projected_point, sun_v);
        std::cout << "Projected UV: (" << u << ", " << v << ") for vertex at distance " << vertex.distance << "\n";
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
    //u_min = 2 * u_min;
    //u_max = 2 * u_max;
    //v_min = 2 * v_min;
    //v_max = 2 * v_max;
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

void parseGeometryCSV(SoltraceState &state, const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::string line;
    // Skip the header line if the CSV has headers
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string value;

        float3 v1, v2, anchor;

        // Read v1 (x, y, z)
        std::getline(ss, value, ',');
        v1.x = std::stof(value);
        std::getline(ss, value, ',');
        v1.y = std::stof(value);
        std::getline(ss, value, ',');
        v1.z = std::stof(value);

        // Read v2 (x, y, z)
        std::getline(ss, value, ',');
        v2.x = std::stof(value);
        std::getline(ss, value, ',');
        v2.y = std::stof(value);
        std::getline(ss, value, ',');
        v2.z = std::stof(value);

        // Read anchor (x, y, z)
        std::getline(ss, value, ',');
        anchor.x = std::stof(value);
        std::getline(ss, value, ',');
        anchor.y = std::stof(value);
        std::getline(ss, value, ',');
        anchor.z = std::stof(value);

        // Create the parallelogram and add it to the heliostats list
        GeometryData::Parallelogram heliostat = { v1, v2, anchor };
        state.heliostats.push_back(heliostat);
    }

    file.close();

    std::cout << "Parsed " << state.heliostats.size() << " heliostats from the CSV file.\n";
}

// Build a GAS (Geometry Acceleration Structure) for the scene.
static void buildGas(
    const SoltraceState& state,
    const OptixAccelBuildOptions& accel_options,
    const OptixBuildInput& build_input,
    OptixTraversableHandle& gas_handle,
    CUdeviceptr& d_gas_output_buffer
) {

    OptixAccelBufferSizes gas_buffer_sizes;     // Holds required sizes for temp and output buffers.
    CUdeviceptr d_temp_buffer_gas;              // Temporary buffer for building the GAS.

    // Query the memory usage required for building the GAS.
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        state.context,
        &accel_options,
        &build_input,
        1,
        &gas_buffer_sizes));

    // Allocate memory for the temporary buffer on the device.
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_temp_buffer_gas),
        gas_buffer_sizes.tempSizeInBytes));

    // Non-compacted output and size of compacted GAS
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
        compactedSizeOffset + 8
    ));

    // Emit property to store the compacted GAS size.
    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    // Build the GAS.
    OPTIX_CHECK(optixAccelBuild(
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
        1));                                           // Number of emitted properties

    CUDA_CHECK(cudaFree((void*)d_temp_buffer_gas));

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    // If the compacted GAS size is smaller, allocate a smaller buffer and compact the GAS
    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), compacted_gas_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(state.context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle));

        CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
    }
    else
    {
        d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}

// Build custom primitives
void createGeometry(SoltraceState& state) {

    // Create OptixAabb list dynamically by looping through all heliostats
    std::vector<OptixAabb> aabbs;
    for (const auto& heliostat : state.heliostats) {
        OptixAabb aabb = parallelogram_bound(heliostat.v1, heliostat.v2, heliostat.anchor);
        aabbs.push_back(aabb);
    }

    // Optionally add other objects like the receiver
    OptixAabb receiver_aabb = parallelogram_bound(state.receiver.v1, state.receiver.v2, state.receiver.anchor);
    aabbs.push_back(receiver_aabb);

    // Container to store all vertices
    std::vector<soltrace::BoundingBoxVertex> bounding_box_vertices;

    // Collect all vertices from AABBs
    collectAllAABBVertices(aabbs.data(), static_cast<int>(aabbs.size()), bounding_box_vertices);

    // Pass the vertices to computeBoundingSunBox
    computeBoundingSunBox(state, bounding_box_vertices);

    // Allocate memory on the device for the AABB array.
    CUdeviceptr d_aabb;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabb
        ), OBJ_COUNT * sizeof(OptixAabb)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_aabb),
        &aabbs,
        OBJ_COUNT * sizeof(OptixAabb),
        cudaMemcpyHostToDevice
    ));

    // Define flags for each AABB. These flags configure how OptiX handles each geometry during traversal.
    uint32_t aabb_input_flags[] = {
        /* flags for heliostat 1 */
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
        /* flags for heliostat 2 */
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
        /* flags for heliostat 3 */
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
        /* flag for receiver */
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
    };

    // Define shader binding table (SBT) indices for each geometry dynamically
    std::vector<uint32_t> sbt_index;
    for (size_t i = 0; i < aabbs.size(); ++i) {
        sbt_index.push_back(static_cast<uint32_t>(i));
    }

    // Allocate GPU memory for the SBT indices
    CUdeviceptr d_sbt_index;
    size_t sbt_index_size = sbt_index.size() * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sbt_index), sbt_index_size));

    // Copy SBT indices to GPU
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_sbt_index), sbt_index.data(), sbt_index_size, cudaMemcpyHostToDevice));


    // Configure the input for the GAS build process.
    OptixBuildInput aabb_input = {};
    aabb_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabb_input.customPrimitiveArray.aabbBuffers = &d_aabb;
    aabb_input.customPrimitiveArray.flags = aabb_input_flags;
    aabb_input.customPrimitiveArray.numSbtRecords = OBJ_COUNT;
    aabb_input.customPrimitiveArray.numPrimitives = OBJ_COUNT;
    aabb_input.customPrimitiveArray.sbtIndexOffsetBuffer = d_sbt_index;
    aabb_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    aabb_input.customPrimitiveArray.primitiveIndexOffset = 0;

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

    CUDA_CHECK(cudaFree((void*)d_aabb));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_sbt_index)));
}
