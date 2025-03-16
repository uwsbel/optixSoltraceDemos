#include "SolTrSystem.h"
#include <cfloat> 
#include <cuda/GeometryDataST.h>
#include <cuda/Soltrace.h>
#include <SoltraceState.h>
#include <optix.h>
#include <vector>

// TODO: Sun program ask Allie: 
// in computeSunFrame, axis compared to noramlized sun vector, not sun vector 
// Do we need sun vector or normalized sun vector? if it's the latter, we should store normalized one in the state 
// do we need struct like ProjectedPoint? does it get used in other places? 
// when generating sun plane, do we include receiver or not? 

// this one need to be refactored
// Cuda/optix ones has to be separated from non-cuda ones
// note that the sun plane is part of the pre-processing
// should be done after the geometry is created and before the launch? 

// PRE-PROCESSING SCENE GEOMETRY AND SUN DEFINITION
// Compute the sun's coordinate frame
static void computeSunFrame(SoltraceState& state, float3& sun_u, float3& sun_v) {
    float3 sun_vector_hat = normalize(state.params.sun_vector);
    float3 axis = (abs(state.params.sun_vector.x) < 0.9f) ? make_float3(1.0f, 0.0f, 0.0f) : make_float3(0.0f, 1.0f, 0.0f);
    // TODO: cross need to follow left
    sun_u = normalize(cross(axis, sun_vector_hat));
    sun_v = normalize(cross(sun_vector_hat, sun_u));
}

// Find the distance to the closest object along the sun vector
static float computeSunPlaneDistance(SoltraceState& state, std::vector<soltrace::BoundingBoxVertex>& bounding_box_vertices) {
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
static float3 projectOntoPlaneAtDistance(SoltraceState& state, const float3& point, float d) {
    float3 sun_vector_hat = normalize(state.params.sun_vector);
    float3 plane_center = d * sun_vector_hat;
    return point - dot(point - plane_center, sun_vector_hat) * sun_vector_hat;
}

// Function to compute all 8 vertices of an AABB
static void getAABBVertices(const OptixAabb& aabb, std::vector<soltrace::BoundingBoxVertex>& vertices) {
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
static void collectAllAABBVertices(const std::vector<OptixAabb>& aabbs,
    std::vector<soltrace::BoundingBoxVertex>& allVertices) {
    for (int i = 0; i < aabbs.size(); ++i) {
        getAABBVertices(aabbs[i], allVertices);
    }
}

// Compute the bounding box of all projected objects onto sun plane
void geometryManager::compute_sun_plane() {

    std::vector<soltrace::BoundingBoxVertex> bounding_box_vertices;
    int count = m_aabb_list.size();

    collectAllAABBVertices(m_aabb_list, bounding_box_vertices);


    float d = computeSunPlaneDistance(m_state, bounding_box_vertices);

    float3 sun_u, sun_v;
    computeSunFrame(m_state, sun_u, sun_v);

    // Project points onto the sun plane
    std::vector<soltrace::ProjectedPoint> projected_points;
    for (const auto& vertex : bounding_box_vertices) {
        // Compute the buffer for this point
        float buffer = vertex.distance * m_state.params.max_sun_angle;
        // Project the point onto the sun plane
        float3 projected_point = projectOntoPlaneAtDistance(m_state, vertex.point, d);
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
        float3 global_vertex = vertex.x * sun_u + vertex.y * sun_v + d * normalize(m_state.params.sun_vector);
        sun_bounds_global_frame.push_back(global_vertex);
    }

    m_state.params.sun_v0 = sun_bounds_global_frame[0];   // bottom-left
    m_state.params.sun_v1 = sun_bounds_global_frame[1];   // bottom-right
    m_state.params.sun_v2 = sun_bounds_global_frame[2];   // top-right
    m_state.params.sun_v3 = sun_bounds_global_frame[3];   // top-left
}


// generate aabb list given all the elements 
void geometryManager::populate_aabb_list(const std::vector<std::shared_ptr<Element>>& element_list) {
	m_aabb_list.clear(); // Clear the existing AABB list

    for (const auto& element : element_list) {
		// Get the geometry data from the element
		element->toDeviceGeometryData();
		
		// Create an OptixAabb from the geometry data
		OptixAabb aabb;
        float3 m_min;
        float3 m_max;

        // now we compute AABB based on its aperture and surface type

        if (element->get_aperture_type() == ApertureType::RECTANGLE) {

			element->toDeviceGeometryData(); // Ensure the geometry data is computed  
			float3 anchor = element->get_aperture()->get_anchor(); // anchor point
			float3 v1 = element->get_aperture()->get_v1(); // first vector
			float3 v2 = element->get_aperture()->get_v2(); // second vector


            float3 p00 = anchor;                 // Lower-left corner
            float3 p01 = anchor + v1;           // Lower-right corner
            float3 p10 = anchor + v2;           // Upper-left corner
            float3 p11 = anchor + v1 + v2;     // Upper-right 

            m_min = fminf(fminf(p00, p01), fminf(p10, p11));
            m_max = fmaxf(fmaxf(p00, p01), fmaxf(p10, p11));
		}
        else if (element->get_aperture_type() == ApertureType::CIRCLE) {

            /********* TODO **********/
			m_min = make_float3(0.0f, 0.0f, 0.0f); // Initialize min to a large value
			m_max = make_float3(0.0f, 0.0f, 0.0f); // Initialize max to a small value

        }


        aabb.minX = m_min.x;
        aabb.minY = m_min.y;
        aabb.minZ = m_min.z;

        aabb.maxX = m_max.x;
        aabb.maxY = m_max.y;
        aabb.maxZ = m_max.z;

		// Add the AABB to the list
		m_aabb_list.push_back(aabb);
	}
}
// TODOs: this need to go inside populuate_aabb_list//
// compute the AABB for a cylinder
//OptixAabb ComputeCylinderYBound(GeometryData::Cylinder_Y cyl)
//{
//    float3 base_z = cyl.base_z;
//    float3 base_x = cyl.base_x;
//    float3 center = cyl.center;
//    float radius = cyl.radius;
//    float half_height = cyl.half_height;
//
//    // Compute the base_y axis (cross product of base_z and base_x)
//    float3 base_y = normalize(cross(base_z, base_x));
//
//    // Local corners of the cylinder in its coordinate system
//    float3 local_min = { -radius, -half_height, -radius };
//    float3 local_max = { radius, half_height, radius };
//
//    // Eight corners of the cylinder in local coordinates
//    float3 corners[] = {
//        make_float3(local_min.x, local_min.y, local_min.z),
//        make_float3(local_min.x, local_min.y, local_max.z),
//        make_float3(local_min.x, local_max.y, local_min.z),
//        make_float3(local_min.x, local_max.y, local_max.z),
//        make_float3(local_max.x, local_min.y, local_min.z),
//        make_float3(local_max.x, local_min.y, local_max.z),
//        make_float3(local_max.x, local_max.y, local_min.z),
//        make_float3(local_max.x, local_max.y, local_max.z),
//    };
//
//    // Transform corners to world coordinates and find the min/max bounds
//    float3 global_min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
//    float3 global_max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
//
//    for (const auto& corner : corners)
//    {
//        // Transform the corner from local to global coordinates
//        float3 world_corner = center
//            + corner.x * base_x
//            + corner.y * base_y
//            + corner.z * base_z;
//
//        // Update the global AABB bounds
//        global_min = fminf(global_min, world_corner);
//        global_max = fmaxf(global_max, world_corner);
//    }
//
//    printf("Cylinder AABB: (%f, %f, %f) - (%f, %f, %f)\n",
//        global_min.x, global_min.y, global_min.z,
//        global_max.x, global_max.y, global_max.z);
//
//    // Return the global AABB
//    return { global_min.x, global_min.y, global_min.z, global_max.x, global_max.y, global_max.z };
//}


// Build a GAS (Geometry Acceleration Structure) for the scene.
static void buildGas(
    const SoltraceState& state,
    const OptixAccelBuildOptions& accel_options,
    const OptixBuildInput& build_input,
    OptixTraversableHandle& gas_handle,
    CUdeviceptr& d_gas_output_buffer) {

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

void geometryManager::create_geometries() {


	int obj_count = m_aabb_list.size();
    std::cout << "size of aabb_list: " << obj_count << std::endl;

    // Allocate memory on the device for the AABB array.
    CUdeviceptr d_aabb;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabb), obj_count * sizeof(OptixAabb)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_aabb), 
               m_aabb_list.data(), 
               obj_count * sizeof(OptixAabb),
               cudaMemcpyHostToDevice));

    // initialize aabb_input_flags vector
    std::vector<uint32_t> aabb_input_flags(obj_count);
    for (int i = 0; i < obj_count; i++) {
        aabb_input_flags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    }

    // Define shader binding table (SBT) indices for each geometry.
    // TODO: for large number of elements with less number of types
	// we do not need sbt to be the same size as the aabb_list 
    std::vector<uint32_t> sbt_index(obj_count);
    for (int i = 0; i < obj_count; i++) {
        sbt_index[i] = i;
    }

    // host to device
    CUdeviceptr    d_sbt_index;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sbt_index), obj_count * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_sbt_index),
                         sbt_index.data(),
                         obj_count * sizeof(uint32_t), 
                         cudaMemcpyHostToDevice));

    // Configure the input for the GAS build process.
    OptixBuildInput aabb_input = {};
    aabb_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabb_input.customPrimitiveArray.aabbBuffers = &d_aabb;
    aabb_input.customPrimitiveArray.flags = aabb_input_flags.data();
    aabb_input.customPrimitiveArray.numSbtRecords = obj_count;
    aabb_input.customPrimitiveArray.numPrimitives = obj_count;
    aabb_input.customPrimitiveArray.sbtIndexOffsetBuffer = d_sbt_index;
    aabb_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    aabb_input.customPrimitiveArray.primitiveIndexOffset = 0;

    // Set up acceleration structure (AS) build options.
    OptixAccelBuildOptions accel_options = {
        OPTIX_BUILD_FLAG_ALLOW_COMPACTION,  // buildFlags. Enable compaction to reduce memory usage.
        OPTIX_BUILD_OPERATION_BUILD         // operation. Build a new acceleration structure (not an update).
    };

    // Build the GAS using the defined AABBs and options.
    buildGas(m_state,                        // input:  state with OptiX context.
             accel_options,                  // input:  build options.
             aabb_input,                     // input:  AABB input description.
             m_state.gas_handle,             // output: traversable handle for the GAS.
             m_state.d_gas_output_buffer);   // output: device buffer for the GAS.
    

    CUDA_CHECK(cudaFree((void*)d_aabb));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_sbt_index)));
}