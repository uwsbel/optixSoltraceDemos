#include "soltrace_system.h"
#include <cfloat> 
#include <cuda/GeometryDataST.h>
#include <cuda/Soltrace.h>
#include <soltrace_state.h>
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
static void computeSunFrame(SoltraceState& state, float3 sun_vector, float3& sun_u, float3& sun_v) {
    float3 axis = (abs(sun_vector.x) < 0.9f) ? make_float3(1.0f, 0.0f, 0.0f) : make_float3(0.0f, 1.0f, 0.0f);
    // TODO: cross need to follow left
    sun_u = normalize(cross(axis, sun_vector));
    sun_v = normalize(cross(sun_vector, sun_u));
}

// Find the distance to the closest object along the sun vector
static float computeSunPlaneDistance(SoltraceState& state, 
                                     float3 sun_vector, 
                                     std::vector<soltrace::BoundingBoxVertex>& bounding_box_vertices) {
    // Max distance from Origin along sun vector
    float max_distance = 0.0f;
    for (auto& vertex : bounding_box_vertices) {
        float distance = abs(dot(vertex.point, sun_vector));
        vertex.distance = distance;
        if (distance > max_distance) {
            max_distance = distance;
        }
    }
    return max_distance;
}

// Project a point onto the plane at distance d along the sun vector
static float3 projectOntoPlaneAtDistance(SoltraceState& state, float3& sun_vector, const float3& point, float d) {
    float3 plane_center = d * sun_vector;
    return point - dot(point - plane_center, sun_vector) * sun_vector;
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
void GeometryManager::compute_sun_plane(LaunchParams& params) {

    std::vector<soltrace::BoundingBoxVertex> bounding_box_vertices;
    int count = m_aabb_list.size();



    collectAllAABBVertices(m_aabb_list, bounding_box_vertices);


    float d = computeSunPlaneDistance(m_state, params.sun_vector, bounding_box_vertices);

    float3 sun_u, sun_v;
    computeSunFrame(m_state, params.sun_vector, sun_u, sun_v);

    // Project points onto the sun plane
    std::vector<soltrace::ProjectedPoint> projected_points;
    for (const auto& vertex : bounding_box_vertices) {
        // Compute the buffer for this point
        float buffer = vertex.distance * tan(params.max_sun_angle);
        // Project the point onto the sun plane
        float3 projected_point = projectOntoPlaneAtDistance(m_state, params.sun_vector, vertex.point, d);
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
        float3 global_vertex = vertex.x * sun_u + vertex.y * sun_v + d * normalize(params.sun_vector);
        sun_bounds_global_frame.push_back(global_vertex);
    }

    params.sun_v0 = sun_bounds_global_frame[0];   // bottom-left
    params.sun_v1 = sun_bounds_global_frame[1];   // bottom-right
    params.sun_v2 = sun_bounds_global_frame[2];   // top-right
    params.sun_v3 = sun_bounds_global_frame[3];   // top-left
}


// generate aabb list given all the elements 
void GeometryManager::populate_aabb_list(const std::vector<std::shared_ptr<Element>>& element_list) {
	m_aabb_list.clear(); // Clear the existing AABB list

    for (const auto& element : element_list) {
		
		// Create an OptixAabb from the geometry data
		OptixAabb aabb;
        float3 m_min;
        float3 m_max;

        if (element->get_aperture_type() == ApertureType::CIRCLE) {

            /********* TODO **********/
			m_min = make_float3(0.0f, 0.0f, 0.0f); // Initialize min to a large value
			m_max = make_float3(0.0f, 0.0f, 0.0f); // Initialize max to a small value

        }


        if (element->get_aperture_type() == ApertureType::RECTANGLE) {
            element->compute_bounding_box();
			m_min.x = element->get_lower_bounding_box()[0];
			m_min.y = element->get_lower_bounding_box()[1];
			m_min.z = element->get_lower_bounding_box()[2];

			m_max.x = element->get_upper_bounding_box()[0];
			m_max.y = element->get_upper_bounding_box()[1];
			m_max.z = element->get_upper_bounding_box()[2];
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

void GeometryManager::create_geometries(const std::vector<std::shared_ptr<Element>>& element_list) {


	int obj_count = m_aabb_list.size();

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
    for (int i = 0; i < obj_count - 1; i++) {
		// find the corresponding shader binding table index (OpticalEntityType) when going through the list
        
		Element my_element = *element_list[i];

		if (my_element.get_aperture_type() == ApertureType::RECTANGLE) {

            if (my_element.get_surface_type() == SurfaceType::PARABOLIC) {
                sbt_index[i] = static_cast<uint32_t>(OpticalEntityType::RECTANGLE_PARABOLIC_MIRROR);
            }
            else if (my_element.get_surface_type() == SurfaceType::FLAT) {
                sbt_index[i] = static_cast<uint32_t>(OpticalEntityType::RECTANGLE_FLAT_MIRROR);
            }
            else {
                // print error message 
				std::cerr << "Error: Unknown surface type for element " << i << std::endl;
            }
		}
        else {
            std::cerr << "Error: Unknown surface type for element " << i << std::endl;
		}
    }

	// add receiver sbt_index
	int i = obj_count - 1;
	Element my_element = *element_list[i];

	if (my_element.get_surface_type() == SurfaceType::FLAT) {
		sbt_index[i] = static_cast<uint32_t>(OpticalEntityType::RECTANGLE_FLAT_RECEIVER);
	}

	if (my_element.get_surface_type() == SurfaceType::CYLINDER) {
        sbt_index[i] = static_cast<uint32_t>(OpticalEntityType::CYLINDRICAL_RECEIVER);
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